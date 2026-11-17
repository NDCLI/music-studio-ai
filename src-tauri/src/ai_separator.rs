//! Stage 5: Local AI Stem Separation Engine (`ort` load-dynamic + `realfft` + `ndarray`).
//! Implements:
//! 1. Exact Stereo STFT / iSTFT with Hann windowing and COLA normalization.
//! 2. Memory-safe chunking (`dim_t` frames) with 50% Windowed Overlap-Add (prevents RAM overflow).
//! 3. ONNX Runtime (`load-dynamic`) Session Management with DirectML GPU + CPU fallback.
//! 4. 3-Stage Stem Separation Pipeline:
//!    - Stage 1: Instrumental vs. All Vocals (`UVR-MDX-NET-Inst_HQ_4.onnx` / `Mel-Band-RoFormer`)
//!    - Stage 2: Backing Vocal vs. Lead Vocal (`UVR_MDXNET_KARA_2.onnx` / `Mel-Band-RoFormer-Karaoke`)
//!    - Stage 3: Dry Lead Vocal vs. Reverb Tail (`Reverb_HQ_By_FoxJoy.onnx`)

use std::f32::consts::PI;
use std::path::Path;

use ndarray::{Array4, ArrayViewD};
use ort::environment::Environment;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::audio_io::{StereoAudioBuffer, STUDIO_SAMPLE_RATE};
use crate::model_manager::discover_onnxruntime_library;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionDevice {
    DirectMl { device_id: i32 },
    Cpu { intra_threads: usize },
}

impl Default for ExecutionDevice {
    fn default() -> Self {
        Self::DirectMl { device_id: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StftModelConfig {
    pub model_filename: String,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub dim_f: usize,
    pub dim_t: usize,
    pub compensate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationPipelineConfig {
    pub stage1_inst_model: StftModelConfig,
    pub stage2_karaoke_model: StftModelConfig,
    pub stage3_dereverb_model: StftModelConfig,
}

impl SeparationPipelineConfig {
    /// Returns exact STFT parameters for the fast MDX-Net 3-stage pipeline.
    pub fn default_mdx_pipeline() -> Self {
        Self {
            stage1_inst_model: StftModelConfig {
                model_filename: "UVR-MDX-NET-Inst_HQ_4.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 5120,
                hop_length: 1024,
                dim_f: 2560,
                dim_t: 256,
                compensate: 1.019,
            },
            stage2_karaoke_model: StftModelConfig {
                model_filename: "UVR_MDXNET_KARA_2.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 5120,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 256,
                compensate: 1.065,
            },
            stage3_dereverb_model: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 256,
                compensate: 1.028,
            },
        }
    }

    /// Returns STFT parameters for high-precision RoFormer 3-stage pipeline.
    pub fn pro_roformer_pipeline() -> Self {
        Self {
            stage1_inst_model: StftModelConfig {
                model_filename: "mel_band_roformer_kim_ft_unwa.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 4096,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 352,
                compensate: 1.0,
            },
            stage2_karaoke_model: StftModelConfig {
                model_filename: "mel_band_roformer_karaoke_aufr33_viperx.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 4096,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 352,
                compensate: 1.0,
            },
            stage3_dereverb_model: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 256,
                compensate: 1.028,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeparatedStems {
    pub lead_vocal: StereoAudioBuffer,
    pub backing_vocal: StereoAudioBuffer,
    pub instrumental: StereoAudioBuffer,
    pub reverb_tail: StereoAudioBuffer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationProgress {
    pub stage_index: u8,
    pub stage_name: String,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub overall_percent: f32,
}

/// Initializes the process-global ONNX Runtime environment using `load-dynamic`
/// (preferring a Microsoft-signed local `onnxruntime.dll` if discovered).
pub fn init_ort_environment(custom_dll_path: Option<&Path>) -> Result<Environment, String> {
    let resolved_dll = custom_dll_path
        .map(|p| p.to_path_buf())
        .or_else(discover_onnxruntime_library);

    let builder = match resolved_dll {
        Some(path) => ort::init_from(&path)
            .map_err(|e| format!("Failed to dynamically load ONNX Runtime at '{}': {}", path.display(), e))?,
        None => ort::init(),
    };

    builder
        .with_name("music-studio-ai")
        .with_telemetry(false)
        .build()
        .map_err(|e| format!("Failed to build ONNX Runtime environment: {}", e))
}

/// Creates an ONNX inference session configured for DirectML GPU acceleration with CPU fallback.
pub fn create_mdx_session<P: AsRef<Path>>(
    env: &Environment,
    model_path: P,
    device: ExecutionDevice,
) -> Result<(Session, &'static str), String> {
    let path_ref = model_path.as_ref();
    if !path_ref.is_file() {
        return Err(format!("ONNX model file not found: '{}'", path_ref.display()));
    }

    match device {
        ExecutionDevice::DirectMl { device_id } => {
            let dml_attempt = Session::builder(env)
                .map_err(|e| e.to_string())?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?
                // DirectML recommends disabling parallel execution & memory pattern for graph stability
                .with_parallel_execution(false)
                .map_err(|e| e.to_string())?
                .with_memory_pattern(false)
                .map_err(|e| e.to_string())?
                .with_execution_providers([
                    ep::DirectML::default().with_device_id(device_id).build(),
                    ep::CPU::default().build(),
                ])
                .map_err(|e| e.to_string())?
                .commit_from_file(path_ref);

            if let Ok(session) = dml_attempt {
                return Ok((session, "DirectML GPU (with CPU fallback)"));
            }

            // Fallback to pure CPU EP if DirectML could not bind on this host
            let cpu_session = Session::builder(env)
                .map_err(|e| e.to_string())?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?
                .with_execution_providers([ep::CPU::default().build()])
                .map_err(|e| e.to_string())?
                .commit_from_file(path_ref)
                .map_err(|e| format!("Failed to load ONNX session '{}': {}", path_ref.display(), e))?;

            Ok((cpu_session, "CPU Execution Provider (DirectML unavailable)"))
        }
        ExecutionDevice::Cpu { intra_threads } => {
            let session = Session::builder(env)
                .map_err(|e| e.to_string())?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?
                .with_intra_threads(intra_threads.max(1))
                .map_err(|e| e.to_string())?
                .with_execution_providers([ep::CPU::default().build()])
                .map_err(|e| e.to_string())?
                .commit_from_file(path_ref)
                .map_err(|e| format!("Failed to load ONNX session '{}': {}", path_ref.display(), e))?;

            Ok((session, "CPU Multi-Threaded Execution Provider"))
        }
    }
}

/// Generates a periodic Hann window of length `size`.
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Computes the number of audio samples corresponding to one `dim_t` STFT chunk.
pub fn chunk_sample_length(cfg: &StftModelConfig) -> usize {
    (cfg.dim_t.saturating_sub(1)) * cfg.hop_length
}

/// Converts a centered stereo chunk of length `chunk_sample_length(cfg)` into an ONNX tensor
/// with shape `[1, 4, dim_f, dim_t]` containing `[Re_L, Im_L, Re_R, Im_R]`.
pub fn stft_stereo_chunk(
    left_chunk: &[f32],
    right_chunk: &[f32],
    cfg: &StftModelConfig,
) -> Array4<f32> {
    let chunk_len = chunk_sample_length(cfg);
    let pad = cfg.n_fft / 2;
    let padded_len = chunk_len + 2 * pad;

    let mut pad_l = vec![0.0_f32; padded_len];
    let mut pad_r = vec![0.0_f32; padded_len];
    let copy_len = chunk_len.min(left_chunk.len()).min(right_chunk.len());
    pad_l[pad..pad + copy_len].copy_from_slice(&left_chunk[..copy_len]);
    pad_r[pad..pad + copy_len].copy_from_slice(&right_chunk[..copy_len]);

    let window = hann_window(cfg.n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(cfg.n_fft);

    let mut frame_buf = vec![0.0_f32; cfg.n_fft];
    let mut spectrum = r2c.make_output_vec();
    let valid_bins = cfg.dim_f.min(spectrum.len());

    let mut tensor = Array4::<f32>::zeros((1, 4, cfg.dim_f, cfg.dim_t));

    for t in 0..cfg.dim_t {
        let start = t * cfg.hop_length;
        if start + cfg.n_fft > padded_len {
            break;
        }

        // Left channel FFT
        for i in 0..cfg.n_fft {
            frame_buf[i] = pad_l[start + i] * window[i];
        }
        let _ = r2c.process(&mut frame_buf, &mut spectrum);
        for f in 0..valid_bins {
            tensor[[0, 0, f, t]] = spectrum[f].re;
            tensor[[0, 1, f, t]] = spectrum[f].im;
        }

        // Right channel FFT
        for i in 0..cfg.n_fft {
            frame_buf[i] = pad_r[start + i] * window[i];
        }
        let _ = r2c.process(&mut frame_buf, &mut spectrum);
        for f in 0..valid_bins {
            tensor[[0, 2, f, t]] = spectrum[f].re;
            tensor[[0, 3, f, t]] = spectrum[f].im;
        }
    }

    tensor
}

/// Reconstructs a stereo time-domain chunk of length `chunk_sample_length(cfg)` from
/// an `[1, 4, dim_f, dim_t]` complex spectrogram tensor using Hann synthesis window + COLA normalization.
pub fn istft_stereo_chunk(
    tensor: ArrayViewD<'_, f32>,
    cfg: &StftModelConfig,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let shape = tensor.shape();
    if shape.len() != 4 || shape[1] != 4 {
        return Err(format!(
            "Unexpected ONNX output tensor shape {:?}; expected [1, 4, dim_f, dim_t]",
            shape
        ));
    }

    let out_dim_f = shape[2].min(cfg.dim_f);
    let out_dim_t = shape[3].min(cfg.dim_t);

    let chunk_len = chunk_sample_length(cfg);
    let pad = cfg.n_fft / 2;
    let padded_len = chunk_len + 2 * pad;

    let window = hann_window(cfg.n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(cfg.n_fft);
    let num_bins = cfg.n_fft / 2 + 1;
    let valid_bins = out_dim_f.min(num_bins);

    let mut spectrum = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
    let mut time_frame = c2r.make_output_vec();

    let mut acc_l = vec![0.0_f32; padded_len];
    let mut acc_r = vec![0.0_f32; padded_len];
    let mut win_norm = vec![0.0_f32; padded_len];
    let fft_scale = 1.0 / (cfg.n_fft as f32);

    for t in 0..out_dim_t {
        let start = t * cfg.hop_length;
        if start + cfg.n_fft > padded_len {
            break;
        }

        // Left channel iFFT
        spectrum.fill(Complex::new(0.0, 0.0));
        for f in 0..valid_bins {
            spectrum[f] = Complex::new(tensor[[0, 0, f, t]], tensor[[0, 1, f, t]]);
        }
        spectrum[0].im = 0.0;
        if num_bins > 1 {
            spectrum[num_bins - 1].im = 0.0;
        }
        let _ = c2r.process(&mut spectrum, &mut time_frame);
        for i in 0..cfg.n_fft {
            let w = window[i];
            acc_l[start + i] += time_frame[i] * fft_scale * w;
            win_norm[start + i] += w * w;
        }

        // Right channel iFFT
        spectrum.fill(Complex::new(0.0, 0.0));
        for f in 0..valid_bins {
            spectrum[f] = Complex::new(tensor[[0, 2, f, t]], tensor[[0, 3, f, t]]);
        }
        spectrum[0].im = 0.0;
        if num_bins > 1 {
            spectrum[num_bins - 1].im = 0.0;
        }
        let _ = c2r.process(&mut spectrum, &mut time_frame);
        for i in 0..cfg.n_fft {
            acc_r[start + i] += time_frame[i] * fft_scale * window[i];
        }
    }

    let mut out_l = Vec::with_capacity(chunk_len);
    let mut out_r = Vec::with_capacity(chunk_len);
    for i in pad..(pad + chunk_len) {
        let denom = win_norm[i].max(1e-6);
        out_l.push((acc_l[i] / denom) * cfg.compensate);
        out_r.push((acc_r[i] / denom) * cfg.compensate);
    }

    Ok((out_l, out_r))
}

/// Runs a single MDX-Net ONNX separation model over a full-length audio buffer
/// using chunked STFT -> ONNX inference -> iSTFT + 50% Windowed Overlap-Add.
pub fn run_mdx_onnx_model<F>(
    session: &mut Session,
    audio: &StereoAudioBuffer,
    cfg: &StftModelConfig,
    mut on_chunk_progress: F,
) -> Result<StereoAudioBuffer, String>
where
    F: FnMut(usize, usize),
{
    let total_frames = audio.num_frames();
    if total_frames == 0 {
        return Ok(audio.clone());
    }

    let chunk_len = chunk_sample_length(cfg);
    let step = (chunk_len / 2).max(1); // 50% overlap between adjacent chunks
    let margin_pad = chunk_len / 2;
    let padded_total = total_frames + 2 * margin_pad;

    let mut src_l = vec![0.0_f32; padded_total];
    let mut src_r = vec![0.0_f32; padded_total];
    src_l[margin_pad..margin_pad + total_frames].copy_from_slice(&audio.left[..total_frames]);
    src_r[margin_pad..margin_pad + total_frames].copy_from_slice(&audio.right[..total_frames]);

    let mut out_l = vec![0.0_f32; padded_total];
    let mut out_r = vec![0.0_f32; padded_total];
    let mut weight_acc = vec![0.0_f32; padded_total];

    // Raised-cosine crossfade window across chunks so chunk borders are completely seamless
    let chunk_blend = hann_window(chunk_len);

    let mut starts = Vec::new();
    let mut pos = 0usize;
    while pos < total_frames + margin_pad {
        starts.push(pos);
        if pos + chunk_len >= padded_total {
            break;
        }
        pos += step;
    }
    let total_chunks = starts.len().max(1);

    for (idx, &start) in starts.iter().enumerate() {
        let end = (start + chunk_len).min(padded_total);
        let valid_len = end - start;

        let mut chunk_l = vec![0.0_f32; chunk_len];
        let mut chunk_r = vec![0.0_f32; chunk_len];
        chunk_l[..valid_len].copy_from_slice(&src_l[start..end]);
        chunk_r[..valid_len].copy_from_slice(&src_r[start..end]);

        let input_tensor = stft_stereo_chunk(&chunk_l, &chunk_r, cfg);
        let tensor_ref = TensorRef::from_array_view(&input_tensor)
            .map_err(|e| format!("Failed to wrap STFT tensor: {}", e))?;

        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| format!("ONNX inference failed on chunk {}/{}: {}", idx + 1, total_chunks, e))?;

        let output_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract f32 output tensor: {}", e))?;

        let (pred_l, pred_r) = istft_stereo_chunk(output_view, cfg)?;

        for i in 0..valid_len {
            let w = chunk_blend[i];
            out_l[start + i] += pred_l[i] * w;
            out_r[start + i] += pred_r[i] * w;
            weight_acc[start + i] += w;
        }

        on_chunk_progress(idx + 1, total_chunks);
    }

    let mut final_l = Vec::with_capacity(total_frames);
    let mut final_r = Vec::with_capacity(total_frames);
    for i in margin_pad..(margin_pad + total_frames) {
        let w = weight_acc[i].max(1e-6);
        final_l.push(out_l[i] / w);
        final_r.push(out_r[i] / w);
    }

    Ok(StereoAudioBuffer::new(audio.sample_rate, final_l, final_r))
}

/// Performs phase-exact subtraction (`minuend - subtrahend`) between two equal-rate stereo buffers.
pub fn subtract_stereo_buffers(
    minuend: &StereoAudioBuffer,
    subtrahend: &StereoAudioBuffer,
) -> StereoAudioBuffer {
    let frames = minuend.num_frames().min(subtrahend.num_frames());
    let mut out_l = Vec::with_capacity(frames);
    let mut out_r = Vec::with_capacity(frames);

    for i in 0..frames {
        out_l.push(minuend.left[i] - subtrahend.left[i]);
        out_r.push(minuend.right[i] - subtrahend.right[i]);
    }

    StereoAudioBuffer::new(minuend.sample_rate, out_l, out_r)
}

/// Runs the complete 3-Stage Local AI Separation Pipeline:
/// 1. Instrumental vs. All Vocals
/// 2. Backing Vocal vs. Lead Vocal
/// 3. Dry Lead Vocal vs. Reverb Tail
pub fn separate_3_stage_stems<F>(
    env: &Environment,
    models_dir: &Path,
    audio: &StereoAudioBuffer,
    pipeline: &SeparationPipelineConfig,
    device: ExecutionDevice,
    mut on_progress: F,
) -> Result<SeparatedStems, String>
where
    F: FnMut(SeparationProgress),
{
    // Stage 1: Instrumental & All Vocals
    let stage1_path = models_dir.join(&pipeline.stage1_inst_model.model_filename);
    let (mut sess1, _) = create_mdx_session(env, &stage1_path, device)?;
    let instrumental = run_mdx_onnx_model(&mut sess1, audio, &pipeline.stage1_inst_model, |c, total| {
        let pct = (c as f32 / total as f32) * 40.0;
        on_progress(SeparationProgress {
            stage_index: 1,
            stage_name: format!("Stage 1/3: {}", pipeline.stage1_inst_model.model_filename),
            chunk_index: c,
            total_chunks: total,
            overall_percent: pct,
        });
    })?;
    let all_vocals = subtract_stereo_buffers(audio, &instrumental);

    // Stage 2: Backing Vocal & Lead Vocal
    let stage2_path = models_dir.join(&pipeline.stage2_karaoke_model.model_filename);
    let (mut sess2, _) = create_mdx_session(env, &stage2_path, device)?;
    let backing_vocal = run_mdx_onnx_model(
        &mut sess2,
        &all_vocals,
        &pipeline.stage2_karaoke_model,
        |c, total| {
            let pct = 40.0 + (c as f32 / total as f32) * 35.0;
            on_progress(SeparationProgress {
                stage_index: 2,
                stage_name: format!("Stage 2/3: {}", pipeline.stage2_karaoke_model.model_filename),
                chunk_index: c,
                total_chunks: total,
                overall_percent: pct,
            });
        },
    )?;
    let lead_with_reverb = subtract_stereo_buffers(&all_vocals, &backing_vocal);

    // Stage 3: De-Reverb (Dry Lead Vocal vs. Reverb Tail)
    let stage3_path = models_dir.join(&pipeline.stage3_dereverb_model.model_filename);
    let (lead_vocal, reverb_tail) = if stage3_path.is_file() {
        let (mut sess3, _) = create_mdx_session(env, &stage3_path, device)?;
        let dry_lead = run_mdx_onnx_model(
            &mut sess3,
            &lead_with_reverb,
            &pipeline.stage3_dereverb_model,
            |c, total| {
                let pct = 75.0 + (c as f32 / total as f32) * 25.0;
                on_progress(SeparationProgress {
                    stage_index: 3,
                    stage_name: format!("Stage 3/3: {}", pipeline.stage3_dereverb_model.model_filename),
                    chunk_index: c,
                    total_chunks: total,
                    overall_percent: pct,
                });
            },
        )?;
        let tail = subtract_stereo_buffers(&lead_with_reverb, &dry_lead);
        (dry_lead, tail)
    } else {
        let silent = vec![0.0_f32; lead_with_reverb.num_frames()];
        let empty_reverb = StereoAudioBuffer::new(audio.sample_rate, silent.clone(), silent);
        on_progress(SeparationProgress {
            stage_index: 3,
            stage_name: "Stage 3/3: Skipped De-Reverb (optional model not cached)".to_string(),
            chunk_index: 1,
            total_chunks: 1,
            overall_percent: 100.0,
        });
        (lead_with_reverb, empty_reverb)
    };

    Ok(SeparatedStems {
        lead_vocal,
        backing_vocal,
        instrumental,
        reverb_tail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_istft_roundtrip_accuracy() {
        // Test exact roundtrip reconstruction when dim_f covers all Nyquist bins (n_fft / 2 + 1)
        let cfg = StftModelConfig {
            model_filename: "test.onnx".to_string(),
            sample_rate: 44_100,
            n_fft: 1024,
            hop_length: 256,
            dim_f: 513, // n_fft / 2 + 1 so no high frequencies are cropped
            dim_t: 32,
            compensate: 1.0,
        };

        let chunk_len = chunk_sample_length(&cfg);
        let mut left = Vec::with_capacity(chunk_len);
        let mut right = Vec::with_capacity(chunk_len);

        for i in 0..chunk_len {
            let t = i as f32 / 44_100.0;
            left.push(0.5 * (2.0 * PI * 440.0 * t).sin() + 0.25 * (2.0 * PI * 1320.0 * t).cos());
            right.push(0.4 * (2.0 * PI * 554.37 * t).sin() + 0.2 * (2.0 * PI * 880.0 * t).cos());
        }

        let tensor = stft_stereo_chunk(&left, &right, &cfg);
        assert_eq!(tensor.shape(), &[1, 4, 513, 32]);

        let (rec_l, rec_r) = istft_stereo_chunk(tensor.view().into_dyn(), &cfg).expect("istft ok");
        assert_eq!(rec_l.len(), chunk_len);
        assert_eq!(rec_r.len(), chunk_len);

        // Verify central samples (away from zero-padded chunk borders) reconstruct cleanly
        let mut max_err = 0.0_f32;
        for i in 1024..(chunk_len - 1024) {
            max_err = max_err.max((left[i] - rec_l[i]).abs());
            max_err = max_err.max((right[i] - rec_r[i]).abs());
        }
        assert!(max_err < 1e-4, "STFT/iSTFT roundtrip error too high: {}", max_err);
    }

    #[test]
    fn test_phase_exact_subtraction() {
        let a = StereoAudioBuffer::new(44_100, vec![0.8, -0.5, 0.3], vec![0.6, 0.2, -0.9]);
        let b = StereoAudioBuffer::new(44_100, vec![0.3, -0.2, 0.1], vec![0.2, 0.1, -0.4]);
        let diff = subtract_stereo_buffers(&a, &b);
        assert!((diff.left[0] - 0.5).abs() < 1e-6);
        assert!((diff.left[1] - (-0.3)).abs() < 1e-6);
        assert!((diff.right[2] - (-0.5)).abs() < 1e-6);
    }
}
