//! Stage 5: Local AI Stem Separation Engine (`ort` load-dynamic + `realfft` + `ndarray`).
//! Implements:
//! 1. Exact Stereo STFT / iSTFT with Hann windowing and COLA normalization.
//! 2. Memory-safe chunking (`dim_t` frames) with 50% Windowed Overlap-Add (prevents RAM overflow).
//! 3. ONNX Runtime (`load-dynamic`) Session Management with DirectML GPU + CPU fallback.
//! 4. 3-Stage Stem Separation Pipeline:
//!    - Stage 1: Instrumental vs. All Vocals (`UVR-MDX-NET-Inst_HQ_1.onnx` / `Mel-Band-RoFormer`)
//!    - Stage 2: Backing Vocal vs. Lead Vocal (`UVR_MDXNET_KARA_2.onnx` / `Mel-Band-RoFormer-Karaoke`)
//!    - Stage 3: Dry Lead Vocal vs. Reverb Tail (`Reverb_HQ_By_FoxJoy.onnx`)

use std::f32::consts::PI;
use std::path::Path;

use ndarray::{Array4, ArrayViewD};
use ort::environment::Environment;
use ort::ep;
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
    #[serde(default)]
    pub outputs_vocals: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationPipelineConfig {
    pub stage1_inst_model: StftModelConfig,
    pub stage2_karaoke_model: StftModelConfig,
    pub stage3_dereverb_model: StftModelConfig,
    #[serde(default)]
    pub enable_dereverb: bool,
}

impl SeparationPipelineConfig {
    /// Returns exact STFT parameters for the fast MDX-Net 3-stage pipeline.
    pub fn default_mdx_pipeline() -> Self {
        Self {
            stage1_inst_model: StftModelConfig {
                model_filename: "UVR-MDX-NET-Inst_HQ_1.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 6144,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 256,
                compensate: 1.019,
                outputs_vocals: false,
            },
            stage2_karaoke_model: StftModelConfig {
                model_filename: "UVR_MDXNET_KARA_2.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 5120,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 256,
                compensate: 1.065,
                outputs_vocals: false,
            },
            stage3_dereverb_model: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 512,
                compensate: 1.028,
                outputs_vocals: false,
            },
            enable_dereverb: false,
        }
    }

    /// Combo 2: UVR-MDX-NET-Inst_HQ_1 + Mel-Band RoFormer Karaoke + optional De-Reverb.
    pub fn pro_roformer_pipeline() -> Self {
        Self {
            stage1_inst_model: StftModelConfig {
                model_filename: "UVR-MDX-NET-Inst_HQ_1.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 6144,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 256,
                compensate: 1.019,
                outputs_vocals: false,
            },
            stage2_karaoke_model: StftModelConfig {
                model_filename: "mel_band_roformer_karaoke_aufr33_viperx.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 2048,
                hop_length: 441,
                dim_f: 1025,
                dim_t: 400,
                compensate: 1.0,
                // mel_band_roformer_karaoke outputs the Karaoke/Backing Vocal stem (lead removed)
                outputs_vocals: false,
            },
            stage3_dereverb_model: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 512,
                compensate: 1.028,
                outputs_vocals: false,
            },
            enable_dereverb: false,
        }
    }

    /// Combo 1: HTDemucs + Mel-Band RoFormer Karaoke + optional De-Reverb (highest accuracy).
    pub fn htdemucs_pipeline() -> Self {
        Self {
            stage1_inst_model: StftModelConfig {
                model_filename: "htdemucs.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 4096,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 256,
                compensate: 1.0,
                outputs_vocals: false,
            },
            stage2_karaoke_model: StftModelConfig {
                model_filename: "mel_band_roformer_karaoke_aufr33_viperx.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 2048,
                hop_length: 441,
                dim_f: 1025,
                dim_t: 400,
                compensate: 1.0,
                // mel_band_roformer_karaoke outputs the Karaoke/Backing Vocal stem (lead removed)
                outputs_vocals: false,
            },
            stage3_dereverb_model: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: STUDIO_SAMPLE_RATE,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 512,
                compensate: 1.028,
                outputs_vocals: false,
            },
            enable_dereverb: false,
        }
    }

    /// Hybrid Combo: HTDemucs (Stage 1 Beat) + MDX Karaoke 2 (Stage 2 Backing Vocal) + optional De-Reverb.
    pub fn htdemucs_mdx_kara_pipeline() -> Self {
        let mut cfg = Self::htdemucs_pipeline();
        cfg.stage2_karaoke_model = Self::default_mdx_pipeline().stage2_karaoke_model;
        cfg
    }
}

#[derive(Debug, Clone)]
pub struct SeparatedStems {
    pub lead_vocal: StereoAudioBuffer,
    pub backing_vocal: StereoAudioBuffer,
    pub instrumental: StereoAudioBuffer,
    pub reverb_tail: StereoAudioBuffer,
    pub execution_backend: String,
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

    if let Some(ref dll_path) = resolved_dll {
        if let Some(dll_dir) = dll_path.parent() {
            let current_path = std::env::var_os("PATH").unwrap_or_default();
            let mut paths: Vec<std::path::PathBuf> = std::env::split_paths(&current_path).collect();
            if !paths.iter().any(|p| p == dll_dir) {
                paths.insert(0, dll_dir.to_path_buf());
                if let Ok(joined) = std::env::join_paths(paths) {
                    std::env::set_var("PATH", joined);
                }
            }
        }
    }

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

fn has_cuda_provider_library() -> bool {
    let cuda_dll = if cfg!(target_os = "windows") {
        "onnxruntime_providers_cuda.dll"
    } else {
        "libonnxruntime_providers_cuda.so"
    };
    if let Some(ort_path) = discover_onnxruntime_library() {
        if let Some(dir) = ort_path.parent() {
            if dir.join(cuda_dll).is_file() {
                return true;
            }
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for entry in std::env::split_paths(&path_var) {
            if entry.join(cuda_dll).is_file() {
                return true;
            }
        }
    }
    false
}

/// Creates an ONNX inference session configured for CUDA/DirectML GPU acceleration with CPU fallback.
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
            // 1. Try NVIDIA CUDA GPU first if onnxruntime_providers_cuda.dll is present
            if has_cuda_provider_library() {
                let cuda_attempt = Session::builder(env)
                    .map_err(|e| e.to_string())?
                    .with_execution_providers([
                        ep::CUDA::default()
                            .with_device_id(device_id)
                            .build()
                            .error_on_failure(),
                    ])
                    .map_err(|e| e.to_string())?
                    .commit_from_file(path_ref);

                match cuda_attempt {
                    Ok(session) => return Ok((session, "CUDA GPU")),
                    Err(_e) => {
                        #[cfg(test)]
                        println!("CUDA EP registration error: {}", _e);
                    }
                }
            }

            // 2. Fallback to DirectML GPU
            let dml_attempt = Session::builder(env)
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

            // 3. Fallback to pure multi-threaded CPU EP if GPU could not bind on this host
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            let cpu_session = Session::builder(env)
                .map_err(|e| e.to_string())?
                .with_intra_threads(threads)
                .map_err(|e| e.to_string())?
                .with_execution_providers([ep::CPU::default().build()])
                .map_err(|e| e.to_string())?
                .commit_from_file(path_ref)
                .map_err(|e| format!("Failed to load ONNX session '{}': {}", path_ref.display(), e))?;

            Ok((cpu_session, "CPU Execution Provider (GPU unavailable)"))
        }
        ExecutionDevice::Cpu { intra_threads } => {
            let auto_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            let threads = if intra_threads == 0 {
                auto_threads
            } else {
                intra_threads.max(1)
            };
            let session = Session::builder(env)
                .map_err(|e| e.to_string())?
                .with_intra_threads(threads)
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

/// Runs HTDemucs ONNX (`[1, 2, 343980]` -> `[1, 4, 2, 343980]`) with 50% Hann overlap-add.
/// Sums Demucs stems 0..3 (`drums + bass + other`) as Instrumental and stem 3 as Vocals.
pub fn run_htdemucs_onnx_model<F>(
    session: &mut Session,
    audio: &StereoAudioBuffer,
    mut on_chunk_progress: F,
) -> Result<(StereoAudioBuffer, StereoAudioBuffer), String>
where
    F: FnMut(usize, usize),
{
    const HTDEMUCS_CHUNK: usize = 343_980;
    let total_frames = audio.num_frames();
    if total_frames == 0 {
        return Ok((audio.clone(), audio.clone()));
    }

    let step = HTDEMUCS_CHUNK / 2;
    let margin_pad = HTDEMUCS_CHUNK / 2;
    let padded_total = total_frames + 2 * margin_pad;

    let mut src_l = vec![0.0_f32; padded_total];
    let mut src_r = vec![0.0_f32; padded_total];
    src_l[margin_pad..margin_pad + total_frames].copy_from_slice(&audio.left[..total_frames]);
    src_r[margin_pad..margin_pad + total_frames].copy_from_slice(&audio.right[..total_frames]);

    let mut inst_l = vec![0.0_f32; padded_total];
    let mut inst_r = vec![0.0_f32; padded_total];
    let mut voc_l = vec![0.0_f32; padded_total];
    let mut voc_r = vec![0.0_f32; padded_total];
    let mut weight_acc = vec![0.0_f32; padded_total];

    let chunk_blend = hann_window(HTDEMUCS_CHUNK);

    let mut starts = Vec::new();
    let mut pos = 0usize;
    while pos < total_frames + margin_pad {
        starts.push(pos);
        if pos + HTDEMUCS_CHUNK >= padded_total {
            break;
        }
        pos += step;
    }
    let total_chunks = starts.len().max(1);

    for (idx, &start) in starts.iter().enumerate() {
        let end = (start + HTDEMUCS_CHUNK).min(padded_total);
        let valid_len = end - start;

        let mut input_tensor = ndarray::Array3::<f32>::zeros((1, 2, HTDEMUCS_CHUNK));
        for i in 0..valid_len {
            input_tensor[[0, 0, i]] = src_l[start + i];
            input_tensor[[0, 1, i]] = src_r[start + i];
        }

        let tensor_ref = TensorRef::from_array_view(&input_tensor)
            .map_err(|e| format!("Failed to wrap HTDemucs tensor: {}", e))?;
        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| format!("HTDemucs inference failed on chunk {}/{}: {}", idx + 1, total_chunks, e))?;

        let out_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract HTDemucs output tensor: {}", e))?;

        for i in 0..valid_len {
            let w = chunk_blend[i];
            let il = out_view[[0, 0, 0, i]] + out_view[[0, 1, 0, i]] + out_view[[0, 2, 0, i]];
            let ir = out_view[[0, 0, 1, i]] + out_view[[0, 1, 1, i]] + out_view[[0, 2, 1, i]];
            let vl = out_view[[0, 3, 0, i]];
            let vr = out_view[[0, 3, 1, i]];
            inst_l[start + i] += il * w;
            inst_r[start + i] += ir * w;
            voc_l[start + i] += vl * w;
            voc_r[start + i] += vr * w;
            weight_acc[start + i] += w;
        }

        on_chunk_progress(idx + 1, total_chunks);
    }

    let mut final_inst_l = Vec::with_capacity(total_frames);
    let mut final_inst_r = Vec::with_capacity(total_frames);
    let mut final_voc_l = Vec::with_capacity(total_frames);
    let mut final_voc_r = Vec::with_capacity(total_frames);
    for i in margin_pad..(margin_pad + total_frames) {
        let w = weight_acc[i].max(1e-6);
        final_inst_l.push(inst_l[i] / w);
        final_inst_r.push(inst_r[i] / w);
        final_voc_l.push(voc_l[i] / w);
        final_voc_r.push(voc_r[i] / w);
    }

    Ok((
        StereoAudioBuffer::new(audio.sample_rate, final_inst_l, final_inst_r),
        StereoAudioBuffer::new(audio.sample_rate, final_voc_l, final_voc_r),
    ))
}

/// Runs `mel_band_roformer_karaoke_aufr33_viperx.onnx` (`stft_features [1, T, 4100]` -> `mask [1, 1, 2050, T, 2]`)
/// with `n_fft = 2048`, `hop_length = 441` and 50% Hann overlap-add.
/// Returns the extracted Lead Vocal buffer.
pub fn run_roformer_karaoke_onnx_model<F>(
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

    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    let num_bins = n_fft / 2 + 1; // 1025
    let dim_t = cfg.dim_t;
    let chunk_len = chunk_sample_length(cfg);
    let step = (chunk_len / 2).max(1);
    let margin_pad = chunk_len / 2;
    let padded_total = total_frames + 2 * margin_pad;

    let mut src_l = vec![0.0_f32; padded_total];
    let mut src_r = vec![0.0_f32; padded_total];
    src_l[margin_pad..margin_pad + total_frames].copy_from_slice(&audio.left[..total_frames]);
    src_r[margin_pad..margin_pad + total_frames].copy_from_slice(&audio.right[..total_frames]);

    let mut out_l = vec![0.0_f32; padded_total];
    let mut out_r = vec![0.0_f32; padded_total];
    let mut weight_acc = vec![0.0_f32; padded_total];
    let chunk_blend = hann_window(chunk_len);

    let window = hann_window(n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let c2r = planner.plan_fft_inverse(n_fft);
    let fft_scale = 1.0 / (n_fft as f32);

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

    let pad = n_fft / 2;
    let padded_chunk = chunk_len + 2 * pad;

    for (idx, &start) in starts.iter().enumerate() {
        let end = (start + chunk_len).min(padded_total);
        let valid_len = end - start;

        let mut pad_l = vec![0.0_f32; padded_chunk];
        let mut pad_r = vec![0.0_f32; padded_chunk];
        pad_l[pad..pad + valid_len].copy_from_slice(&src_l[start..end]);
        pad_r[pad..pad + valid_len].copy_from_slice(&src_r[start..end]);

        let mut stft_features = ndarray::Array3::<f32>::zeros((1, dim_t, 4100));
        let mut spec_l = vec![vec![Complex::new(0.0_f32, 0.0_f32); num_bins]; dim_t];
        let mut spec_r = vec![vec![Complex::new(0.0_f32, 0.0_f32); num_bins]; dim_t];

        let mut frame_buf = vec![0.0_f32; n_fft];
        let mut spectrum = r2c.make_output_vec();

        for t in 0..dim_t {
            let s = t * hop;
            if s + n_fft > padded_chunk {
                break;
            }
            for i in 0..n_fft {
                frame_buf[i] = pad_l[s + i] * window[i];
            }
            let _ = r2c.process(&mut frame_buf, &mut spectrum);
            for f in 0..num_bins.min(1025) {
                spec_l[t][f] = spectrum[f];
                stft_features[[0, t, 2 * f]] = spectrum[f].re;
                stft_features[[0, t, 2 * f + 1]] = spectrum[f].im;
            }

            for i in 0..n_fft {
                frame_buf[i] = pad_r[s + i] * window[i];
            }
            let _ = r2c.process(&mut frame_buf, &mut spectrum);
            for f in 0..num_bins.min(1025) {
                spec_r[t][f] = spectrum[f];
                stft_features[[0, t, 2050 + 2 * f]] = spectrum[f].re;
                stft_features[[0, t, 2050 + 2 * f + 1]] = spectrum[f].im;
            }
        }

        let tensor_ref = TensorRef::from_array_view(&stft_features)
            .map_err(|e| format!("Failed to wrap RoFormer Karaoke tensor: {}", e))?;
        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| format!("RoFormer Karaoke inference failed on chunk {}/{}: {}", idx + 1, total_chunks, e))?;

        let mask_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract RoFormer Karaoke mask: {}", e))?;

        let mut acc_l = vec![0.0_f32; padded_chunk];
        let mut acc_r = vec![0.0_f32; padded_chunk];
        let mut win_norm = vec![0.0_f32; padded_chunk];
        let mut out_spec = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
        let mut time_frame = c2r.make_output_vec();

        for t in 0..dim_t {
            let s = t * hop;
            if s + n_fft > padded_chunk {
                break;
            }

            // Left channel masked iFFT
            for f in 0..num_bins.min(1025) {
                let mr = mask_view[[0, 0, f, t, 0]];
                let mi = mask_view[[0, 0, f, t, 1]];
                let c = spec_l[t][f];
                out_spec[f] = Complex::new(c.re * mr - c.im * mi, c.re * mi + c.im * mr);
            }
            out_spec[0].im = 0.0;
            out_spec[num_bins - 1].im = 0.0;
            let _ = c2r.process(&mut out_spec, &mut time_frame);
            for i in 0..n_fft {
                let w = window[i];
                acc_l[s + i] += time_frame[i] * fft_scale * w;
                win_norm[s + i] += w * w;
            }

            // Right channel masked iFFT
            for f in 0..num_bins.min(1025) {
                let mr = mask_view[[0, 0, 1025 + f, t, 0]];
                let mi = mask_view[[0, 0, 1025 + f, t, 1]];
                let c = spec_r[t][f];
                out_spec[f] = Complex::new(c.re * mr - c.im * mi, c.re * mi + c.im * mr);
            }
            out_spec[0].im = 0.0;
            out_spec[num_bins - 1].im = 0.0;
            let _ = c2r.process(&mut out_spec, &mut time_frame);
            for i in 0..n_fft {
                acc_r[s + i] += time_frame[i] * fft_scale * window[i];
            }
        }

        for i in 0..valid_len {
            let denom = win_norm[pad + i].max(1e-6);
            let pl = acc_l[pad + i] / denom;
            let pr = acc_r[pad + i] / denom;
            let w = chunk_blend[i];
            out_l[start + i] += pl * w;
            out_r[start + i] += pr * w;
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
    let auto_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    // Stage 1: Instrumental & All Vocals
    let stage1_path = models_dir.join(&pipeline.stage1_inst_model.model_filename);
    let (mut sess1, mut active_backend) = create_mdx_session(env, &stage1_path, device)?;
    let (instrumental, all_vocals) = if pipeline.stage1_inst_model.model_filename == "htdemucs.onnx" {
        let dml_res = run_htdemucs_onnx_model(&mut sess1, audio, |c, total| {
            let pct = (c as f32 / total as f32) * 40.0;
            on_progress(SeparationProgress {
                stage_index: 1,
                stage_name: format!("Stage 1/3: {}", pipeline.stage1_inst_model.model_filename),
                chunk_index: c,
                total_chunks: total,
                overall_percent: pct,
            });
        });
        match dml_res {
            Ok(pair) => pair,
            Err(_) if !matches!(device, ExecutionDevice::Cpu { .. }) => {
                drop(sess1);
                let (mut cpu_sess, cpu_ep) = create_mdx_session(
                    env,
                    &stage1_path,
                    ExecutionDevice::Cpu {
                        intra_threads: auto_threads,
                    },
                )?;
                active_backend = cpu_ep;
                run_htdemucs_onnx_model(&mut cpu_sess, audio, |c, total| {
                    let pct = (c as f32 / total as f32) * 40.0;
                    on_progress(SeparationProgress {
                        stage_index: 1,
                        stage_name: format!("Stage 1/3 (CPU): {}", pipeline.stage1_inst_model.model_filename),
                        chunk_index: c,
                        total_chunks: total,
                        overall_percent: pct,
                    });
                })?
            }
            Err(e) => return Err(e),
        }
    } else {
        let stage1_out = run_mdx_onnx_model(&mut sess1, audio, &pipeline.stage1_inst_model, |c, total| {
            let pct = (c as f32 / total as f32) * 40.0;
            on_progress(SeparationProgress {
                stage_index: 1,
                stage_name: format!("Stage 1/3: {}", pipeline.stage1_inst_model.model_filename),
                chunk_index: c,
                total_chunks: total,
                overall_percent: pct,
            });
        })?;
        if pipeline.stage1_inst_model.outputs_vocals {
            let inst = subtract_stereo_buffers(audio, &stage1_out);
            (inst, stage1_out)
        } else {
            let voc = subtract_stereo_buffers(audio, &stage1_out);
            (stage1_out, voc)
        }
    };

    // Stage 2: Backing Vocal & Lead Vocal
    let stage2_path = models_dir.join(&pipeline.stage2_karaoke_model.model_filename);
    let (mut sess2, ep2) = create_mdx_session(env, &stage2_path, device)?;
    if active_backend.contains("CPU") || ep2.contains("CUDA") {
        active_backend = ep2;
    }
    let (backing_vocal, lead_with_reverb) =
        if pipeline.stage2_karaoke_model.model_filename == "mel_band_roformer_karaoke_aufr33_viperx.onnx" {
            let dml_res = run_roformer_karaoke_onnx_model(
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
            );
            let model_out = match dml_res {
                Ok(out) => out,
                Err(_) if !matches!(device, ExecutionDevice::Cpu { .. }) => {
                    drop(sess2);
                    let (mut cpu_sess, cpu_ep) = create_mdx_session(
                        env,
                        &stage2_path,
                        ExecutionDevice::Cpu {
                            intra_threads: auto_threads,
                        },
                    )?;
                    active_backend = cpu_ep;
                    run_roformer_karaoke_onnx_model(
                        &mut cpu_sess,
                        &all_vocals,
                        &pipeline.stage2_karaoke_model,
                        |c, total| {
                            let pct = 40.0 + (c as f32 / total as f32) * 35.0;
                            on_progress(SeparationProgress {
                                stage_index: 2,
                                stage_name: format!("Stage 2/3 (CPU): {}", pipeline.stage2_karaoke_model.model_filename),
                                chunk_index: c,
                                total_chunks: total,
                                overall_percent: pct,
                            });
                        },
                    )?
                }
                Err(e) => return Err(e),
            };
            if pipeline.stage2_karaoke_model.outputs_vocals {
                let backing = subtract_stereo_buffers(&all_vocals, &model_out);
                (backing, model_out)
            } else {
                // mel_band_roformer_karaoke_aufr33_viperx outputs Karaoke/Backing Vocal (lead removed)
                let lead = subtract_stereo_buffers(&all_vocals, &model_out);
                (model_out, lead)
            }
        } else {
            let backing = run_mdx_onnx_model(
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
            let lead = subtract_stereo_buffers(&all_vocals, &backing);
            (backing, lead)
        };

    // Stage 3: Optional De-Reverb (Dry Lead Vocal vs. Reverb Tail)
    let stage3_path = models_dir.join(&pipeline.stage3_dereverb_model.model_filename);
    let (lead_vocal, reverb_tail) = if pipeline.enable_dereverb && stage3_path.is_file() {
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
            stage_name: "Stage 3/3: Skipped De-Reverb (Beat + Vocal + Bè mode)".to_string(),
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
        execution_backend: active_backend.to_string(),
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
            outputs_vocals: false,
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

    #[test]
    fn test_cuda_ep_registration() {
        let models_dir = crate::model_manager::resolve_default_models_dir();
        let env = init_ort_environment(None).expect("init ort env");
        let p = models_dir.join("UVR-MDX-NET-Inst_HQ_1.onnx");
        if p.is_file() {
            let (_sess, ep_name) = create_mdx_session(&env, &p, ExecutionDevice::DirectMl { device_id: 0 }).expect("session");
            println!("ACTIVE_EP => {}", ep_name);
        }
    }

    #[test]
    #[ignore]
    fn test_real_onnx_models_smoke() {
        let models_dir = crate::model_manager::resolve_default_models_dir();
        let env = init_ort_environment(None).expect("init ort env");

        for name in [
            "htdemucs.onnx",
            "UVR-MDX-NET-Inst_HQ_1.onnx",
            "UVR_MDXNET_KARA_2.onnx",
            "Reverb_HQ_By_FoxJoy.onnx",
            "mel-roformer.fp16.onnx",
            "mel_band_roformer_karaoke_aufr33_viperx.onnx",
        ] {
            let p = models_dir.join(name);
            if let Ok((sess, ep_name)) = create_mdx_session(&env, &p, ExecutionDevice::DirectMl { device_id: 0 }) {
                println!("MODEL {} ({}):", name, ep_name);
                for inp in sess.inputs() {
                    println!("  IN  {:?}", inp);
                }
                for out in sess.outputs() {
                    println!("  OUT {:?}", out);
                }
            } else {
                println!("MODEL {} FAILED TO LOAD", name);
            }
        }

        let dummy = StereoAudioBuffer::new(44_100, vec![0.01; 44_100 * 2], vec![0.01; 44_100 * 2]);
        let fast_cfg = SeparationPipelineConfig::default_mdx_pipeline();
        let fast_res = separate_3_stage_stems(
            &env,
            &models_dir,
            &dummy,
            &fast_cfg,
            ExecutionDevice::DirectMl { device_id: 0 },
            |_| {},
        );
        println!("FAST_MDX_DML => {:?}", fast_res.as_ref().map(|_| "OK"));

        let pro_cfg = SeparationPipelineConfig::pro_roformer_pipeline();
        let pro_res = separate_3_stage_stems(
            &env,
            &models_dir,
            &dummy,
            &pro_cfg,
            ExecutionDevice::DirectMl { device_id: 0 },
            |_| {},
        );
        println!("PRO_ROFORMER_DML => {:?}", pro_res.as_ref().map(|_| "OK"));

        let htd_cfg = SeparationPipelineConfig::htdemucs_pipeline();
        let htd_res = separate_3_stage_stems(
            &env,
            &models_dir,
            &dummy,
            &htd_cfg,
            ExecutionDevice::DirectMl { device_id: 0 },
            |_| {},
        );
        println!("HTDEMUCS_DML => {:?}", htd_res.as_ref().map(|_| "OK"));

        let rof_kara_path = models_dir.join("mel_band_roformer_karaoke_aufr33_viperx.onnx");
        if let Ok((mut sess_rk, _)) = create_mdx_session(&env, &rof_kara_path, ExecutionDevice::DirectMl { device_id: 0 }) {
            let dummy_feat = ndarray::Array3::<f32>::zeros((1, 100, 4100));
            let tref = TensorRef::from_array_view(&dummy_feat).unwrap();
            match sess_rk.run(ort::inputs![tref]) {
                Ok(outs) => {
                    let v = outs[0].try_extract_array::<f32>().unwrap();
                    println!("ROFORMER_KARA_OUT_SHAPE => {:?}", v.shape());
                }
                Err(e) => println!("ROFORMER_KARA_ERR => {}", e),
            }
        }
    }
}
