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

/// Maps any signed index `k` into `0..len` using PyTorch-compatible mirror reflection (`mode='reflect'`).
#[inline]
fn reflect_index(k: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let mut m = k.rem_euclid(period);
    if m >= len as isize {
        m = period - m;
    }
    m as usize
}

/// Pads a 1D audio slice using mirror reflection (`pad_mode='reflect'`) so STFT/chunk boundaries
/// never suffer from step-to-zero broadband click/splatter artifacts.
pub fn reflect_pad_slice(src: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let total = pad_left + src.len() + pad_right;
    if src.is_empty() {
        return vec![0.0_f32; total];
    }
    if src.len() == 1 {
        return vec![src[0]; total];
    }
    let mut out = Vec::with_capacity(total);
    let len = src.len();
    for i in 0..total {
        let k = (i as isize) - (pad_left as isize);
        if k >= 0 && (k as usize) < len {
            out.push(src[k as usize]);
        } else {
            out.push(src[reflect_index(k, len)]);
        }
    }
    out
}

/// Builds a center-weighted raised-cosine overlap-add window that zeros out the outermost ~6.25%
/// edges of each neural chunk (eliminating U-Net / Transformer receptive-field border chatter)
/// while summing to an exact constant `1.0` at 50% hop (`step = chunk_len / 2`).
pub fn trimmed_chunk_window(chunk_len: usize, step: usize) -> Vec<f32> {
    if chunk_len < 32 || step < 8 {
        return hann_window(chunk_len.max(1));
    }
    let trim = (step / 8).max(1);
    let fade_len = step.saturating_sub(2 * trim).max(1);
    let flat_start = trim + fade_len;
    let flat_end = step + trim;
    let fade_out_end = flat_end + fade_len;

    let mut win = vec![0.0_f32; chunk_len];
    for (i, slot) in win.iter_mut().enumerate() {
        *slot = if i < trim {
            0.0
        } else if i < flat_start {
            let theta = (i - trim) as f32 / (fade_len as f32);
            let s = (0.5 * PI * theta).sin();
            s * s
        } else if i < flat_end {
            1.0
        } else if i < fade_out_end {
            let theta = (i - flat_end) as f32 / (fade_len as f32);
            let c = (0.5 * PI * theta).cos();
            c * c
        } else {
            0.0
        };
    }
    win
}

/// Computes chunk start offsets covering `[margin_pad .. margin_pad + total_frames]` with full
/// center-weighted overlap so even the final samples of the song have unity blend weight.
fn build_chunk_starts(total_frames: usize, step: usize, margin_pad: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pos = 0usize;
    let target_end = margin_pad + total_frames;
    while pos <= target_end {
        starts.push(pos);
        pos += step.max(1);
    }
    if starts.is_empty() {
        starts.push(0);
    }
    starts
}

/// Computes the number of audio samples corresponding to one `dim_t` STFT chunk.
pub fn chunk_sample_length(cfg: &StftModelConfig) -> usize {
    (cfg.dim_t.saturating_sub(1)) * cfg.hop_length
}

/// Computes `[1, 4, dim_f, dim_t]` complex spectrogram tensor from pre-padded stereo slices
/// of length `chunk_sample_length(cfg) + 2 * (cfg.n_fft / 2)`.
fn stft_stereo_from_padded(
    pad_l: &[f32],
    pad_r: &[f32],
    cfg: &StftModelConfig,
    window: &[f32],
    r2c: &dyn realfft::RealToComplex<f32>,
) -> Array4<f32> {
    let padded_len = pad_l.len().min(pad_r.len());
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

/// Converts a centered stereo chunk of length `chunk_sample_length(cfg)` into an ONNX tensor
/// with shape `[1, 4, dim_f, dim_t]` containing `[Re_L, Im_L, Re_R, Im_R]` using reflect padding.
pub fn stft_stereo_chunk(
    left_chunk: &[f32],
    right_chunk: &[f32],
    cfg: &StftModelConfig,
) -> Array4<f32> {
    let chunk_len = chunk_sample_length(cfg);
    let pad = cfg.n_fft / 2;
    let copy_len = chunk_len.min(left_chunk.len()).min(right_chunk.len());

    let pad_l = if copy_len == chunk_len {
        reflect_pad_slice(&left_chunk[..copy_len], pad, pad)
    } else {
        let mut tmp = vec![0.0_f32; chunk_len];
        tmp[..copy_len].copy_from_slice(&left_chunk[..copy_len]);
        reflect_pad_slice(&tmp, pad, pad)
    };
    let pad_r = if copy_len == chunk_len {
        reflect_pad_slice(&right_chunk[..copy_len], pad, pad)
    } else {
        let mut tmp = vec![0.0_f32; chunk_len];
        tmp[..copy_len].copy_from_slice(&right_chunk[..copy_len]);
        reflect_pad_slice(&tmp, pad, pad)
    };

    let window = hann_window(cfg.n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(cfg.n_fft);
    stft_stereo_from_padded(&pad_l, &pad_r, cfg, &window, r2c.as_ref())
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
/// using continuous reflect-padded STFT -> ONNX inference -> iSTFT + Trimmed Cosine Overlap-Add.
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
    let margin_pad = step;
    let stft_pad = cfg.n_fft / 2;

    let starts = build_chunk_starts(total_frames, step, margin_pad);
    let total_chunks = starts.len().max(1);
    let acc_len = starts.last().copied().unwrap_or(0) + chunk_len;
    let right_pad_needed = acc_len.saturating_sub(margin_pad + total_frames);

    // Continuous mirror-reflected audio with extra `stft_pad` on both sides so every STFT frame
    // in every chunk sees true neighboring waveform samples instead of zero-step discontinuities.
    let ext_l = reflect_pad_slice(
        &audio.left[..total_frames],
        margin_pad + stft_pad,
        right_pad_needed + stft_pad,
    );
    let ext_r = reflect_pad_slice(
        &audio.right[..total_frames],
        margin_pad + stft_pad,
        right_pad_needed + stft_pad,
    );

    let mut out_l = vec![0.0_f32; acc_len];
    let mut out_r = vec![0.0_f32; acc_len];
    let mut weight_acc = vec![0.0_f32; acc_len];

    let chunk_blend = trimmed_chunk_window(chunk_len, step);
    let stft_win = hann_window(cfg.n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(cfg.n_fft);
    let padded_slice_len = chunk_len + 2 * stft_pad;

    for (idx, &start) in starts.iter().enumerate() {
        let pad_slice_l = &ext_l[start..start + padded_slice_len];
        let pad_slice_r = &ext_r[start..start + padded_slice_len];

        let input_tensor =
            stft_stereo_from_padded(pad_slice_l, pad_slice_r, cfg, &stft_win, r2c.as_ref());
        let tensor_ref = TensorRef::from_array_view(&input_tensor)
            .map_err(|e| format!("Failed to wrap STFT tensor: {}", e))?;

        let outputs = session
            .run(ort::inputs![tensor_ref])
            .map_err(|e| format!("ONNX inference failed on chunk {}/{}: {}", idx + 1, total_chunks, e))?;

        let output_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Failed to extract f32 output tensor: {}", e))?;

        let (pred_l, pred_r) = istft_stereo_chunk(output_view, cfg)?;

        for i in 0..chunk_len {
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

/// Runs HTDemucs ONNX (`[1, 2, 343980]` -> `[1, 4, 2, 343980]`) with reflect-padded Trimmed Cosine overlap-add.
/// Sums Demucs stems 0..3 (`drums + bass + other`) as Instrumental and stem 3 as Vocals,
/// followed by Complex Wiener Phase-Consistent projection onto the original mixture.
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
    let margin_pad = step;
    let starts = build_chunk_starts(total_frames, step, margin_pad);
    let total_chunks = starts.len().max(1);
    let acc_len = starts.last().copied().unwrap_or(0) + HTDEMUCS_CHUNK;
    let right_pad_needed = acc_len.saturating_sub(margin_pad + total_frames);

    let src_l = reflect_pad_slice(&audio.left[..total_frames], margin_pad, right_pad_needed);
    let src_r = reflect_pad_slice(&audio.right[..total_frames], margin_pad, right_pad_needed);

    let mut inst_l = vec![0.0_f32; acc_len];
    let mut inst_r = vec![0.0_f32; acc_len];
    let mut voc_l = vec![0.0_f32; acc_len];
    let mut voc_r = vec![0.0_f32; acc_len];
    let mut weight_acc = vec![0.0_f32; acc_len];

    let chunk_blend = trimmed_chunk_window(HTDEMUCS_CHUNK, step);
    let mut input_tensor = ndarray::Array3::<f32>::zeros((1, 2, HTDEMUCS_CHUNK));

    for (idx, &start) in starts.iter().enumerate() {
        for i in 0..HTDEMUCS_CHUNK {
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

        for i in 0..HTDEMUCS_CHUNK {
            let w = chunk_blend[i];
            if w <= 0.0 {
                continue;
            }
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

    let raw_inst = StereoAudioBuffer::new(audio.sample_rate, final_inst_l, final_inst_r);
    let raw_voc = StereoAudioBuffer::new(audio.sample_rate, final_voc_l, final_voc_r);
    Ok(apply_htdemucs_wiener_consistency(audio, &raw_inst, &raw_voc))
}

/// Runs `mel_band_roformer_karaoke_aufr33_viperx.onnx` (`stft_features [1, T, 4100]` -> `mask [1, 1, 2050, T, 2]`)
/// with `n_fft = 2048`, `hop_length = 441` and continuous-context Trimmed Cosine overlap-add.
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
    let margin_pad = step;
    let pad = n_fft / 2;

    let starts = build_chunk_starts(total_frames, step, margin_pad);
    let total_chunks = starts.len().max(1);
    let acc_len = starts.last().copied().unwrap_or(0) + chunk_len;
    let right_pad_needed = acc_len.saturating_sub(margin_pad + total_frames);

    // Continuous mirror-reflected waveform context across chunks (zero boundary step clicks)
    let ext_l = reflect_pad_slice(
        &audio.left[..total_frames],
        margin_pad + pad,
        right_pad_needed + pad,
    );
    let ext_r = reflect_pad_slice(
        &audio.right[..total_frames],
        margin_pad + pad,
        right_pad_needed + pad,
    );

    let mut out_l = vec![0.0_f32; acc_len];
    let mut out_r = vec![0.0_f32; acc_len];
    let mut weight_acc = vec![0.0_f32; acc_len];
    let chunk_blend = trimmed_chunk_window(chunk_len, step);

    let window = hann_window(n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let c2r = planner.plan_fft_inverse(n_fft);
    let fft_scale = 1.0 / (n_fft as f32);

    let padded_chunk = chunk_len + 2 * pad;
    let mut stft_features = ndarray::Array3::<f32>::zeros((1, dim_t, 4100));
    let mut spec_l = vec![vec![Complex::new(0.0_f32, 0.0_f32); num_bins]; dim_t];
    let mut spec_r = vec![vec![Complex::new(0.0_f32, 0.0_f32); num_bins]; dim_t];
    let mut frame_buf = vec![0.0_f32; n_fft];
    let mut spectrum = r2c.make_output_vec();

    for (idx, &start) in starts.iter().enumerate() {
        let pad_l = &ext_l[start..start + padded_chunk];
        let pad_r = &ext_r[start..start + padded_chunk];
        stft_features.fill(0.0);

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

        for i in 0..chunk_len {
            let w = chunk_blend[i];
            if w <= 0.0 {
                continue;
            }
            let denom = win_norm[pad + i].max(1e-6);
            let pl = (acc_l[pad + i] / denom) * cfg.compensate;
            let pr = (acc_r[pad + i] / denom) * cfg.compensate;
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

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn median3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).min(a.min(b).max(c))
}

/// Computes a 1D sliding-window median (up to window size 11 on stack) to reject short
/// `< 45 ms` transient splats while preserving sustained vocal/harmony phrases intact.
fn sliding_median_1d(values: &[f32], half_win: usize) -> Vec<f32> {
    let n = values.len();
    let hw = half_win.min(5);
    let mut out = vec![0.0_f32; n];
    let mut buf = [0.0_f32; 11];
    for i in 0..n {
        let lo = i.saturating_sub(hw);
        let hi = (i + hw).min(n.saturating_sub(1));
        let count = hi - lo + 1;
        buf[..count].copy_from_slice(&values[lo..=hi]);
        buf[..count].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[i] = buf[count / 2];
    }
    out
}

struct StereoStftGrid {
    sample_rate: u32,
    orig_frames: usize,
    n_fft: usize,
    hop: usize,
    num_bins: usize,
    num_time: usize,
    left: Vec<Complex<f32>>,
    right: Vec<Complex<f32>>,
}

/// Computes a flat row-major `[num_time, num_bins]` stereo STFT with mirror-reflected boundaries.
fn stereo_stft_grid(audio: &StereoAudioBuffer, n_fft: usize, hop: usize) -> StereoStftGrid {
    let orig_frames = audio.num_frames();
    let num_bins = n_fft / 2 + 1;
    if orig_frames == 0 {
        return StereoStftGrid {
            sample_rate: audio.sample_rate,
            orig_frames: 0,
            n_fft,
            hop,
            num_bins,
            num_time: 0,
            left: Vec::new(),
            right: Vec::new(),
        };
    }

    let pad = n_fft / 2;
    let pad_right = pad + hop;
    let pad_l = reflect_pad_slice(&audio.left[..orig_frames], pad, pad_right);
    let pad_r = reflect_pad_slice(&audio.right[..orig_frames], pad, pad_right);
    let padded_len = pad_l.len().min(pad_r.len());
    let num_time = (padded_len.saturating_sub(n_fft)) / hop + 1;

    let window = hann_window(n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let mut frame_buf = vec![0.0_f32; n_fft];
    let mut spec = r2c.make_output_vec();
    let mut left = vec![Complex::new(0.0_f32, 0.0_f32); num_time * num_bins];
    let mut right = vec![Complex::new(0.0_f32, 0.0_f32); num_time * num_bins];

    for t in 0..num_time {
        let s = t * hop;
        let row = t * num_bins;

        for i in 0..n_fft {
            frame_buf[i] = pad_l[s + i] * window[i];
        }
        let _ = r2c.process(&mut frame_buf, &mut spec);
        left[row..row + num_bins].copy_from_slice(&spec[..num_bins]);

        for i in 0..n_fft {
            frame_buf[i] = pad_r[s + i] * window[i];
        }
        let _ = r2c.process(&mut frame_buf, &mut spec);
        right[row..row + num_bins].copy_from_slice(&spec[..num_bins]);
    }

    StereoStftGrid {
        sample_rate: audio.sample_rate,
        orig_frames,
        n_fft,
        hop,
        num_bins,
        num_time,
        left,
        right,
    }
}

/// Exact linear inverse STFT matching `stereo_stft_grid` via Hann COLA normalization.
fn stereo_istft_grid(grid: &StereoStftGrid) -> StereoAudioBuffer {
    if grid.orig_frames == 0 || grid.num_time == 0 {
        return StereoAudioBuffer::new(grid.sample_rate, Vec::new(), Vec::new());
    }

    let pad = grid.n_fft / 2;
    let padded_len = (grid.num_time - 1) * grid.hop + grid.n_fft;
    let window = hann_window(grid.n_fft);
    let mut planner = RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(grid.n_fft);
    let fft_scale = 1.0 / (grid.n_fft as f32);

    let mut acc_l = vec![0.0_f32; padded_len];
    let mut acc_r = vec![0.0_f32; padded_len];
    let mut win_norm = vec![0.0_f32; padded_len];
    let mut spec = vec![Complex::new(0.0_f32, 0.0_f32); grid.num_bins];
    let mut time_frame = c2r.make_output_vec();

    for t in 0..grid.num_time {
        let s = t * grid.hop;
        let row = t * grid.num_bins;

        spec.copy_from_slice(&grid.left[row..row + grid.num_bins]);
        spec[0].im = 0.0;
        if grid.num_bins > 1 {
            spec[grid.num_bins - 1].im = 0.0;
        }
        let _ = c2r.process(&mut spec, &mut time_frame);
        for i in 0..grid.n_fft {
            let w = window[i];
            acc_l[s + i] += time_frame[i] * fft_scale * w;
            win_norm[s + i] += w * w;
        }

        spec.copy_from_slice(&grid.right[row..row + grid.num_bins]);
        spec[0].im = 0.0;
        if grid.num_bins > 1 {
            spec[grid.num_bins - 1].im = 0.0;
        }
        let _ = c2r.process(&mut spec, &mut time_frame);
        for i in 0..grid.n_fft {
            acc_r[s + i] += time_frame[i] * fft_scale * window[i];
        }
    }

    let mut out_l = Vec::with_capacity(grid.orig_frames);
    let mut out_r = Vec::with_capacity(grid.orig_frames);
    for i in pad..(pad + grid.orig_frames) {
        let denom = win_norm[i].max(1e-6);
        out_l.push(acc_l[i] / denom);
        out_r.push(acc_r[i] / denom);
    }

    StereoAudioBuffer::new(grid.sample_rate, out_l, out_r)
}

/// Applies Complex Wiener mixture consistency to HTDemucs raw outputs, suppressing ConvTranspose1d
/// time-domain popping/splat artifacts while guaranteeing `instrumental + vocals == mix`.
pub fn apply_htdemucs_wiener_consistency(
    mix: &StereoAudioBuffer,
    raw_inst: &StereoAudioBuffer,
    raw_voc: &StereoAudioBuffer,
) -> (StereoAudioBuffer, StereoAudioBuffer) {
    const N_FFT: usize = 2048;
    const HOP: usize = 512;
    let frames = mix.num_frames().min(raw_inst.num_frames()).min(raw_voc.num_frames());
    if frames < N_FFT {
        return (raw_inst.clone(), raw_voc.clone());
    }

    let mix_g = stereo_stft_grid(mix, N_FFT, HOP);
    let inst_g = stereo_stft_grid(raw_inst, N_FFT, HOP);
    let voc_g = stereo_stft_grid(raw_voc, N_FFT, HOP);

    let nt = mix_g.num_time.min(inst_g.num_time).min(voc_g.num_time);
    let nb = mix_g.num_bins;

    let mut raw_mask_l = vec![0.0_f32; nt * nb];
    let mut raw_mask_r = vec![0.0_f32; nt * nb];

    for idx in 0..(nt * nb) {
        let vl2 = voc_g.left[idx].norm_sqr();
        let vr2 = voc_g.right[idx].norm_sqr();
        let il2 = inst_g.left[idx].norm_sqr();
        let ir2 = inst_g.right[idx].norm_sqr();

        // Power-law p = 1.6 (0.8 on squared magnitude) suppresses low-energy checkerboard noise
        let pv_st = (vl2 + vr2 + 1e-18).powf(0.8);
        let pi_st = (il2 + ir2 + 1e-18).powf(0.8);
        let m_st = pv_st / (pv_st + pi_st + 1e-12);

        let pv_l = (vl2 + 1e-18).powf(0.8);
        let pi_l = (il2 + 1e-18).powf(0.8);
        let m_l = pv_l / (pv_l + pi_l + 1e-12);

        let pv_r = (vr2 + 1e-18).powf(0.8);
        let pi_r = (ir2 + 1e-18).powf(0.8);
        let m_r = pv_r / (pv_r + pi_r + 1e-12);

        raw_mask_l[idx] = 0.70 * m_st + 0.30 * m_l;
        raw_mask_r[idx] = 0.70 * m_st + 0.30 * m_r;
    }

    let mut out_voc = StereoStftGrid {
        sample_rate: mix_g.sample_rate,
        orig_frames: mix_g.orig_frames,
        n_fft: N_FFT,
        hop: HOP,
        num_bins: nb,
        num_time: nt,
        left: vec![Complex::new(0.0, 0.0); nt * nb],
        right: vec![Complex::new(0.0, 0.0); nt * nb],
    };
    let mut out_inst = StereoStftGrid {
        sample_rate: mix_g.sample_rate,
        orig_frames: mix_g.orig_frames,
        n_fft: N_FFT,
        hop: HOP,
        num_bins: nb,
        num_time: nt,
        left: vec![Complex::new(0.0, 0.0); nt * nb],
        right: vec![Complex::new(0.0, 0.0); nt * nb],
    };

    for t in 0..nt {
        let row = t * nb;
        for f in 0..nb {
            let idx = row + f;
            // Light 3-bin frequency smoothing on Wiener mask prevents spectral notches
            let f_prev = f.saturating_sub(1);
            let f_next = (f + 1).min(nb - 1);
            let ml = 0.25 * raw_mask_l[row + f_prev] + 0.50 * raw_mask_l[idx] + 0.25 * raw_mask_l[row + f_next];
            let mr = 0.25 * raw_mask_r[row + f_prev] + 0.50 * raw_mask_r[idx] + 0.25 * raw_mask_r[row + f_next];

            let xl = mix_g.left[idx];
            let xr = mix_g.right[idx];

            // Clamp raw HTDemucs prediction magnitude against mixture magnitude to eliminate spurious pops
            let cap_l = xl.norm() * 1.04 + 1e-9;
            let raw_vl = voc_g.left[idx];
            let raw_vl_norm = raw_vl.norm();
            let safe_vl = if raw_vl_norm > cap_l {
                raw_vl * (cap_l / raw_vl_norm)
            } else {
                raw_vl
            };

            let cap_r = xr.norm() * 1.04 + 1e-9;
            let raw_vr = voc_g.right[idx];
            let raw_vr_norm = raw_vr.norm();
            let safe_vr = if raw_vr_norm > cap_r {
                raw_vr * (cap_r / raw_vr_norm)
            } else {
                raw_vr
            };

            let vl = xl * (ml * 0.82) + safe_vl * 0.18;
            let vr = xr * (mr * 0.82) + safe_vr * 0.18;

            out_voc.left[idx] = vl;
            out_voc.right[idx] = vr;
            out_inst.left[idx] = xl - vl;
            out_inst.right[idx] = xr - vr;
        }
    }

    (stereo_istft_grid(&out_inst), stereo_istft_grid(&out_voc))
}

/// Applies two-dimensional spectro-temporal smoothing to a `[nt, nb]` gain mask:
/// - 3-tap frequency smoothing `[0.25, 0.50, 0.25]`
/// - Asymmetric attack/release forward pass + 1-frame pre-attack lookahead pass
fn smooth_gain_mask_2d(raw: &[f32], nt: usize, nb: usize) -> Vec<f32> {
    let mut freq_smoothed = vec![0.0_f32; nt * nb];
    for t in 0..nt {
        let row = t * nb;
        for f in 0..nb {
            let f0 = f.saturating_sub(1);
            let f2 = (f + 1).min(nb - 1);
            freq_smoothed[row + f] =
                0.25 * raw[row + f0] + 0.50 * raw[row + f] + 0.25 * raw[row + f2];
        }
    }

    let mut temporal = freq_smoothed.clone();
    let alpha_attack = 0.80_f32;
    let alpha_release = 0.32_f32;

    for f in 0..nb {
        let mut env = freq_smoothed[f];
        for t in 1..nt {
            let idx = t * nb + f;
            let target = freq_smoothed[idx];
            if target > env {
                env = alpha_attack * target + (1.0 - alpha_attack) * env;
            } else {
                env = alpha_release * target + (1.0 - alpha_release) * env;
            }
            temporal[idx] = env;
        }

        // Gentle backward lookahead prevents clipping vocal consonant onsets
        for t in (0..nt.saturating_sub(1)).rev() {
            let idx = t * nb + f;
            let next_val = temporal[(t + 1) * nb + f] * 0.65;
            if next_val > temporal[idx] {
                temporal[idx] = next_val;
            }
        }
    }

    temporal
}

/// Eliminates low-level percussive transient splats ("bẹp bẹp nhỏ") and isolated musical-noise
/// islands leaking into `vocals`, returning the removed instrument energy back into `inst`
/// so `inst_clean + vocals_clean == inst + vocals` with zero phase distortion.
pub fn suppress_vocal_bleed_and_musical_noise(
    inst: &StereoAudioBuffer,
    vocals: &StereoAudioBuffer,
) -> (StereoAudioBuffer, StereoAudioBuffer) {
    const N_FFT: usize = 2048;
    const HOP: usize = 512;
    let frames = inst.num_frames().min(vocals.num_frames());
    if frames < N_FFT {
        return (inst.clone(), vocals.clone());
    }

    let mut inst_g = stereo_stft_grid(inst, N_FFT, HOP);
    let mut voc_g = stereo_stft_grid(vocals, N_FFT, HOP);
    let nt = inst_g.num_time.min(voc_g.num_time);
    let nb = inst_g.num_bins;
    let bin_hz = (vocals.sample_rate.max(1) as f32) / (N_FFT as f32);

    let mut ev = vec![0.0_f32; nt * nb];
    let mut ei = vec![0.0_f32; nt * nb];
    let mut ratio = vec![0.0_f32; nt * nb];
    let mut peak_ev = 0.0_f32;

    for idx in 0..(nt * nb) {
        let v_pow = voc_g.left[idx].norm_sqr() + voc_g.right[idx].norm_sqr();
        let i_pow = inst_g.left[idx].norm_sqr() + inst_g.right[idx].norm_sqr();
        ev[idx] = v_pow;
        ei[idx] = i_pow;
        ratio[idx] = v_pow / (v_pow + i_pow + 1e-14);
        if v_pow > peak_ev {
            peak_ev = v_pow;
        }
    }

    if peak_ev < 1e-14 {
        return (inst.clone(), vocals.clone());
    }

    // 1. Frame-level Vocal Activity Detection across core vocal band (140 Hz .. 4800 Hz)
    let f_lo = ((140.0 / bin_hz).round() as usize).min(nb.saturating_sub(1));
    let f_hi = ((4800.0 / bin_hz).round() as usize).min(nb.saturating_sub(1)).max(f_lo + 1);
    let mut vad_raw = vec![0.0_f32; nt];
    let mut band_ev = vec![0.0_f32; nt];
    let mut peak_band_ev = 1e-14_f32;

    for t in 0..nt {
        let row = t * nb;
        let mut sum_v = 0.0_f32;
        let mut sum_i = 0.0_f32;
        for f in f_lo..=f_hi {
            sum_v += ev[row + f];
            sum_i += ei[row + f];
        }
        band_ev[t] = sum_v;
        if sum_v > peak_band_ev {
            peak_band_ev = sum_v;
        }
        vad_raw[t] = sum_v / (sum_v + sum_i + 1e-12);
    }

    // Gate out ultra-quiet ambient frames (< -42 dB below peak vocal phrase) before dilating
    for t in 0..nt {
        let rel_gate = smoothstep(peak_band_ev * 2e-5, peak_band_ev * 2e-4, band_ev[t]);
        vad_raw[t] *= rel_gate;
    }

    // Reject <35 ms transient drum spikes via 7-frame median, then max-dilate +-4 frames (~93 ms)
    // so consonant attacks & breath releases around real vocal phrases stay 100% protected
    let vad_med = sliding_median_1d(&vad_raw, 3);
    let mut vad_dilated = vec![0.0_f32; nt];
    for t in 0..nt {
        let t0 = t.saturating_sub(4);
        let t1 = (t + 4).min(nt - 1);
        let mut m = 0.0_f32;
        for item in &vad_med[t0..=t1] {
            if *item > m {
                m = *item;
            }
        }
        vad_dilated[t] = m;
    }

    // 2. Spectro-temporal morphological island suppression & sub-bass thump cleanup
    let mut raw_gain = vec![1.0_f32; nt * nb];
    for t in 0..nt {
        let presence = smoothstep(0.025, 0.11, vad_dilated[t]);
        let low_thr = 0.060 * (1.0 - presence) + 0.006 * presence;
        let high_thr = 0.240 * (1.0 - presence) + 0.038 * presence;

        let t_prev = t.saturating_sub(1);
        let t_next = (t + 1).min(nt - 1);

        for f in 0..nb {
            let f_prev = f.saturating_sub(1);
            let f_next = (f + 1).min(nb - 1);

            // 3x3 weighted neighborhood suppresses isolated single-bin musical noise blips
            let r_center = ratio[t * nb + f];
            let r_cross = ratio[t_prev * nb + f]
                + ratio[t_next * nb + f]
                + ratio[t * nb + f_prev]
                + ratio[t * nb + f_next];
            let r_diag = ratio[t_prev * nb + f_prev]
                + ratio[t_prev * nb + f_next]
                + ratio[t_next * nb + f_prev]
                + ratio[t_next * nb + f_next];
            let r_bar = (4.0 * r_center + 2.0 * r_cross + r_diag) * (1.0 / 16.0);

            // 3-frame temporal median rejects broadband 1-2 frame snare/kick transient splats
            let r_med = median3(
                ratio[t_prev * nb + f],
                r_center,
                ratio[t_next * nb + f],
            );
            let r_eff = 0.50 * r_bar + 0.50 * r_med;

            let frame_guard = presence.max(smoothstep(0.55, 0.88, r_eff));
            let mut g = smoothstep(low_thr, high_thr, r_eff) * (0.04 + 0.96 * frame_guard);

            // Sub-bass kick/tom thump cleanup (< 82 Hz) where fundamental human vocals do not exist
            let freq = (f as f32) * bin_hz;
            if freq < 82.0 && r_eff < 0.50 {
                let sub_roll = smoothstep(38.0, 82.0, freq);
                g *= sub_roll * (0.20 + 0.80 * presence);
            }

            // Quiet-tail floor cleanup for residual -58 dBFS chatter
            let v_pow = ev[t * nb + f];
            if v_pow < peak_ev * 2e-6 && r_eff < 0.14 {
                g *= smoothstep(peak_ev * 1.5e-7, peak_ev * 2e-6, v_pow);
            }

            raw_gain[t * nb + f] = g;
        }
    }

    let gain = smooth_gain_mask_2d(&raw_gain, nt, nb);

    for idx in 0..(nt * nb) {
        let g = gain[idx];
        let keep_l = voc_g.left[idx] * g;
        let keep_r = voc_g.right[idx] * g;
        let bleed_l = voc_g.left[idx] - keep_l;
        let bleed_r = voc_g.right[idx] - keep_r;

        voc_g.left[idx] = keep_l;
        voc_g.right[idx] = keep_r;
        inst_g.left[idx] += bleed_l;
        inst_g.right[idx] += bleed_r;
    }

    (stereo_istft_grid(&inst_g), stereo_istft_grid(&voc_g))
}

/// Suppresses intermittent lead-vocal consonant chatter leaking into `backing_vocal` during solo
/// passages (returning those fragments cleanly to `lead_vocal`) and smoothly restores high-band
/// air above 17.4 kHz when `restore_high_band` is true (e.g. `UVR_MDXNET_KARA_2` cropped `dim_f`).
pub fn suppress_backing_chatter_and_restore_highs(
    lead: &StereoAudioBuffer,
    backing: &StereoAudioBuffer,
    restore_high_band: bool,
) -> (StereoAudioBuffer, StereoAudioBuffer) {
    const N_FFT: usize = 2048;
    const HOP: usize = 512;
    let frames = lead.num_frames().min(backing.num_frames());
    if frames < N_FFT {
        return (lead.clone(), backing.clone());
    }

    let mut lead_g = stereo_stft_grid(lead, N_FFT, HOP);
    let mut back_g = stereo_stft_grid(backing, N_FFT, HOP);
    let nt = lead_g.num_time.min(back_g.num_time);
    let nb = lead_g.num_bins;
    let bin_hz = (lead.sample_rate.max(1) as f32) / (N_FFT as f32);

    // Optional anti-Gibbs high-shelf continuity for MDX models cropped at ~17.6 kHz
    if restore_high_band {
        let ref_lo = ((11_500.0 / bin_hz).round() as usize).min(nb.saturating_sub(1));
        let ref_hi = ((16_800.0 / bin_hz).round() as usize).min(nb.saturating_sub(1)).max(ref_lo + 1);
        let cut_start = ((17_200.0 / bin_hz).round() as usize).min(nb.saturating_sub(1));

        for t in 0..nt {
            let row = t * nb;
            let mut eb_ref = 0.0_f32;
            let mut el_ref = 0.0_f32;
            for f in ref_lo..=ref_hi {
                eb_ref += back_g.left[row + f].norm_sqr() + back_g.right[row + f].norm_sqr();
                el_ref += lead_g.left[row + f].norm_sqr() + lead_g.right[row + f].norm_sqr();
            }
            let hi_share = (eb_ref / (eb_ref + el_ref + 1e-12)).clamp(0.0, 0.85);
            if hi_share > 0.02 {
                for f in cut_start..nb {
                    let freq = (f as f32) * bin_hz;
                    let blend = smoothstep(17_200.0, 18_100.0, freq) * hi_share;
                    let idx = row + f;
                    let tot_l = lead_g.left[idx] + back_g.left[idx];
                    let tot_r = lead_g.right[idx] + back_g.right[idx];
                    let new_bl = back_g.left[idx] * (1.0 - blend) + tot_l * blend;
                    let new_br = back_g.right[idx] * (1.0 - blend) + tot_r * blend;
                    back_g.left[idx] = new_bl;
                    back_g.right[idx] = new_br;
                    lead_g.left[idx] = tot_l - new_bl;
                    lead_g.right[idx] = tot_r - new_br;
                }
            }
        }
    }

    let mut eb = vec![0.0_f32; nt * nb];
    let mut el = vec![0.0_f32; nt * nb];
    let mut ratio = vec![0.0_f32; nt * nb];
    let mut peak_tot = 1e-14_f32;

    for idx in 0..(nt * nb) {
        let b_pow = back_g.left[idx].norm_sqr() + back_g.right[idx].norm_sqr();
        let l_pow = lead_g.left[idx].norm_sqr() + lead_g.right[idx].norm_sqr();
        eb[idx] = b_pow;
        el[idx] = l_pow;
        ratio[idx] = b_pow / (b_pow + l_pow + 1e-14);
        if b_pow + l_pow > peak_tot {
            peak_tot = b_pow + l_pow;
        }
    }

    // Sustain detector across 160 Hz .. 12000 Hz (~104 ms median window) distinguishes true
    // harmony phrases from short solo-vocal consonant splatter leaking into the backing stem.
    let f_lo = ((160.0 / bin_hz).round() as usize).min(nb.saturating_sub(1));
    let f_wide_hi = ((12_000.0 / bin_hz).round() as usize).min(nb.saturating_sub(1)).max(f_lo + 1);
    let mut wide_b_ratio = vec![0.0_f32; nt];
    let mut band_b_pow = vec![0.0_f32; nt];
    let mut peak_band = 1e-14_f32;

    for t in 0..nt {
        let row = t * nb;
        let mut sb = 0.0_f32;
        let mut sl = 0.0_f32;
        for f in f_lo..=f_wide_hi {
            sb += eb[row + f];
            sl += el[row + f];
        }
        band_b_pow[t] = sb;
        if sb + sl > peak_band {
            peak_band = sb + sl;
        }
        wide_b_ratio[t] = sb / (sb + sl + 1e-12);
    }

    for t in 0..nt {
        let gate = smoothstep(peak_band * 2e-5, peak_band * 2.5e-4, band_b_pow[t]);
        wide_b_ratio[t] *= gate;
    }

    // 9-frame (~104 ms) sliding median rejects transient <45 ms consonant/drum splats completely
    // while keeping any real harmony note (>= 60 ms) at full strength, dilated by +-2 frames for attack/release
    let med_b = sliding_median_1d(&wide_b_ratio, 4);
    let mut b_sustain = vec![0.0_f32; nt];
    for t in 0..nt {
        let t0 = t.saturating_sub(2);
        let t1 = (t + 2).min(nt - 1);
        let mut m = 0.0_f32;
        for &v in &med_b[t0..=t1] {
            if v > m {
                m = v;
            }
        }
        b_sustain[t] = m;
    }

    let mut raw_gain = vec![1.0_f32; nt * nb];
    for t in 0..nt {
        let harmony_active = smoothstep(0.025, 0.10, b_sustain[t]);
        let low_thr = 0.055 * (1.0 - harmony_active) + 0.006 * harmony_active;
        let high_thr = 0.220 * (1.0 - harmony_active) + 0.035 * harmony_active;

        let t_prev = t.saturating_sub(1);
        let t_next = (t + 1).min(nt - 1);

        for f in 0..nb {
            let f_prev = f.saturating_sub(1);
            let f_next = (f + 1).min(nb - 1);
            let r_c = ratio[t * nb + f];
            let r_cross = ratio[t_prev * nb + f]
                + ratio[t_next * nb + f]
                + ratio[t * nb + f_prev]
                + ratio[t * nb + f_next];
            let r_bar = (2.0 * r_c + r_cross) * (1.0 / 6.0);
            let r_med = median3(ratio[t_prev * nb + f], r_c, ratio[t_next * nb + f]);
            let r_eff = 0.55 * r_bar + 0.45 * r_med;

            // Scale by frame-level harmony envelope so isolated consonant splats stolen from lead
            // (even if el[f] == 0 at that specific harmonic bin) return cleanly to `lead_vocal`!
            let mut g = smoothstep(low_thr, high_thr, r_eff) * harmony_active;
            let b_pow = eb[t * nb + f];
            if b_pow < peak_tot * 1.5e-6 && r_eff < 0.12 {
                g *= smoothstep(peak_tot * 1.5e-7, peak_tot * 1.5e-6, b_pow);
            }
            raw_gain[t * nb + f] = g;
        }
    }

    let gain = smooth_gain_mask_2d(&raw_gain, nt, nb);

    for idx in 0..(nt * nb) {
        let g = gain[idx];
        let keep_l = back_g.left[idx] * g;
        let keep_r = back_g.right[idx] * g;
        let leak_l = back_g.left[idx] - keep_l;
        let leak_r = back_g.right[idx] - keep_r;

        back_g.left[idx] = keep_l;
        back_g.right[idx] = keep_r;
        lead_g.left[idx] += leak_l;
        lead_g.right[idx] += leak_r;
    }

    (stereo_istft_grid(&lead_g), stereo_istft_grid(&back_g))
}

/// Applies a single uniform scalar across all 4 stems only if any stem or the summed mix exceeds
/// `0.985`, preventing hard clipping while preserving 100% phase cancellation relationships.
fn normalize_stems_headroom_coherent(
    lead: &mut StereoAudioBuffer,
    backing: &mut StereoAudioBuffer,
    inst: &mut StereoAudioBuffer,
    reverb: &mut StereoAudioBuffer,
) {
    let frames = lead
        .num_frames()
        .min(backing.num_frames())
        .min(inst.num_frames())
        .min(reverb.num_frames());
    let mut peak = 0.0_f32;
    for i in 0..frames {
        let ll = lead.left[i];
        let lr = lead.right[i];
        let bl = backing.left[i];
        let br = backing.right[i];
        let il = inst.left[i];
        let ir = inst.right[i];
        let rl = reverb.left[i];
        let rr = reverb.right[i];

        peak = peak
            .max(ll.abs())
            .max(lr.abs())
            .max(bl.abs())
            .max(br.abs())
            .max(il.abs())
            .max(ir.abs())
            .max(rl.abs())
            .max(rr.abs())
            .max((ll + bl + il + rl).abs())
            .max((lr + br + ir + rr).abs());
    }

    if peak > 0.985 {
        let scale = 0.985 / peak;
        for buf in [lead, backing, inst, reverb] {
            for s in &mut buf.left {
                *s *= scale;
            }
            for s in &mut buf.right {
                *s *= scale;
            }
        }
    }
}

/// Runs the complete 3-Stage Local AI Separation Pipeline:
/// 1. Instrumental vs. All Vocals (+ Spectro-Temporal De-Bleed & Sub-Bass Cleanup)
/// 2. Backing Vocal vs. Lead Vocal (+ Backing De-Chatter & High-Band Continuity)
/// 3. Dry Lead Vocal vs. Reverb Tail (+ Coherent Headroom Protection)
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
    let (raw_inst, raw_all_vocals) = if pipeline.stage1_inst_model.model_filename == "htdemucs.onnx" {
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

    // Clean percussive transient bleed ("bẹp bẹp nhỏ") from vocals before Stage 2 Karaoke split
    let (mut instrumental, all_vocals) =
        suppress_vocal_bleed_and_musical_noise(&raw_inst, &raw_all_vocals);

    // Stage 2: Backing Vocal & Lead Vocal
    let stage2_path = models_dir.join(&pipeline.stage2_karaoke_model.model_filename);
    let (mut sess2, ep2) = create_mdx_session(env, &stage2_path, device)?;
    if active_backend.contains("CPU") || ep2.contains("CUDA") {
        active_backend = ep2;
    }
    let is_roformer_kara =
        pipeline.stage2_karaoke_model.model_filename == "mel_band_roformer_karaoke_aufr33_viperx.onnx";
    let (raw_backing, raw_lead_with_reverb) = if is_roformer_kara {
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

    let stage2_cropped_highs = !is_roformer_kara
        && pipeline.stage2_karaoke_model.dim_f + 64 < (pipeline.stage2_karaoke_model.n_fft / 2 + 1);
    let (lead_with_reverb, mut backing_vocal) = suppress_backing_chatter_and_restore_highs(
        &raw_lead_with_reverb,
        &raw_backing,
        stage2_cropped_highs,
    );

    // Stage 3: Optional De-Reverb (Dry Lead Vocal vs. Reverb Tail)
    let stage3_path = models_dir.join(&pipeline.stage3_dereverb_model.model_filename);
    let (mut lead_vocal, mut reverb_tail) = if pipeline.enable_dereverb && stage3_path.is_file() {
        let (mut sess3, _) = create_mdx_session(env, &stage3_path, device)?;
        let raw_dry_lead = run_mdx_onnx_model(
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
        let raw_tail = subtract_stereo_buffers(&lead_with_reverb, &raw_dry_lead);
        let stage3_cropped_highs = pipeline.stage3_dereverb_model.dim_f + 64
            < (pipeline.stage3_dereverb_model.n_fft / 2 + 1);
        suppress_backing_chatter_and_restore_highs(&raw_dry_lead, &raw_tail, stage3_cropped_highs)
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

    normalize_stems_headroom_coherent(
        &mut lead_vocal,
        &mut backing_vocal,
        &mut instrumental,
        &mut reverb_tail,
    );

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
    fn test_reflect_padding_and_trimmed_chunk_seamless_unity() {
        let src = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let padded = reflect_pad_slice(&src, 3, 3);
        assert_eq!(
            padded,
            vec![40.0, 30.0, 20.0, 10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 30.0, 20.0]
        );

        // Verify unity overlap-add for both even (MDX 4096) and odd (RoFormer-like 3999) chunk lengths
        for &chunk_len in &[4096usize, 3999usize] {
            let total_frames = 11_250usize;
            let step = chunk_len / 2;
            let margin_pad = step;
            let starts = build_chunk_starts(total_frames, step, margin_pad);
            let acc_len = starts.last().copied().unwrap_or(0) + chunk_len;
            let win = trimmed_chunk_window(chunk_len, step);

            // Outermost edges of each chunk must be strictly 0.0 to trim neural border artifacts
            assert_eq!(win[0], 0.0);
            assert_eq!(win[chunk_len - 1], 0.0);

            let mut weight_acc = vec![0.0_f32; acc_len];
            for &s in &starts {
                for i in 0..chunk_len {
                    weight_acc[s + i] += win[i];
                }
            }

            for i in margin_pad..(margin_pad + total_frames) {
                assert!(
                    weight_acc[i] > 0.98,
                    "Overlap weight dipped too low at sample {} (chunk_len={}): {}",
                    i,
                    chunk_len,
                    weight_acc[i]
                );
            }
        }
    }

    #[test]
    fn test_vocal_bleed_and_musical_noise_suppression_preserves_sum_and_removes_splats() {
        let sr = 44_100u32;
        let frames = sr as usize * 2; // 2.0 seconds
        let mut inst_l = vec![0.0_f32; frames];
        let mut inst_r = vec![0.0_f32; frames];
        let mut voc_l = vec![0.0_f32; frames];
        let mut voc_r = vec![0.0_f32; frames];

        for i in 0..frames {
            let t = i as f32 / sr as f32;
            // Instrumental: continuous 55 Hz bass/kick + sharp snare transient burst at t = 0.30..0.33s
            let kick = 0.45 * (2.0 * PI * 55.0 * t).sin();
            let snare = if (0.30..0.33).contains(&t) {
                0.55 * (2.0 * PI * 920.0 * t).sin() + 0.35 * (2.0 * PI * 2450.0 * t).cos()
            } else {
                0.0
            };
            inst_l[i] = kick + snare;
            inst_r[i] = kick + snare * 0.9;

            // Vocals: silent in 0.0..0.85s EXCEPT a 6% drum splat ("bẹp bẹp nhỏ") + 55 Hz thump bleed,
            // followed by a real sustained harmonic vocal phrase in 1.0..1.75s.
            let bleed_splat = 0.055 * kick + 0.065 * snare;
            let real_vocal = if (1.0..1.75).contains(&t) {
                0.42 * (2.0 * PI * 330.0 * t).sin()
                    + 0.24 * (2.0 * PI * 660.0 * t).sin()
                    + 0.12 * (2.0 * PI * 1320.0 * t).cos()
            } else {
                0.0
            };
            voc_l[i] = bleed_splat + real_vocal;
            voc_r[i] = bleed_splat + real_vocal;
        }

        let raw_inst = StereoAudioBuffer::new(sr, inst_l, inst_r);
        let raw_voc = StereoAudioBuffer::new(sr, voc_l, voc_r);

        let (clean_inst, clean_voc) = suppress_vocal_bleed_and_musical_noise(&raw_inst, &raw_voc);

        // 1. Exact mixture conservation: clean_inst + clean_voc == raw_inst + raw_voc
        let mut max_sum_err = 0.0_f32;
        for i in 2048..(frames - 2048) {
            let orig_sum = raw_inst.left[i] + raw_voc.left[i];
            let new_sum = clean_inst.left[i] + clean_voc.left[i];
            max_sum_err = max_sum_err.max((orig_sum - new_sum).abs());
        }
        assert!(
            max_sum_err < 1e-4,
            "Phase/mixture sum drifted by {}",
            max_sum_err
        );

        // 2. Drum splat + sub-bass bleed in non-vocal region (0.15..0.55s) must be heavily attenuated (> 95% power drop)
        let splat_range = (sr as usize * 15 / 100)..(sr as usize * 55 / 100);
        let raw_bleed_pow: f32 = splat_range.clone().map(|i| raw_voc.left[i].powi(2)).sum();
        let clean_bleed_pow: f32 = splat_range.map(|i| clean_voc.left[i].powi(2)).sum();
        assert!(
            clean_bleed_pow < raw_bleed_pow * 0.04,
            "Expected >14 dB bleed power reduction, got raw={} clean={}",
            raw_bleed_pow,
            clean_bleed_pow
        );

        // 3. Real sustained vocal phrase (1.15..1.60s) must remain intact
        let phrase_range = (sr as usize * 115 / 100)..(sr as usize * 160 / 100);
        let raw_voc_pow: f32 = phrase_range.clone().map(|i| raw_voc.left[i].powi(2)).sum();
        let clean_voc_pow: f32 = phrase_range.map(|i| clean_voc.left[i].powi(2)).sum();
        let ratio_preserved = clean_voc_pow / raw_voc_pow;
        assert!(
            ratio_preserved > 0.94,
            "Sustained vocal phrase attenuated too much: ratio={}",
            ratio_preserved
        );
    }

    #[test]
    fn test_backing_chatter_suppression_and_htdemucs_wiener_consistency() {
        let sr = 44_100u32;
        let frames = sr as usize * 2;
        let mut lead_l = vec![0.0_f32; frames];
        let mut back_l = vec![0.0_f32; frames];

        for i in 0..frames {
            let t = i as f32 / sr as f32;
            // Lead vocal sings solo in 0.2..0.9s, then duet harmony joins in 1.1..1.8s
            let solo_lead = if (0.2..1.8).contains(&t) {
                0.45 * (2.0 * PI * 440.0 * t).sin() + 0.20 * (2.0 * PI * 880.0 * t).sin()
            } else {
                0.0
            };
            // Spurious short 20ms consonant leak at t=0.50..0.52s on backing stem vs true harmony in 1.1..1.8s
            let chatter = if (0.50..0.52).contains(&t) {
                0.05 * (2.0 * PI * 1760.0 * t).sin()
            } else {
                0.0
            };
            let true_harmony = if (1.1..1.8).contains(&t) {
                0.32 * (2.0 * PI * 554.37 * t).sin() + 0.16 * (2.0 * PI * 1108.73 * t).sin()
            } else {
                0.0
            };
            lead_l[i] = solo_lead;
            back_l[i] = chatter + true_harmony;
        }

        let lead_buf = StereoAudioBuffer::new(sr, lead_l.clone(), lead_l);
        let back_buf = StereoAudioBuffer::new(sr, back_l.clone(), back_l);
        let (clean_lead, clean_back) =
            suppress_backing_chatter_and_restore_highs(&lead_buf, &back_buf, true);

        let chatter_slice = (sr as usize * 48 / 100)..(sr as usize * 55 / 100);
        let raw_chat_pow: f32 = chatter_slice.clone().map(|i| back_buf.left[i].powi(2)).sum();
        let clean_chat_pow: f32 = chatter_slice.map(|i| clean_back.left[i].powi(2)).sum();
        assert!(
            clean_chat_pow < raw_chat_pow * 0.05,
            "Backing chatter not sufficiently suppressed: raw={} clean={}",
            raw_chat_pow,
            clean_chat_pow
        );

        let harmony_slice = (sr as usize * 125 / 100)..(sr as usize * 165 / 100);
        let raw_harm_pow: f32 = harmony_slice.clone().map(|i| back_buf.left[i].powi(2)).sum();
        let clean_harm_pow: f32 = harmony_slice.map(|i| clean_back.left[i].powi(2)).sum();
        assert!(
            clean_harm_pow > raw_harm_pow * 0.90,
            "True backing harmony should remain intact"
        );

        // Verify HTDemucs Wiener consistency removes synthetic time-domain pops & preserves mix sum
        let mix = StereoAudioBuffer::new(
            sr,
            (0..frames)
                .map(|i| clean_lead.left[i] + clean_back.left[i])
                .collect(),
            (0..frames)
                .map(|i| clean_lead.right[i] + clean_back.right[i])
                .collect(),
        );
        let mut noisy_voc = clean_back.clone();
        // Inject an artificial ConvTranspose1d pop where mix is completely silent (t = 0.08s)
        for i in 3500..3600 {
            noisy_voc.left[i] += 0.35;
            noisy_voc.right[i] += 0.35;
        }
        let (w_inst, w_voc) = apply_htdemucs_wiener_consistency(&mix, &clean_lead, &noisy_voc);
        for i in 3500..3600 {
            assert!(
                w_voc.left[i].abs() < 0.02,
                "HTDemucs Wiener projection must suppress spurious pop in silent mix region, got {}",
                w_voc.left[i]
            );
            assert!(
                (w_inst.left[i] + w_voc.left[i] - mix.left[i]).abs() < 1e-4,
                "HTDemucs Wiener sum must equal original mix"
            );
        }
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
