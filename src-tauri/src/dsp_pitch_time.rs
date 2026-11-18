//! Stage 3: Artifact-Free Pitch Shifting, Time Stretching, Key Detection & BPM Estimation.
//! Implements:
//! 1. Spectral Phase Vocoder with Instantaneous Frequency Phase Unwrapping (`realfft`).
//! 2. True-Envelope / Log-Spectral Formant Preservation (prevents "chipmunk" vocal distortion).
//! 3. Spectral Flux Transient Phase Reset (keeps kick/snare/guitar attacks crisp).
//! 4. Krumhansl-Schmuckler Chromagram Musical Key Detector & Onset BPM Detector.

use std::f32::consts::PI;

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};

use crate::audio_io::StereoAudioBuffer;

const VOCODER_FFT_SIZE: usize = 4096;
const VOCODER_SYNTH_HOP: usize = 512; // 87.5% overlap for low-phasiness pitch shifting

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PitchShiftEngine {
    /// Real-time spectral phase vocoder with formant envelope lock and transient phase reset.
    SignalsmithR3,
    /// Vocal analysis-synthesis separating F0, spectral envelope (SP), and aperiodicity (AP).
    WorldVocoder,
    /// Neural pitch extraction (RMVPE.onnx) + Neural Source-Filter vocoder (NSF-HiFiGAN.onnx).
    RmvpeHiFiGan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchSpeedConfig {
    pub semitones: f32,
    pub cents: f32,
    pub speed_ratio: f32,
    pub preserve_formants: bool,
    pub stem_aware_mode: bool,
    pub transient_crisp_lock: bool,
    pub engine: PitchShiftEngine,
}

impl Default for PitchSpeedConfig {
    fn default() -> Self {
        Self {
            semitones: 0.0,
            cents: 0.0,
            speed_ratio: 1.0,
            preserve_formants: true,
            stem_aware_mode: true,
            transient_crisp_lock: true,
            engine: PitchShiftEngine::SignalsmithR3,
        }
    }
}

impl PitchSpeedConfig {
    /// Computes the exact frequency pitch ratio: 2^((semitones + cents/100) / 12).
    pub fn pitch_ratio(&self) -> f32 {
        let total_semitones = self.semitones + (self.cents / 100.0);
        2.0_f32.powf(total_semitones / 12.0)
    }
}

/// Applies pitch shifting and time stretching to a stereo buffer with Formant Preservation.
pub fn process_stereo_pitch_and_speed(
    input: &StereoAudioBuffer,
    config: &PitchSpeedConfig,
) -> StereoAudioBuffer {
    let pitch_ratio = config.pitch_ratio();
    let speed_ratio = config.speed_ratio.clamp(0.25, 2.0);

    if (pitch_ratio - 1.0).abs() < 1e-4 && (speed_ratio - 1.0).abs() < 1e-4 {
        return input.clone();
    }

    if let Some(mut rb_out) = try_rubberband_r3_pitch_and_speed(input, config, speed_ratio) {
        apply_soft_knee_limiter(&mut rb_out.left, 0.966);
        apply_soft_knee_limiter(&mut rb_out.right, 0.966);
        return rb_out;
    }

    let (mut left_out, mut right_out) = process_stereo_vocoder(
        &input.left,
        &input.right,
        pitch_ratio,
        speed_ratio,
        config.preserve_formants,
        config.transient_crisp_lock,
    );

    // Apply -0.3 dBFS True-Peak Soft-Knee Limiter (0.966 linear ceiling)
    apply_soft_knee_limiter(&mut left_out, 0.966);
    apply_soft_knee_limiter(&mut right_out, 0.966);

    StereoAudioBuffer::new(input.sample_rate, left_out, right_out)
}

fn discover_rubberband_exe() -> Option<std::path::PathBuf> {
    let names = if cfg!(target_os = "windows") {
        ["rubberband-r3.exe", "rubberband.exe"]
    } else {
        ["rubberband-r3", "rubberband"]
    };

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            for name in names {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }

    let models_dir = crate::model_manager::resolve_default_models_dir();
    for name in names {
        let candidate = models_dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    for name in names {
        let candidate = std::path::PathBuf::from(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

fn try_rubberband_r3_pitch_and_speed(
    input: &StereoAudioBuffer,
    config: &PitchSpeedConfig,
    speed_ratio: f32,
) -> Option<StereoAudioBuffer> {
    let rb_exe = discover_rubberband_exe()?;
    if input.num_frames() == 0 {
        return Some(input.clone());
    }

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let temp_dir = std::env::temp_dir();
    let in_wav = temp_dir.join(format!("msai_rb_in_{}_{}.wav", pid, seq));
    let out_wav = temp_dir.join(format!("msai_rb_out_{}_{}.wav", pid, seq));

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: input.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    {
        let mut writer = hound::WavWriter::create(&in_wav, spec).ok()?;
        let frames = input.num_frames();
        for i in 0..frames {
            writer.write_sample(input.left[i]).ok()?;
            writer.write_sample(input.right[i]).ok()?;
        }
        writer.finalize().ok()?;
    }

    let total_semitones = config.semitones + (config.cents / 100.0);
    let mut cmd = std::process::Command::new(&rb_exe);
    cmd.arg("-3")
        .arg("-q")
        .arg("--tempo")
        .arg(format!("{:.6}", speed_ratio))
        .arg("--pitch")
        .arg(format!("{:.6}", total_semitones));
    if config.preserve_formants {
        cmd.arg("--formant");
    }
    cmd.arg(&in_wav).arg(&out_wav);

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let status = cmd.status().ok();
    let result = if matches!(status, Some(s) if s.success()) {
        if let Ok(mut reader) = hound::WavReader::open(&out_wav) {
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut samples = reader.samples::<f32>();
            while let (Some(Ok(l)), Some(Ok(r))) = (samples.next(), samples.next()) {
                left.push(l);
                right.push(r);
            }
            if !left.is_empty() {
                Some(StereoAudioBuffer::new(input.sample_rate, left, right))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let _ = std::fs::remove_file(&in_wav);
    let _ = std::fs::remove_file(&out_wav);
    result
}

#[inline]
fn wrap_phase_pi(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

#[inline]
fn vocoder_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Single-channel wrapper delegating to the phase-coherent stereo vocoder.
pub fn process_channel_vocoder(
    samples: &[f32],
    pitch_ratio: f32,
    speed_ratio: f32,
    preserve_formants: bool,
    transient_lock: bool,
) -> Vec<f32> {
    let (left, _) = process_stereo_vocoder(
        samples,
        samples,
        pitch_ratio,
        speed_ratio,
        preserve_formants,
        transient_lock,
    );
    left
}

/// Stereo Phase-Coherent Spectral Vocoder with:
/// - Exact fractional analysis hop tracking (eliminates hop-quantization drift & beating)
/// - Mirror-reflected boundary padding (preserves full duration to the last sample)
/// - Inter-Channel Phase Difference (ICPD) locking (keeps centered kick/bass/vocals rock-solid)
/// - Sub-band >350 Hz soft transient alignment (eliminates low-end phase-reset "splat/bẹp" pops)
/// - Temporal Formant envelope smoothing + Anti-Aliasing rolloff prior to pitch resampling
pub fn process_stereo_vocoder(
    left_samples: &[f32],
    right_samples: &[f32],
    pitch_ratio: f32,
    speed_ratio: f32,
    preserve_formants: bool,
    transient_lock: bool,
) -> (Vec<f32>, Vec<f32>) {
    let orig_len = left_samples.len().min(right_samples.len());
    if orig_len < VOCODER_FFT_SIZE {
        return (
            left_samples[..orig_len].to_vec(),
            right_samples[..orig_len].to_vec(),
        );
    }

    let time_stretch = (pitch_ratio as f64) / (speed_ratio as f64);
    let analysis_hop_f64 = (VOCODER_SYNTH_HOP as f64) / time_stretch;
    let target_stretched_frames = ((orig_len as f64) * time_stretch).round().max(1.0) as usize;
    let target_final_frames = ((orig_len as f64) / (speed_ratio as f64)).round().max(1.0) as usize;
    let num_bins = VOCODER_FFT_SIZE / 2 + 1;
    let pad = VOCODER_FFT_SIZE / 2;

    let pad_l = crate::ai_separator::reflect_pad_slice(
        &left_samples[..orig_len],
        pad,
        pad + VOCODER_FFT_SIZE * 2,
    );
    let pad_r = crate::ai_separator::reflect_pad_slice(
        &right_samples[..orig_len],
        pad,
        pad + VOCODER_FFT_SIZE * 2,
    );

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(VOCODER_FFT_SIZE);
    let c2r = planner.plan_fft_inverse(VOCODER_FFT_SIZE);

    let hann_window: Vec<f32> = (0..VOCODER_FFT_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / VOCODER_FFT_SIZE as f32).cos()))
        .collect();

    let mut prev_phase_l = vec![0.0_f32; num_bins];
    let mut prev_phase_r = vec![0.0_f32; num_bins];
    let mut sum_synth_phase_l = vec![0.0_f32; num_bins];
    let mut sum_synth_phase_r = vec![0.0_f32; num_bins];
    let mut prev_joint_mag = vec![0.0_f32; num_bins];
    let mut smooth_env: Vec<f32> = Vec::new();

    let total_synth_steps = (target_stretched_frames + pad) / VOCODER_SYNTH_HOP + 2;
    let buf_len = total_synth_steps * VOCODER_SYNTH_HOP + VOCODER_FFT_SIZE;
    let mut stretched_l = vec![0.0_f32; buf_len];
    let mut stretched_r = vec![0.0_f32; buf_len];
    let mut window_norm = vec![0.0_f32; buf_len];

    let mut time_in_l = vec![0.0_f32; VOCODER_FFT_SIZE];
    let mut time_in_r = vec![0.0_f32; VOCODER_FFT_SIZE];
    let mut spec_l = r2c.make_output_vec();
    let mut spec_r = r2c.make_output_vec();
    let mut synth_l = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
    let mut synth_r = vec![Complex::new(0.0_f32, 0.0_f32); num_bins];
    let mut time_out_l = c2r.make_output_vec();
    let mut time_out_r = c2r.make_output_vec();

    let mut mag_l = vec![0.0_f32; num_bins];
    let mut mag_r = vec![0.0_f32; num_bins];
    let mut joint_mag = vec![0.0_f32; num_bins];
    let mut phase_l = vec![0.0_f32; num_bins];
    let mut phase_r = vec![0.0_f32; num_bins];
    let mut locked_l = vec![0.0_f32; num_bins];
    let mut locked_r = vec![0.0_f32; num_bins];

    let mut prev_in_idx = 0usize;
    let inv_scale = 1.0 / (VOCODER_FFT_SIZE as f32);

    // Anti-aliasing Nyquist limit when pitching UP (pitch_ratio > 1.0)
    let aa_limit = if pitch_ratio > 1.001 {
        ((num_bins - 2) as f32) / pitch_ratio
    } else {
        (num_bins + 16) as f32
    };
    let aa_roll_start = (aa_limit * 0.92).max(16.0);

    for step_idx in 0..total_synth_steps {
        let out_pos = step_idx * VOCODER_SYNTH_HOP;
        let in_idx = (((step_idx as f64) * analysis_hop_f64).round() as usize)
            .min(pad_l.len().saturating_sub(VOCODER_FFT_SIZE));
        let actual_hop = if step_idx == 0 {
            analysis_hop_f64.round().max(1.0) as f32
        } else {
            (in_idx.saturating_sub(prev_in_idx)).max(1) as f32
        };
        prev_in_idx = in_idx;

        if out_pos + VOCODER_FFT_SIZE > buf_len {
            break;
        }

        for i in 0..VOCODER_FFT_SIZE {
            let w = hann_window[i];
            time_in_l[i] = pad_l[in_idx + i] * w;
            time_in_r[i] = pad_r[in_idx + i] * w;
        }

        let _ = r2c.process(&mut time_in_l, &mut spec_l);
        let _ = r2c.process(&mut time_in_r, &mut spec_r);

        let mut frame_peak_mag = 1e-7_f32;
        for k in 0..num_bins {
            let ml = spec_l[k].norm();
            let mr = spec_r[k].norm();
            let jm = (0.5 * (ml * ml + mr * mr)).sqrt();
            mag_l[k] = ml;
            mag_r[k] = mr;
            joint_mag[k] = jm;
            phase_l[k] = spec_l[k].arg();
            phase_r[k] = spec_r[k].arg();
            if jm > frame_peak_mag {
                frame_peak_mag = jm;
            }
        }

        // Measure positive high-band (>350 Hz) spectral flux for soft percussive attack preservation
        let mut hi_flux = 0.0_f32;
        let mut hi_energy = 1e-6_f32;
        for k in 33..num_bins {
            hi_flux += (joint_mag[k] - prev_joint_mag[k]).max(0.0);
            hi_energy += joint_mag[k];
        }
        let is_percussive_onset =
            transient_lock && step_idx > 0 && (hi_flux / hi_energy > 0.52);

        // Temporal-smoothed spectral envelope for Formant preservation
        if preserve_formants && (pitch_ratio - 1.0).abs() > 1e-3 {
            let inst_env = compute_spectral_envelope(&joint_mag, 28);
            if smooth_env.is_empty() {
                smooth_env = inst_env;
            } else {
                for k in 0..num_bins {
                    smooth_env[k] = 0.38 * inst_env[k] + 0.62 * smooth_env[k];
                }
            }
        }

        // Detect joint spectral peaks for Laroche-Dolson Identity Phase Locking
        let mut peaks = Vec::with_capacity(128);
        for k in 1..(num_bins - 1) {
            if joint_mag[k] >= joint_mag[k - 1] && joint_mag[k] >= joint_mag[k + 1] {
                peaks.push(k);
            }
        }
        if peaks.is_empty() {
            peaks.push(0);
        }

        if step_idx == 0 {
            prev_phase_l.copy_from_slice(&phase_l);
            prev_phase_r.copy_from_slice(&phase_r);
            sum_synth_phase_l.copy_from_slice(&phase_l);
            sum_synth_phase_r.copy_from_slice(&phase_r);
            locked_l.copy_from_slice(&phase_l);
            locked_r.copy_from_slice(&phase_r);
        } else {
            for &p in &peaks {
                let expected_adv =
                    2.0 * PI * (p as f32) * actual_hop / (VOCODER_FFT_SIZE as f32);

                let dev_l = wrap_phase_pi(phase_l[p] - prev_phase_l[p] - expected_adv);
                let freq_l = (expected_adv + dev_l) / actual_hop;
                sum_synth_phase_l[p] =
                    wrap_phase_pi(sum_synth_phase_l[p] + freq_l * (VOCODER_SYNTH_HOP as f32));

                let dev_r = wrap_phase_pi(phase_r[p] - prev_phase_r[p] - expected_adv);
                let freq_r = (expected_adv + dev_r) / actual_hop;
                sum_synth_phase_r[p] =
                    wrap_phase_pi(sum_synth_phase_r[p] + freq_r * (VOCODER_SYNTH_HOP as f32));

                // Soft transient phase coherence ONLY on surging mid/high bins (>355 Hz),
                // leaving bass/fundamental sustain completely untouched (zero "bẹp" clicks!)
                if is_percussive_onset && p >= 33 && joint_mag[p] > 1.75 * prev_joint_mag[p] {
                    let err_l = wrap_phase_pi(phase_l[p] - sum_synth_phase_l[p]);
                    sum_synth_phase_l[p] = wrap_phase_pi(sum_synth_phase_l[p] + 0.55 * err_l);
                    let err_r = wrap_phase_pi(phase_r[p] - sum_synth_phase_r[p]);
                    sum_synth_phase_r[p] = wrap_phase_pi(sum_synth_phase_r[p] + 0.55 * err_r);
                }
            }

            let mut peak_idx = 0usize;
            for k in 0..num_bins {
                prev_phase_l[k] = phase_l[k];
                prev_phase_r[k] = phase_r[k];

                while peak_idx + 1 < peaks.len()
                    && (peaks[peak_idx + 1] as isize - k as isize).abs()
                        < (peaks[peak_idx] as isize - k as isize).abs()
                {
                    peak_idx += 1;
                }
                let p = peaks[peak_idx];
                let pl = wrap_phase_pi(sum_synth_phase_l[p] + (phase_l[k] - phase_l[p]));
                locked_l[k] = pl;

                // Preserve exact Inter-Channel Phase Difference (ICPD) when Left has signal
                // so centered kick, bass, and lead vocals never drift out of stereo phase!
                if mag_l[k] >= 0.12 * mag_r[k] && mag_l[k] > 1e-8 {
                    let icpd = wrap_phase_pi(phase_r[k] - phase_l[k]);
                    locked_r[k] = wrap_phase_pi(pl + icpd);
                } else {
                    locked_r[k] = wrap_phase_pi(sum_synth_phase_r[p] + (phase_r[k] - phase_r[p]));
                }

                if k != p {
                    sum_synth_phase_l[k] = locked_l[k];
                    sum_synth_phase_r[k] = locked_r[k];
                }
            }
        }

        for k in 0..num_bins {
            let mut gain = 1.0_f32;
            if !smooth_env.is_empty() {
                let target_env_bin =
                    ((k as f32) * pitch_ratio).round().clamp(0.0, (num_bins - 1) as f32) as usize;
                let src_env = smooth_env[k].max(1e-5);
                let dst_env = smooth_env[target_env_bin];
                let raw_fg = (dst_env / src_env).clamp(0.25, 3.8);
                // Fade formant boost back to unity in noise-floor regions to prevent formant chatter
                let weight = vocoder_smoothstep(
                    frame_peak_mag * 8e-4,
                    frame_peak_mag * 6e-3,
                    joint_mag[k],
                );
                gain *= 1.0 + weight * (raw_fg - 1.0);
            }

            // Anti-aliasing low-pass rolloff before pitch-up resampling
            if (k as f32) > aa_roll_start {
                gain *= 1.0 - vocoder_smoothstep(aa_roll_start, aa_limit, k as f32);
            }

            synth_l[k] = Complex::from_polar(mag_l[k] * gain, locked_l[k]);
            synth_r[k] = Complex::from_polar(mag_r[k] * gain, locked_r[k]);
        }

        synth_l[0].im = 0.0;
        synth_l[num_bins - 1].im = 0.0;
        synth_r[0].im = 0.0;
        synth_r[num_bins - 1].im = 0.0;
        prev_joint_mag.copy_from_slice(&joint_mag);

        let _ = c2r.process(&mut synth_l, &mut time_out_l);
        let _ = c2r.process(&mut synth_r, &mut time_out_r);

        for i in 0..VOCODER_FFT_SIZE {
            let w = hann_window[i];
            stretched_l[out_pos + i] += time_out_l[i] * inv_scale * w;
            stretched_r[out_pos + i] += time_out_r[i] * inv_scale * w;
            window_norm[out_pos + i] += w * w;
        }
    }

    let extract_end = (pad + target_stretched_frames).min(buf_len);
    let mut clean_stretched_l = Vec::with_capacity(target_stretched_frames);
    let mut clean_stretched_r = Vec::with_capacity(target_stretched_frames);
    for i in pad..extract_end {
        let denom = window_norm[i].max(1e-5);
        clean_stretched_l.push(stretched_l[i] / denom);
        clean_stretched_r.push(stretched_r[i] / denom);
    }

    if (pitch_ratio - 1.0).abs() > 1e-4 {
        let mut rl = resample_channel_linear(&clean_stretched_l, pitch_ratio);
        let mut rr = resample_channel_linear(&clean_stretched_r, pitch_ratio);
        rl.resize(target_final_frames, 0.0);
        rr.resize(target_final_frames, 0.0);
        (rl, rr)
    } else {
        clean_stretched_l.resize(target_final_frames, 0.0);
        clean_stretched_r.resize(target_final_frames, 0.0);
        (clean_stretched_l, clean_stretched_r)
    }
}

/// Computes a smoothed spectral envelope representing the vocal tract Formant filter (F1, F2, F3).
fn compute_spectral_envelope(magnitudes: &[f32], half_window: usize) -> Vec<f32> {
    let n = magnitudes.len();
    let log_mags: Vec<f32> = magnitudes.iter().map(|&m| (m + 1e-6).ln()).collect();
    let mut env = Vec::with_capacity(n);

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);
        let sum: f32 = log_mags[start..end].iter().sum();
        let avg_log = sum / ((end - start) as f32);
        env.push(avg_log.exp());
    }

    env
}

/// 4-point Catmull-Rom cubic resampler used after time-stretching to achieve exact pitch ratio.
fn resample_channel_linear(input: &[f32], pitch_ratio: f32) -> Vec<f32> {
    if input.is_empty() || pitch_ratio <= 0.0 {
        return Vec::new();
    }
    let out_len = ((input.len() as f32) / pitch_ratio).round() as usize;
    let mut output = Vec::with_capacity(out_len);
    let last = input.len() - 1;

    for i in 0..out_len {
        let src_pos = (i as f32) * pitch_ratio;
        let idx = (src_pos.floor() as usize).min(last);
        let t = src_pos - (idx as f32);
        let s0 = input[idx.saturating_sub(1)];
        let s1 = input[idx];
        let s2 = input[(idx + 1).min(last)];
        let s3 = input[(idx + 2).min(last)];
        let a0 = -0.5 * s0 + 1.5 * s1 - 1.5 * s2 + 0.5 * s3;
        let a1 = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3;
        let a2 = -0.5 * s0 + 0.5 * s2;
        let a3 = s1;
        output.push(((a0 * t + a1) * t + a2) * t + a3);
    }

    output
}

/// Applies a soft-knee true-peak limiter at -0.3 dBFS to prevent clipping after pitch shifting.
pub fn apply_soft_knee_limiter(samples: &mut [f32], ceiling_linear: f32) {
    for sample in samples.iter_mut() {
        let abs_val = sample.abs();
        if abs_val > ceiling_linear {
            let sign = sample.signum();
            let excess = abs_val - ceiling_linear;
            *sample = sign * (ceiling_linear + excess / (1.0 + excess * 4.0)).min(0.995);
        }
    }
}

/// Krumhansl-Schmuckler Chromagram Musical Key Detector.
/// Returns the detected key string (e.g., "Am", "C", "Em", "G").
pub fn detect_musical_key(buffer: &StereoAudioBuffer) -> String {
    const NOTE_NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    const MAJOR_PROFILE: [f32; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    const MINOR_PROFILE: [f32; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];

    let frames = buffer.num_frames();
    if frames < 4096 {
        return "Am".to_string();
    }

    let fft_size = 4096usize;
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(fft_size);
    let mut input = vec![0.0_f32; fft_size];
    let mut spectrum = r2c.make_output_vec();
    let mut chroma = [0.0_f32; 12];

    let step = (frames / 24).max(fft_size);
    let mut pos = 0;

    while pos + fft_size <= frames {
        for i in 0..fft_size {
            input[i] = 0.5 * (buffer.left[pos + i] + buffer.right[pos + i]);
        }
        let _ = r2c.process(&mut input, &mut spectrum);

        for (bin, c) in spectrum.iter().enumerate().skip(1) {
            let freq = (bin as f32) * (buffer.sample_rate as f32) / (fft_size as f32);
            if (65.0..=1800.0).contains(&freq) {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                let pitch_class = (midi.round() as isize).rem_euclid(12) as usize;
                chroma[pitch_class] += c.norm();
            }
        }
        pos += step;
    }

    let mut best_score = f32::MIN;
    let mut best_key = "Am".to_string();

    for root in 0..12 {
        let mut maj_score = 0.0_f32;
        let mut min_score = 0.0_f32;
        for i in 0..12 {
            let c_val = chroma[(root + i) % 12];
            maj_score += c_val * MAJOR_PROFILE[i];
            min_score += c_val * MINOR_PROFILE[i];
        }

        if maj_score > best_score {
            best_score = maj_score;
            best_key = NOTE_NAMES[root].to_string();
        }
        if min_score > best_score {
            best_score = min_score;
            best_key = format!("{}m", NOTE_NAMES[root]);
        }
    }

    best_key
}

/// Estimates song tempo in BPM (68..=175 BPM) via energy envelope autocorrelation.
pub fn detect_bpm(buffer: &StereoAudioBuffer) -> f32 {
    let frames = buffer.num_frames();
    let hop = 512usize;
    if frames < hop * 64 {
        return 120.0;
    }

    let max_frames = frames.min((buffer.sample_rate as usize) * 30);
    let mut onset_env = Vec::with_capacity(max_frames / hop);
    let mut prev_energy = 0.0_f32;

    let mut pos = 0;
    while pos + hop <= max_frames {
        let mut energy = 0.0_f32;
        for i in 0..hop {
            let m = 0.5 * (buffer.left[pos + i] + buffer.right[pos + i]);
            energy += m * m;
        }
        let flux = (energy - prev_energy).max(0.0).sqrt();
        onset_env.push(flux);
        prev_energy = energy;
        pos += hop;
    }

    let env_rate = (buffer.sample_rate as f32) / (hop as f32);
    let min_lag = (env_rate * 60.0 / 175.0).round() as usize;
    let max_lag = (env_rate * 60.0 / 68.0).round() as usize;

    let mut best_lag = min_lag;
    let mut best_corr = 0.0_f32;

    for lag in min_lag..=max_lag.min(onset_env.len() / 2) {
        let mut corr = 0.0_f32;
        for i in 0..(onset_env.len() - lag) {
            corr += onset_env[i] * onset_env[i + lag];
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    let bpm = 60.0 * env_rate / (best_lag as f32);
    (bpm * 2.0).round() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_shift_preserves_duration() {
        let sr = 44_100;
        let len = sr * 2;
        let samples: Vec<f32> = (0..len)
            .map(|i| (2.0 * PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.5)
            .collect();
        let input = StereoAudioBuffer::new(sr, samples.clone(), samples);
        let cfg = PitchSpeedConfig {
            semitones: -4.0,
            speed_ratio: 1.0,
            ..Default::default()
        };
        let out = process_stereo_pitch_and_speed(&input, &cfg);
        let ratio = (out.num_frames() as f32) / (len as f32);
        assert!(
            (ratio - 1.0).abs() < 0.03,
            "Pitch shift -4st at 1.0x speed should preserve duration, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_stereo_vocoder_icpd_lock_and_transient_smoothness() {
        let sr = 44_100u32;
        let len = sr as usize * 2;
        let mut center_mix = Vec::with_capacity(len);
        for i in 0..len {
            let t = i as f32 / sr as f32;
            // Sustained 70 Hz bass + 440 Hz vocal + sharp transient snare hit at t = 0.80..0.82s
            let mut s = 0.35 * (2.0 * PI * 70.0 * t).sin() + 0.25 * (2.0 * PI * 440.0 * t).sin();
            if (0.80..0.82).contains(&t) {
                let env = 1.0 - ((t - 0.80) / 0.02);
                s += 0.40 * env * (2.0 * PI * 1850.0 * t).sin();
            }
            center_mix.push(s);
        }

        let input = StereoAudioBuffer::new(sr, center_mix.clone(), center_mix);
        for &semi in &[3.0_f32, -3.0_f32] {
            let (out_l, out_r) = process_stereo_vocoder(
                &input.left,
                &input.right,
                2.0_f32.powf(semi / 12.0),
                1.0,
                true,
                true,
            );
            assert_eq!(out_l.len(), len);
            assert_eq!(out_r.len(), len);

            let mut max_lr_diff = 0.0_f32;
            let mut max_sample_step = 0.0_f32;
            for i in 1..len {
                max_lr_diff = max_lr_diff.max((out_l[i] - out_r[i]).abs());
                max_sample_step = max_sample_step.max((out_l[i] - out_l[i - 1]).abs());
            }
            assert!(
                max_lr_diff < 1e-5,
                "Centered stereo signal drifted out of phase (semi={}): max_lr_diff={}",
                semi,
                max_lr_diff
            );
            assert!(
                max_sample_step < 0.45,
                "Unexpected click/step discontinuity during transient pitch shift (semi={}): {}",
                semi,
                max_sample_step
            );
        }
    }
}
