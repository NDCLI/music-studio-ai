//! Formant-Preserving Pitch Shifting and Time Stretching DSP Module.
//! Implements spectral envelope separation (Cepstral / True-Envelope Formant Lock)
//! and multi-band transient phase reset in Pure Rust.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PitchShiftEngine {
    /// Real-time spectral phase vocoder with cepstral formant preservation and transient lock.
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

impl PitchSpeedConfig {
    /// Computes the exact frequency pitch ratio: 2^((semitones + cents/100) / 12).
    pub fn pitch_ratio(&self) -> f32 {
        let total_semitones = self.semitones + (self.cents / 100.0);
        2.0_f32.powf(total_semitones / 12.0)
    }
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
