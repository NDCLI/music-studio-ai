//! Stage 2: Sample-Accurate Multi-Stem Audio Engine (`cpal` + `rtrb`).
//! Mixes Lead Vocal, Backing Vocal, Instrumental, and Reverb Tail in real time
//! with Mute/Solo, Constant-Power Panning, A/B Loop, and -0.3 dBFS True-Peak Limiting.

use std::f32::consts::FRAC_PI_4;

use rtrb::{Consumer, Producer, RingBuffer};
use serde::{Deserialize, Serialize};

use crate::audio_io::{StereoAudioBuffer, STUDIO_SAMPLE_RATE};
use crate::dsp_pitch_time::{
    apply_soft_knee_limiter, process_stereo_pitch_and_speed, PitchSpeedConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StemRole {
    LeadVocal,
    BackingVocal,
    Instrumental,
    ReverbTail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemChannelParams {
    pub role: StemRole,
    pub volume_percent: f32, // 0.0 to 150.0
    pub pan: f32,            // -1.0 (Left) to +1.0 (Right)
    pub muted: bool,
    pub solo: bool,
    pub formant_locked: bool,
}

#[derive(Debug, Clone)]
pub struct StemTrackData {
    pub params: StemChannelParams,
    pub audio: StereoAudioBuffer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterTransportState {
    pub is_playing: bool,
    pub current_frame: usize,
    pub total_frames: usize,
    pub loop_enabled: bool,
    pub loop_start_frame: usize,
    pub loop_end_frame: usize,
    pub master_volume_percent: f32,
    pub limiter_enabled: bool,
}

impl Default for MasterTransportState {
    fn default() -> Self {
        Self {
            is_playing: false,
            current_frame: 0,
            total_frames: (STUDIO_SAMPLE_RATE as usize) * 180,
            loop_enabled: false,
            loop_start_frame: (STUDIO_SAMPLE_RATE as usize) * 30,
            loop_end_frame: (STUDIO_SAMPLE_RATE as usize) * 75,
            master_volume_percent: 100.0,
            limiter_enabled: true,
        }
    }
}

/// Mixes multiple stems at a given frame range with Mute/Solo, Constant-Power Panning, and Limiter.
pub fn render_mixed_block(
    stems: &[StemTrackData],
    transport: &mut MasterTransportState,
    block_frames: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out_left = vec![0.0_f32; block_frames];
    let mut out_right = vec![0.0_f32; block_frames];

    if !transport.is_playing || stems.is_empty() {
        return (out_left, out_right);
    }

    let any_solo = stems.iter().any(|s| s.params.solo);
    let master_gain = (transport.master_volume_percent / 100.0).clamp(0.0, 1.5);

    for frame_offset in 0..block_frames {
        let mut cursor = transport.current_frame;
        if transport.loop_enabled
            && transport.loop_end_frame > transport.loop_start_frame
            && cursor >= transport.loop_end_frame
        {
            cursor = transport.loop_start_frame;
        } else if cursor >= transport.total_frames && transport.total_frames > 0 {
            cursor = 0;
        }
        transport.current_frame = cursor + 1;

        let mut mix_l = 0.0_f32;
        let mut mix_r = 0.0_f32;

        for stem in stems {
            if stem.params.muted || (any_solo && !stem.params.solo) {
                continue;
            }
            if cursor >= stem.audio.num_frames() {
                continue;
            }

            let gain = (stem.params.volume_percent / 100.0).clamp(0.0, 1.5);
            // Constant-power pan law
            let pan_angle = (stem.params.pan.clamp(-1.0, 1.0) + 1.0) * FRAC_PI_4;
            let pan_l = pan_angle.cos();
            let pan_r = pan_angle.sin();

            mix_l += stem.audio.left[cursor] * gain * pan_l;
            mix_r += stem.audio.right[cursor] * gain * pan_r;
        }

        out_left[frame_offset] = mix_l * master_gain;
        out_right[frame_offset] = mix_r * master_gain;
    }

    if transport.limiter_enabled {
        // -0.3 dBFS True-Peak Ceiling (0.966 linear)
        apply_soft_knee_limiter(&mut out_left, 0.966);
        apply_soft_knee_limiter(&mut out_right, 0.966);
    }

    (out_left, out_right)
}

/// Renders the entire song with Stem-Aware Pitch Shifting (Vocal Formant Lock vs. Instrumental Transient Lock).
pub fn render_stem_aware_master_mix(
    stems: &[StemTrackData],
    pitch_speed: &PitchSpeedConfig,
    limiter_enabled: bool,
) -> StereoAudioBuffer {
    let any_solo = stems.iter().any(|s| s.params.solo);
    let mut processed_stems: Vec<(f32, StereoAudioBuffer)> = Vec::new();
    let mut max_len = 0usize;

    for stem in stems {
        if stem.params.muted || (any_solo && !stem.params.solo) {
            continue;
        }

        let mut stem_cfg = pitch_speed.clone();
        if pitch_speed.stem_aware_mode {
            match stem.params.role {
                StemRole::LeadVocal | StemRole::BackingVocal => {
                    stem_cfg.preserve_formants = true;
                    stem_cfg.transient_crisp_lock = false;
                }
                StemRole::Instrumental => {
                    // Keep polyphonic instruments natural without vocal tract formant warping
                    stem_cfg.preserve_formants = false;
                    stem_cfg.transient_crisp_lock = true;
                }
                StemRole::ReverbTail => {
                    // Reverb tail is time-stretched without pitch warping to avoid metallic room artifacts
                    stem_cfg.semitones = 0.0;
                    stem_cfg.cents = 0.0;
                    stem_cfg.preserve_formants = false;
                }
            }
        }

        let rendered = process_stereo_pitch_and_speed(&stem.audio, &stem_cfg);
        max_len = max_len.max(rendered.num_frames());
        let gain = (stem.params.volume_percent / 100.0).clamp(0.0, 1.5);
        processed_stems.push((gain, rendered));
    }

    let mut mix_l = vec![0.0_f32; max_len];
    let mut mix_r = vec![0.0_f32; max_len];

    for (gain, buf) in processed_stems {
        let len = buf.num_frames();
        for i in 0..len {
            mix_l[i] += buf.left[i] * gain;
            mix_r[i] += buf.right[i] * gain;
        }
    }

    if limiter_enabled {
        apply_soft_knee_limiter(&mut mix_l, 0.966);
        apply_soft_knee_limiter(&mut mix_r, 0.966);
    }

    StereoAudioBuffer::new(STUDIO_SAMPLE_RATE, mix_l, mix_r)
}

/// Creates a lock-free SPSC audio ring buffer pair for real-time CPAL callback feeding.
pub fn create_realtime_ring_buffer(capacity_samples: usize) -> (Producer<f32>, Consumer<f32>) {
    RingBuffer::new(capacity_samples)
}
