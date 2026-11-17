//! Single-Binary Rust Entrypoint for Music Studio AI.
//! Integrates Stage 2 (Core Audio I/O & Multi-Stem Mixer) and Stage 3 (Formant-Preserving Pitch & Time DSP).

pub mod ai_separator;
pub mod audio_engine;
pub mod audio_io;
pub mod dsp_pitch_time;

use ai_separator::SeparationPipelineConfig;
use audio_engine::MasterTransportState;
use audio_io::STUDIO_SAMPLE_RATE;
use dsp_pitch_time::{PitchShiftEngine, PitchSpeedConfig};

fn main() {
    let pitch_config = PitchSpeedConfig {
        semitones: 0.0,
        cents: 0.0,
        speed_ratio: 1.0,
        preserve_formants: true,
        stem_aware_mode: true,
        transient_crisp_lock: true,
        engine: PitchShiftEngine::SignalsmithR3,
    };

    let transport = MasterTransportState::default();
    let pipeline_config = SeparationPipelineConfig::default_mdx_pipeline();

    println!(
        "Music Studio AI Core initialized @ {} Hz | Pitch Ratio: {:.4} | Engine: {:?} | Total Frames: {} | Stage 1 ONNX: {}",
        STUDIO_SAMPLE_RATE,
        pitch_config.pitch_ratio(),
        pitch_config.engine,
        transport.total_frames,
        pipeline_config.stage1_inst_model.model_filename
    );
}
