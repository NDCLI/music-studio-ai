//! Single-Binary Rust Entrypoint for Music Studio AI.
//! Designed to minimize process spawning and avoid AhnLab antivirus scan bottlenecks.

pub mod ai_separator;
pub mod dsp_pitch_time;

use ai_separator::{SeparationPipelineConfig, StftModelConfig};
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

    let pipeline_config = SeparationPipelineConfig::default_mdx_pipeline();

    println!(
        "Music Studio AI Core initialized | Pitch Engine: {:?} | Stage 1 ONNX: {}",
        pitch_config.engine, pipeline_config.stage1_inst_model.model_filename
    );
}
