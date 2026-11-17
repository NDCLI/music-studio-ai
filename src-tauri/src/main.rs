//! Single-Binary Rust Entrypoint for Music Studio AI.
//! Integrates:
//! - Stage 2: Core Audio & Video Track Demuxer (MP3/WAV/FLAC/MP4/MKV/WEBM) + Multi-Stem Mixer
//! - Stage 3: Formant-Preserving Pitch & Time DSP + Key/BPM Detection
//! - Direct YouTube Single-Track URL Parser (strips playlist bulk downloading)

pub mod ai_separator;
pub mod audio_engine;
pub mod audio_io;
pub mod dsp_pitch_time;
pub mod youtube_loader;

use ai_separator::SeparationPipelineConfig;
use audio_engine::MasterTransportState;
use audio_io::STUDIO_SAMPLE_RATE;
use dsp_pitch_time::{PitchShiftEngine, PitchSpeedConfig};
use youtube_loader::parse_youtube_single_video_url;

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

    if let Ok(parsed_yt) = parse_youtube_single_video_url(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&index=2",
    ) {
        println!(
            "YouTube Single-Track Parser active: {} (playlist stripped: {})",
            parsed_yt.canonical_watch_url, parsed_yt.was_in_playlist
        );
    }

    println!(
        "Music Studio AI Core initialized @ {} Hz | Pitch Ratio: {:.4} | Engine: {:?} | Total Frames: {} | Stage 1 ONNX: {}",
        STUDIO_SAMPLE_RATE,
        pitch_config.pitch_ratio(),
        pitch_config.engine,
        transport.total_frames,
        pipeline_config.stage1_inst_model.model_filename
    );
}
