//! Single-Binary Rust Entrypoint for Music Studio AI.
//! Integrates:
//! - Stage 2: Core Audio & Video Track Demuxer (`MP3`/`WAV`/`FLAC`/`MP4`/`MKV`/`WEBM`) + Multi-Stem Mixer
//! - Stage 3: Formant-Preserving Pitch & Time DSP + Key/BPM Detection
//! - Stage 4: Resumable HuggingFace ONNX Model Manager + SHA-256 Verification
//! - Stage 5: 3-Stage STFT/iSTFT + ONNX Runtime (`ort` load-dynamic + DirectML) Stem Separator
//! - Stage 6: Tauri v2 IPC Commands (`--features desktop-tauri`) + Multi-Format Export (`WAV` / `FLAC`)

pub mod ai_separator;
pub mod audio_engine;
pub mod audio_io;
pub mod dsp_pitch_time;
pub mod ipc_commands;
pub mod model_manager;
pub mod youtube_loader;

#[cfg(not(feature = "desktop-tauri"))]
fn main() {
    use ai_separator::{init_ort_environment, SeparationPipelineConfig};
    use audio_engine::MasterTransportState;
    use audio_io::STUDIO_SAMPLE_RATE;
    use dsp_pitch_time::{PitchShiftEngine, PitchSpeedConfig};
    use model_manager::{discover_onnxruntime_library, ModelManager};
    use youtube_loader::parse_youtube_single_video_url;

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

    if let Ok(manager) = ModelManager::new("models") {
        let cached = manager.scan_cached_models();
        let installed_count = cached.iter().filter(|m| m.installed).count();
        println!(
            "Model Manager ready at '{}' | Cached ONNX models: {}/{}",
            manager.models_dir().display(),
            installed_count,
            cached.len()
        );
    }

    if let Some(ort_dll) = discover_onnxruntime_library() {
        if let Ok(_env) = init_ort_environment(Some(&ort_dll)) {
            println!("ONNX Runtime initialized dynamically from '{}'", ort_dll.display());
        }
    } else {
        println!("ONNX Runtime DLL (`onnxruntime.dll`) not placed yet; ready for runtime loading.");
    }

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

#[cfg(feature = "desktop-tauri")]
fn main() {
    use std::sync::{Arc, Mutex};
    use ipc_commands::{tauri_commands, StudioAppState};

    let app_state = Arc::new(Mutex::new(
        StudioAppState::new("models").expect("Failed to initialize Music Studio AI state"),
    ));

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            tauri_commands::ipc_load_track,
            tauri_commands::ipc_parse_youtube_url,
            tauri_commands::ipc_update_pitch_speed,
            tauri_commands::ipc_list_models,
            tauri_commands::ipc_import_local_onnx,
            tauri_commands::ipc_delete_model,
            tauri_commands::ipc_run_separation,
            tauri_commands::ipc_export_audio,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to start Music Studio AI desktop runtime");
}
