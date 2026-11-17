//! Stage 6: Tauri v2 IPC Bridge & Studio Application State (`ipc_commands`).
//! Exposes a unified Rust backend API for:
//! - Audio & Video (`MP4`/`MKV`/`WEBM`) decoding + Key/BPM detection
//! - Direct YouTube single-track playlist filtering
//! - Real-time & Stem-Aware Pitch/Speed configuration
//! - Local ONNX Model Manager (`download`, `import_local_onnx`, `delete`)
//! - 3-Stage ONNX Runtime (`ort` load-dynamic + DirectML) Stem Separation
//! - Multi-format Studio Export (`WAV 16/24-bit`, `WAV 32-bit Float`, `FLAC 24-bit Lossless`)

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::ai_separator::{
    init_ort_environment, separate_3_stage_stems, ExecutionDevice, SeparatedStems,
    SeparationPipelineConfig,
};
use crate::audio_engine::{render_stem_aware_master_mix, StemChannelParams, StemRole, StemTrackData};
use crate::audio_io::{
    decode_audio_file, export_audio_file, is_video_container_path, AudioExportFormat,
    LoadedTrackSummary, StereoAudioBuffer,
};
use crate::dsp_pitch_time::{
    detect_bpm, detect_musical_key, process_stereo_pitch_and_speed, PitchSpeedConfig,
};
use crate::model_manager::{CachedModelStatus, ModelCatalogEntry, ModelManager};
use crate::youtube_loader::{parse_youtube_single_video_url, ParsedYouTubeTarget};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeparationMode {
    FastMdx,
    ProRoformer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportTargetMode {
    MasterMix,
    AllStems,
    KaraokeBacking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationStemPeaksResponse {
    pub lead_vocal: Vec<f32>,
    pub backing_vocal: Vec<f32>,
    pub instrumental: Vec<f32>,
    pub reverb_tail: Vec<f32>,
    pub execution_backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportAudioResponse {
    pub exported_paths: Vec<String>,
    pub format: AudioExportFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsOverviewResponse {
    pub models_dir: String,
    pub catalog: Vec<ModelCatalogEntry>,
    pub cached_status: Vec<CachedModelStatus>,
    pub ort_dll_found: Option<String>,
}

/// Shared thread-safe Studio state held inside Tauri managed state.
pub struct StudioAppState {
    pub original_track: Option<StereoAudioBuffer>,
    pub separated_stems: Option<SeparatedStems>,
    pub pitch_speed: PitchSpeedConfig,
    pub model_manager: ModelManager,
    pub ort_env: Option<ort::environment::Environment>,
}

impl StudioAppState {
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Result<Self, String> {
        let model_manager = ModelManager::new(models_dir)?;
        Ok(Self {
            original_track: None,
            separated_stems: None,
            pitch_speed: PitchSpeedConfig::default(),
            model_manager,
            ort_env: None,
        })
    }
}

pub fn handle_load_track(
    state: &Arc<Mutex<StudioAppState>>,
    file_path: &str,
) -> Result<LoadedTrackSummary, String> {
    let path = Path::new(file_path);
    let buffer = decode_audio_file(path)?;
    let detected_key = detect_musical_key(&buffer);
    let detected_bpm = detect_bpm(&buffer);
    let waveform_peaks = buffer.compute_waveform_peaks(140);
    let duration_sec = buffer.duration_seconds();
    let sample_rate = buffer.sample_rate;
    let is_video = is_video_container_path(path);
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("track")
        .to_string();

    let mut guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;
    guard.original_track = Some(buffer);
    guard.separated_stems = None;

    Ok(LoadedTrackSummary {
        file_name,
        is_video_source: is_video,
        sample_rate,
        channels: 2,
        duration_sec,
        detected_key,
        detected_bpm,
        waveform_peaks,
    })
}

pub fn handle_parse_youtube(raw_url: &str) -> Result<ParsedYouTubeTarget, String> {
    parse_youtube_single_video_url(raw_url)
}

pub fn handle_update_pitch_speed(
    state: &Arc<Mutex<StudioAppState>>,
    config: PitchSpeedConfig,
) -> Result<f32, String> {
    let mut guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;
    guard.pitch_speed = config;
    Ok(guard.pitch_speed.pitch_ratio())
}

pub fn handle_list_models(state: &Arc<Mutex<StudioAppState>>) -> Result<ModelsOverviewResponse, String> {
    let guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;
    let ort_dll = crate::model_manager::discover_onnxruntime_library()
        .map(|p| p.display().to_string());

    Ok(ModelsOverviewResponse {
        models_dir: guard.model_manager.models_dir().display().to_string(),
        catalog: guard.model_manager.catalog().to_vec(),
        cached_status: guard.model_manager.scan_cached_models(),
        ort_dll_found: ort_dll,
    })
}

pub fn handle_import_local_onnx(
    state: &Arc<Mutex<StudioAppState>>,
    source_path: &str,
    target_model_id: Option<String>,
) -> Result<CachedModelStatus, String> {
    let guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;
    guard
        .model_manager
        .import_local_onnx_model(source_path, target_model_id.as_deref())
}

pub fn handle_delete_model(
    state: &Arc<Mutex<StudioAppState>>,
    file_name: &str,
) -> Result<(), String> {
    let guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;
    guard.model_manager.delete_cached_model(file_name)
}

pub fn handle_download_model(
    state: &Arc<Mutex<StudioAppState>>,
    model_id: &str,
) -> Result<CachedModelStatus, String> {
    let manager = {
        let guard = state
            .lock()
            .map_err(|_| "Failed to acquire studio state lock".to_string())?;
        guard.model_manager.clone()
    };
    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    manager.download_model_resumable(model_id, cancel_flag, |_| {})
}

/// Resolves the default writable export directory (`%USERPROFILE%\Music\Music Studio AI` on Windows).
pub fn resolve_default_export_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let base = PathBuf::from(home);
        let music_dir = base.join("Music");
        if music_dir.exists() {
            return music_dir.join("Music Studio AI");
        }
        return base.join("Downloads").join("Music Studio AI");
    }
    PathBuf::from("exports")
}

pub fn handle_get_default_export_dir() -> Result<String, String> {
    let dir = resolve_default_export_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to prepare default export folder: {}", e))?;
    Ok(dir.display().to_string())
}

/// Opens a native Windows folder selection dialog via Win32 `SHBrowseForFolderW` (no external `.exe`).
pub fn handle_pick_export_folder() -> Result<Option<String>, String> {
    #[cfg(target_os = "windows")]
    {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::os::windows::ffi::OsStringExt;

        #[repr(C)]
        struct BrowseInfoW {
            hwnd_owner: *mut core::ffi::c_void,
            pidl_root: *const core::ffi::c_void,
            psz_display_name: *mut u16,
            lpsz_title: *const u16,
            ul_flags: u32,
            lpfn: *const core::ffi::c_void,
            l_param: isize,
            i_image: i32,
        }

        #[link(name = "shell32")]
        extern "system" {
            fn SHBrowseForFolderW(lpbi: *mut BrowseInfoW) -> *mut core::ffi::c_void;
            fn SHGetPathFromIDListW(pidl: *const core::ffi::c_void, psz_path: *mut u16) -> i32;
        }

        #[link(name = "ole32")]
        extern "system" {
            fn CoInitializeEx(pv_reserved: *mut core::ffi::c_void, dw_co_init: u32) -> i32;
            fn CoTaskMemFree(pv: *mut core::ffi::c_void);
        }

        unsafe {
            let _ = CoInitializeEx(core::ptr::null_mut(), 0x2); // COINIT_APARTMENTTHREADED
        }

        let mut display_buf = [0_u16; 260];
        let title: Vec<u16> = OsStr::new("Chon thu muc luu file nhac xuat - Music Studio AI")
            .encode_wide()
            .chain(Some(0))
            .collect();

        let mut bi = BrowseInfoW {
            hwnd_owner: core::ptr::null_mut(),
            pidl_root: core::ptr::null(),
            psz_display_name: display_buf.as_mut_ptr(),
            lpsz_title: title.as_ptr(),
            ul_flags: 0x0000_0041, // BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE
            lpfn: core::ptr::null(),
            l_param: 0,
            i_image: 0,
        };

        let pidl = unsafe { SHBrowseForFolderW(&mut bi) };
        if pidl.is_null() {
            return Ok(None);
        }

        let mut path_buf = [0_u16; 1024];
        let ok = unsafe { SHGetPathFromIDListW(pidl, path_buf.as_mut_ptr()) };
        unsafe { CoTaskMemFree(pidl) };

        if ok == 0 {
            return Ok(None);
        }

        let len = path_buf.iter().position(|&c| c == 0).unwrap_or(path_buf.len());
        let selected = std::ffi::OsString::from_wide(&path_buf[..len])
            .to_string_lossy()
            .into_owned();
        return Ok(Some(selected));
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err("Native folder picker is only supported on Windows".to_string())
    }
}

/// Saves rendered audio bytes directly into the user-selected export folder safely.
pub fn handle_save_exported_file(
    output_dir: &str,
    file_name: &str,
    bytes: &[u8],
) -> Result<String, String> {
    let dir = PathBuf::from(output_dir);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to create export folder '{}': {}", dir.display(), e))?;

    let safe_name = Path::new(file_name)
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "Invalid export file name".to_string())?;
    let target_path = dir.join(safe_name);

    std::fs::write(&target_path, bytes)
        .map_err(|e| format!("Failed to write '{}': {}", target_path.display(), e))?;

    Ok(target_path.display().to_string())
}

/// Opens a local directory in Windows File Explorer without spawning shell commands.
pub fn handle_open_local_folder(folder_path: &str) -> Result<(), String> {
    let dir = Path::new(folder_path);
    if !dir.is_dir() {
        return Err(format!("Folder '{}' does not exist", dir.display()));
    }

    #[cfg(target_os = "windows")]
    {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        #[link(name = "shell32")]
        extern "system" {
            fn ShellExecuteW(
                hwnd: *mut core::ffi::c_void,
                lp_operation: *const u16,
                lp_file: *const u16,
                lp_parameters: *const u16,
                lp_directory: *const u16,
                n_show_cmd: i32,
            ) -> isize;
        }

        let op: Vec<u16> = OsStr::new("open").encode_wide().chain(Some(0)).collect();
        let path_w: Vec<u16> = OsStr::new(folder_path).encode_wide().chain(Some(0)).collect();
        let res = unsafe {
            ShellExecuteW(
                core::ptr::null_mut(),
                op.as_ptr(),
                path_w.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                1,
            )
        };
        if res <= 32 {
            return Err(format!("Failed to open folder (code {})", res));
        }
        return Ok(());
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = dir;
        Err("Open folder is only supported on Windows".to_string())
    }
}

/// Validates and opens an external `https://` URL in the OS default browser without spawning shell `.exe`s.
pub fn handle_open_external_url(url: &str) -> Result<(), String> {
    let trimmed = url.trim();
    if !trimmed.starts_with("https://")
        || trimmed.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        return Err("Only valid HTTPS URLs without whitespace are permitted".to_string());
    }

    #[cfg(target_os = "windows")]
    {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        #[link(name = "shell32")]
        extern "system" {
            fn ShellExecuteW(
                hwnd: *mut core::ffi::c_void,
                lp_operation: *const u16,
                lp_file: *const u16,
                lp_parameters: *const u16,
                lp_directory: *const u16,
                n_show_cmd: i32,
            ) -> isize;
        }

        let op: Vec<u16> = OsStr::new("open").encode_wide().chain(Some(0)).collect();
        let file: Vec<u16> = OsStr::new(trimmed).encode_wide().chain(Some(0)).collect();
        let res = unsafe {
            ShellExecuteW(
                core::ptr::null_mut(),
                op.as_ptr(),
                file.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                1, // SW_SHOWNORMAL
            )
        };
        if res <= 32 {
            return Err(format!("ShellExecuteW failed with code {}", res));
        }
        return Ok(());
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = trimmed;
        Err("External URL opener is only configured for Windows".to_string())
    }
}

pub fn handle_run_separation(
    state: &Arc<Mutex<StudioAppState>>,
    mode: SeparationMode,
    use_directml: bool,
) -> Result<SeparationStemPeaksResponse, String> {
    let (track, models_dir, env) = {
        let mut guard = state
            .lock()
            .map_err(|_| "Failed to acquire studio state lock".to_string())?;

        let track = guard
            .original_track
            .clone()
            .ok_or_else(|| "No audio/video track loaded yet".to_string())?;

        if guard.ort_env.is_none() {
            guard.ort_env = Some(init_ort_environment(None)?);
        }

        let env = guard
            .ort_env
            .clone()
            .ok_or_else(|| "ONNX Runtime environment not initialized".to_string())?;
        let models_dir = guard.model_manager.models_dir().to_path_buf();
        (track, models_dir, env)
    };

    let pipeline = match mode {
        SeparationMode::FastMdx => SeparationPipelineConfig::default_mdx_pipeline(),
        SeparationMode::ProRoformer => SeparationPipelineConfig::pro_roformer_pipeline(),
    };

    let device = if use_directml {
        ExecutionDevice::DirectMl { device_id: 0 }
    } else {
        ExecutionDevice::Cpu { intra_threads: 4 }
    };

    let separated = separate_3_stage_stems(&env, &models_dir, &track, &pipeline, device, |_| {})?;

    let response = SeparationStemPeaksResponse {
        lead_vocal: separated.lead_vocal.compute_waveform_peaks(140),
        backing_vocal: separated.backing_vocal.compute_waveform_peaks(140),
        instrumental: separated.instrumental.compute_waveform_peaks(140),
        reverb_tail: separated.reverb_tail.compute_waveform_peaks(140),
        execution_backend: if use_directml {
            "ONNX Runtime DirectML GPU (ort load-dynamic)".to_string()
        } else {
            "ONNX Runtime CPU Multi-Threaded (ort load-dynamic)".to_string()
        },
    };

    let mut guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;
    guard.separated_stems = Some(separated);

    Ok(response)
}

pub fn handle_export_audio(
    state: &Arc<Mutex<StudioAppState>>,
    output_dir: &str,
    base_name: &str,
    target: ExportTargetMode,
    format: AudioExportFormat,
    stem_params: Vec<StemChannelParams>,
    limiter_enabled: bool,
) -> Result<ExportAudioResponse, String> {
    let guard = state
        .lock()
        .map_err(|_| "Failed to acquire studio state lock".to_string())?;

    let ext = match format {
        AudioExportFormat::Flac24BitLossless => "flac",
        _ => "wav",
    };

    let out_root = PathBuf::from(output_dir);
    std::fs::create_dir_all(&out_root)
        .map_err(|e| format!("Failed to create export directory '{}': {}", out_root.display(), e))?;

    let mut exported_paths = Vec::new();

    if let Some(sep) = &guard.separated_stems {
        let find_param = |role: StemRole, default_vol: f32| -> StemChannelParams {
            stem_params
                .iter()
                .find(|p| p.role == role)
                .cloned()
                .unwrap_or(StemChannelParams {
                    role,
                    volume_percent: default_vol,
                    pan: 0.0,
                    muted: false,
                    solo: false,
                    formant_locked: matches!(role, StemRole::LeadVocal | StemRole::BackingVocal),
                })
        };

        let tracks = vec![
            StemTrackData {
                params: find_param(StemRole::LeadVocal, 100.0),
                audio: sep.lead_vocal.clone(),
            },
            StemTrackData {
                params: find_param(StemRole::BackingVocal, 90.0),
                audio: sep.backing_vocal.clone(),
            },
            StemTrackData {
                params: find_param(StemRole::Instrumental, 100.0),
                audio: sep.instrumental.clone(),
            },
            StemTrackData {
                params: find_param(StemRole::ReverbTail, 55.0),
                audio: sep.reverb_tail.clone(),
            },
        ];

        match target {
            ExportTargetMode::AllStems => {
                let stem_names = [
                    (StemRole::LeadVocal, "lead_vocal"),
                    (StemRole::BackingVocal, "backing_vocal"),
                    (StemRole::Instrumental, "instrumental"),
                    (StemRole::ReverbTail, "reverb_tail"),
                ];
                for (role, suffix) in stem_names {
                    let filtered: Vec<StemTrackData> = tracks
                        .iter()
                        .filter(|t| t.params.role == role)
                        .cloned()
                        .map(|mut t| {
                            t.params.muted = false;
                            t.params.solo = true;
                            t
                        })
                        .collect();
                    let rendered =
                        render_stem_aware_master_mix(&filtered, &guard.pitch_speed, limiter_enabled);
                    let out_path = out_root.join(format!("{}_{}.{}", base_name, suffix, ext));
                    export_audio_file(&out_path, &rendered, format)?;
                    exported_paths.push(out_path.display().to_string());
                }
            }
            ExportTargetMode::KaraokeBacking => {
                let karaoke_tracks: Vec<StemTrackData> = tracks
                    .into_iter()
                    .map(|mut t| {
                        t.params.solo = false;
                        t.params.muted = matches!(
                            t.params.role,
                            StemRole::LeadVocal | StemRole::ReverbTail
                        );
                        t
                    })
                    .collect();
                let rendered = render_stem_aware_master_mix(
                    &karaoke_tracks,
                    &guard.pitch_speed,
                    limiter_enabled,
                );
                let out_path = out_root.join(format!("{}_karaoke_with_backing.{}", base_name, ext));
                export_audio_file(&out_path, &rendered, format)?;
                exported_paths.push(out_path.display().to_string());
            }
            ExportTargetMode::MasterMix => {
                let rendered =
                    render_stem_aware_master_mix(&tracks, &guard.pitch_speed, limiter_enabled);
                let out_path = out_root.join(format!("{}.{}", base_name, ext));
                export_audio_file(&out_path, &rendered, format)?;
                exported_paths.push(out_path.display().to_string());
            }
        }
    } else if let Some(raw) = &guard.original_track {
        let rendered = process_stereo_pitch_and_speed(raw, &guard.pitch_speed);
        let out_path = out_root.join(format!("{}.{}", base_name, ext));
        export_audio_file(&out_path, &rendered, format)?;
        exported_paths.push(out_path.display().to_string());
    } else {
        return Err("No loaded audio track available to export".to_string());
    }

    Ok(ExportAudioResponse {
        exported_paths,
        format,
    })
}

#[cfg(feature = "desktop-tauri")]
pub mod tauri_commands {
    use super::*;

    #[tauri::command]
    pub fn ipc_load_track(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        file_path: String,
    ) -> Result<LoadedTrackSummary, String> {
        handle_load_track(state.inner(), &file_path)
    }

    #[tauri::command]
    pub fn ipc_parse_youtube_url(raw_url: String) -> Result<ParsedYouTubeTarget, String> {
        handle_parse_youtube(&raw_url)
    }

    #[tauri::command]
    pub fn ipc_update_pitch_speed(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        config: PitchSpeedConfig,
    ) -> Result<f32, String> {
        handle_update_pitch_speed(state.inner(), config)
    }

    #[tauri::command]
    pub fn ipc_list_models(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
    ) -> Result<ModelsOverviewResponse, String> {
        handle_list_models(state.inner())
    }

    #[tauri::command]
    pub fn ipc_import_local_onnx(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        source_path: String,
        target_model_id: Option<String>,
    ) -> Result<CachedModelStatus, String> {
        handle_import_local_onnx(state.inner(), &source_path, target_model_id)
    }

    #[tauri::command]
    pub fn ipc_delete_model(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        file_name: String,
    ) -> Result<(), String> {
        handle_delete_model(state.inner(), &file_name)
    }

    #[tauri::command]
    pub async fn ipc_download_model(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        model_id: String,
    ) -> Result<CachedModelStatus, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || handle_download_model(&state_arc, &model_id))
            .await
            .map_err(|e| format!("Download worker panicked: {}", e))?
    }

    #[tauri::command]
    pub fn ipc_open_external_url(url: String) -> Result<(), String> {
        handle_open_external_url(&url)
    }

    #[tauri::command]
    pub fn ipc_get_default_export_dir() -> Result<String, String> {
        handle_get_default_export_dir()
    }

    #[tauri::command]
    pub async fn ipc_pick_export_folder() -> Result<Option<String>, String> {
        tauri::async_runtime::spawn_blocking(handle_pick_export_folder)
            .await
            .map_err(|e| format!("Folder picker worker panicked: {}", e))?
    }

    #[tauri::command]
    pub fn ipc_save_exported_file(
        output_dir: String,
        file_name: String,
        bytes: Vec<u8>,
    ) -> Result<String, String> {
        handle_save_exported_file(&output_dir, &file_name, &bytes)
    }

    #[tauri::command]
    pub fn ipc_open_local_folder(folder_path: String) -> Result<(), String> {
        handle_open_local_folder(&folder_path)
    }

    #[tauri::command]
    pub fn ipc_run_separation(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        mode: SeparationMode,
        use_directml: bool,
    ) -> Result<SeparationStemPeaksResponse, String> {
        handle_run_separation(state.inner(), mode, use_directml)
    }

    #[tauri::command]
    pub fn ipc_export_audio(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        output_dir: String,
        base_name: String,
        target: ExportTargetMode,
        format: AudioExportFormat,
        stem_params: Vec<StemChannelParams>,
        limiter_enabled: bool,
    ) -> Result<ExportAudioResponse, String> {
        handle_export_audio(
            state.inner(),
            &output_dir,
            &base_name,
            target,
            format,
            stem_params,
            limiter_enabled,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_state_pitch_and_youtube_bridge() {
        let temp_models = std::env::temp_dir().join("music_studio_ipc_test_models");
        let state = Arc::new(Mutex::new(
            StudioAppState::new(&temp_models).expect("init state"),
        ));

        let yt = handle_parse_youtube(
            "https://www.youtube.com/watch?v=kJQP7kiw5Fk&list=PLMC9KNkIncKtP&index=3",
        )
        .expect("parse youtube playlist url");
        assert_eq!(yt.video_id, "kJQP7kiw5Fk");
        assert!(yt.was_in_playlist);
        assert_eq!(yt.playlist_index, Some(3));

        let ratio = handle_update_pitch_speed(
            &state,
            PitchSpeedConfig {
                semitones: 12.0,
                cents: 0.0,
                speed_ratio: 1.0,
                ..Default::default()
            },
        )
        .expect("update pitch speed");
        assert!((ratio - 2.0).abs() < 1e-5);

        let overview = handle_list_models(&state).expect("list models");
        assert_eq!(overview.catalog.len(), 6);

        let default_export = resolve_default_export_dir();
        assert!(default_export.ends_with("Music Studio AI"));

        let temp_export = std::env::temp_dir().join("music_studio_ipc_test_exports");
        let saved_path = handle_save_exported_file(
            &temp_export.display().to_string(),
            "subdir/mix_test.wav",
            b"RIFFtest",
        )
        .expect("save exported file");
        assert!(saved_path.ends_with("mix_test.wav"));
        assert_eq!(std::fs::read(&saved_path).expect("read saved"), b"RIFFtest");

        let _ = std::fs::remove_dir_all(&temp_export);
        let _ = std::fs::remove_dir_all(&temp_models);
    }
}
