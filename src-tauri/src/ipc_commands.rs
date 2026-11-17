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
    decode_audio_file, encode_wav_16bit_bytes, export_audio_file, is_video_container_path,
    AudioExportFormat, LoadedTrackSummary, StereoAudioBuffer,
};
use crate::dsp_pitch_time::{
    detect_bpm, detect_musical_key, process_stereo_pitch_and_speed, PitchSpeedConfig,
};
use crate::model_manager::{
    discover_onnxruntime_library, CachedModelStatus, ModelCatalogEntry, ModelManager,
};
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

const B64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes raw bytes into standard Base64 without external crate dependencies.
pub fn encode_base64(input: &[u8]) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(((input.len() + 2) / 3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64_ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Decodes a standard Base64 string into bytes.
pub fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let mut lut = [255_u8; 256];
    for (idx, &ch) in B64_ALPHABET.iter().enumerate() {
        lut[ch as usize] = idx as u8;
    }

    let cleaned: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if cleaned.len() % 4 != 0 {
        return Err("Invalid Base64 length".to_string());
    }

    let mut out = Vec::with_capacity((cleaned.len() / 4) * 3);
    for chunk in cleaned.chunks_exact(4) {
        let v0 = lut[chunk[0] as usize];
        let v1 = lut[chunk[1] as usize];
        if v0 == 255 || v1 == 255 {
            return Err("Invalid Base64 character".to_string());
        }
        let v2 = if chunk[2] == b'=' { 0 } else { lut[chunk[2] as usize] };
        let v3 = if chunk[3] == b'=' { 0 } else { lut[chunk[3] as usize] };
        if (chunk[2] != b'=' && v2 == 255) || (chunk[3] != b'=' && v3 == 255) {
            return Err("Invalid Base64 padding/character".to_string());
        }
        let triple = ((v0 as u32) << 18) | ((v1 as u32) << 12) | ((v2 as u32) << 6) | (v3 as u32);
        out.push(((triple >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            out.push((triple & 0xFF) as u8);
        }
    }
    Ok(out)
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct OpenFileNameW {
    l_struct_size: u32,
    hwnd_owner: *mut core::ffi::c_void,
    h_instance: *mut core::ffi::c_void,
    lpstr_filter: *const u16,
    lpstr_custom_filter: *mut u16,
    n_max_cust_filter: u32,
    n_filter_index: u32,
    lpstr_file: *mut u16,
    n_max_file: u32,
    lpstr_file_title: *mut u16,
    n_max_file_title: u32,
    lpstr_initial_dir: *const u16,
    lpstr_title: *const u16,
    flags: u32,
    n_file_offset: u16,
    n_file_extension: u16,
    lpstr_def_ext: *const u16,
    l_cust_data: isize,
    lpfn_hook: *const core::ffi::c_void,
    lp_template_name: *const u16,
    pv_reserved: *mut core::ffi::c_void,
    dw_reserved: u32,
    flags_ex: u32,
}

#[cfg(target_os = "windows")]
fn pick_windows_file(title_text: &str, filter_spec: &str) -> Result<Option<String>, String> {
    use std::ffi::{OsStr, OsString};
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    #[link(name = "comdlg32")]
    extern "system" {
        fn GetOpenFileNameW(lpofn: *mut OpenFileNameW) -> i32;
    }

    let filter: Vec<u16> = OsStr::new(filter_spec).encode_wide().collect();
    let title: Vec<u16> = OsStr::new(title_text)
        .encode_wide()
        .chain(Some(0))
        .collect();

    let mut file_buf = vec![0_u16; 2048];
    let mut ofn = OpenFileNameW {
        l_struct_size: std::mem::size_of::<OpenFileNameW>() as u32,
        hwnd_owner: core::ptr::null_mut(),
        h_instance: core::ptr::null_mut(),
        lpstr_filter: filter.as_ptr(),
        lpstr_custom_filter: core::ptr::null_mut(),
        n_max_cust_filter: 0,
        n_filter_index: 1,
        lpstr_file: file_buf.as_mut_ptr(),
        n_max_file: file_buf.len() as u32,
        lpstr_file_title: core::ptr::null_mut(),
        n_max_file_title: 0,
        lpstr_initial_dir: core::ptr::null(),
        lpstr_title: title.as_ptr(),
        flags: 0x0000_1808, // OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
        n_file_offset: 0,
        n_file_extension: 0,
        lpstr_def_ext: core::ptr::null(),
        l_cust_data: 0,
        lpfn_hook: core::ptr::null(),
        lp_template_name: core::ptr::null(),
        pv_reserved: core::ptr::null_mut(),
        dw_reserved: 0,
        flags_ex: 0,
    };

    let ok = unsafe { GetOpenFileNameW(&mut ofn) };
    if ok == 0 {
        return Ok(None);
    }

    let len = file_buf.iter().position(|&c| c == 0).unwrap_or(file_buf.len());
    Ok(Some(
        OsString::from_wide(&file_buf[..len])
            .to_string_lossy()
            .into_owned(),
    ))
}

/// Opens a native Windows file picker dialog for audio and video files without external `.exe` sidecars.
pub fn handle_pick_audio_file() -> Result<Option<String>, String> {
    #[cfg(target_os = "windows")]
    {
        return pick_windows_file(
            "Mo file Nhac hoac Video - Music Studio AI",
            "Nhac & Video (*.mp3;*.wav;*.flac;*.ogg;*.m4a;*.aac;*.opus;*.mp4;*.mkv;*.webm;*.mov;*.avi)\0\
             *.mp3;*.wav;*.flac;*.ogg;*.m4a;*.aac;*.opus;*.mp4;*.mkv;*.webm;*.mov;*.avi\0\
             Tat ca file (*.*)\0*.*\0\0",
        );
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err("Native file picker is only supported on Windows".to_string())
    }
}

/// Opens a native Windows file picker dialog for local `.onnx` model files.
pub fn handle_pick_onnx_file() -> Result<Option<String>, String> {
    #[cfg(target_os = "windows")]
    {
        return pick_windows_file(
            "Chon file Model AI (.onnx) - Music Studio AI",
            "ONNX Model (*.onnx)\0*.onnx\0Tat ca file (*.*)\0*.*\0\0",
        );
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err("Native file picker is only supported on Windows".to_string())
    }
}

/// Reads a local audio/video file up to 120 MB as Base64 for WebAudio preview decoding.
pub fn handle_read_file_base64(file_path: &str) -> Result<String, String> {
    let meta = std::fs::metadata(file_path)
        .map_err(|e| format!("Failed to read file metadata '{}': {}", file_path, e))?;
    if meta.len() > 120 * 1024 * 1024 {
        return Err("File vuot qua gioi han 120 MB cho phat preview truc tiep".to_string());
    }
    let bytes = std::fs::read(file_path)
        .map_err(|e| format!("Failed to read file '{}': {}", file_path, e))?;
    Ok(encode_base64(&bytes))
}

/// Encodes the currently loaded Rust track into 16-bit WAV Base64 for containers Chromium cannot decode directly (e.g. MKV/FLAC).
pub fn handle_prepare_preview_wav_b64(state: &Arc<Mutex<StudioAppState>>) -> Result<String, String> {
    let track = {
        let guard = state
            .lock()
            .map_err(|_| "Failed to acquire studio state lock".to_string())?;
        guard
            .original_track
            .clone()
            .ok_or_else(|| "No track loaded in Rust studio state".to_string())?
    };
    let wav_bytes = encode_wav_16bit_bytes(&track)?;
    Ok(encode_base64(&wav_bytes))
}

/// Loads raw audio/video bytes received via drag-and-drop into Rust StudioAppState.
pub fn handle_load_audio_bytes_b64(
    state: &Arc<Mutex<StudioAppState>>,
    file_name: &str,
    audio_b64: &str,
) -> Result<LoadedTrackSummary, String> {
    let bytes = decode_base64(audio_b64)?;
    let safe_name = Path::new(file_name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("dropped_track.wav");
    let pid = std::process::id();
    let temp_path = std::env::temp_dir().join(format!("music_studio_drop_{}_{}", pid, safe_name));
    std::fs::write(&temp_path, &bytes)
        .map_err(|e| format!("Failed to buffer dropped file: {}", e))?;

    let result = handle_load_track(state, &temp_path.display().to_string());
    let _ = std::fs::remove_file(&temp_path);

    let mut summary = result?;
    summary.file_name = safe_name.to_string();
    summary.is_video_source = is_video_container_path(Path::new(safe_name));
    Ok(summary)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparatedStemsWavB64Response {
    pub lead_vocal_b64: String,
    pub backing_vocal_b64: String,
    pub instrumental_b64: String,
    pub reverb_tail_b64: String,
}

/// Exports the 4 Rust ONNX-separated stems as Base64 16-bit WAV buffers for live WebView stem mixing.
pub fn handle_export_separated_stems_b64(
    state: &Arc<Mutex<StudioAppState>>,
) -> Result<SeparatedStemsWavB64Response, String> {
    let sep = {
        let guard = state
            .lock()
            .map_err(|_| "Failed to acquire studio state lock".to_string())?;
        guard
            .separated_stems
            .clone()
            .ok_or_else(|| "No separated stems in Rust state".to_string())?
    };

    Ok(SeparatedStemsWavB64Response {
        lead_vocal_b64: encode_base64(&encode_wav_16bit_bytes(&sep.lead_vocal)?),
        backing_vocal_b64: encode_base64(&encode_wav_16bit_bytes(&sep.backing_vocal)?),
        instrumental_b64: encode_base64(&encode_wav_16bit_bytes(&sep.instrumental)?),
        reverb_tail_b64: encode_base64(&encode_wav_16bit_bytes(&sep.reverb_tail)?),
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
    let pipeline = match mode {
        SeparationMode::FastMdx => SeparationPipelineConfig::default_mdx_pipeline(),
        SeparationMode::ProRoformer => SeparationPipelineConfig::pro_roformer_pipeline(),
    };

    let (track, models_dir, env) = {
        let mut guard = state
            .lock()
            .map_err(|_| "Failed to acquire studio state lock".to_string())?;

        let track = guard
            .original_track
            .clone()
            .ok_or_else(|| "Chưa có bài hát nào được nạp vào bộ xử lý Rust.".to_string())?;

        let models_dir = guard.model_manager.models_dir().to_path_buf();

        let stage1_path = models_dir.join(&pipeline.stage1_inst_model.model_filename);
        if !stage1_path.is_file() {
            return Err(format!(
                "ONNX_PREREQ_MISSING: Thiếu model tầng 1 '{}' trong '{}'. Hãy tải hoặc nhập file .ONNX trong mục Model AI.",
                pipeline.stage1_inst_model.model_filename,
                models_dir.display()
            ));
        }
        let stage2_path = models_dir.join(&pipeline.stage2_karaoke_model.model_filename);
        if !stage2_path.is_file() {
            return Err(format!(
                "ONNX_PREREQ_MISSING: Thiếu model tầng 2 '{}' trong '{}'. Hãy tải hoặc nhập file .ONNX trong mục Model AI.",
                pipeline.stage2_karaoke_model.model_filename,
                models_dir.display()
            ));
        }

        if discover_onnxruntime_library().is_none() {
            return Err(
                "ONNX_PREREQ_MISSING: Không tìm thấy file 'onnxruntime.dll' (Microsoft signed). Vui lòng đặt onnxruntime.dll cạnh file thực thi hoặc trong thư mục models."
                    .to_string(),
            );
        }

        if guard.ort_env.is_none() {
            guard.ort_env = Some(init_ort_environment(None)?);
        }

        let env = guard
            .ort_env
            .clone()
            .ok_or_else(|| "ONNX Runtime environment not initialized".to_string())?;
        (track, models_dir, env)
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
            "AI ONNX (GPU / DirectML)".to_string()
        } else {
            "AI ONNX (CPU đa luồng)".to_string()
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
    pub async fn ipc_pick_audio_file() -> Result<Option<String>, String> {
        tauri::async_runtime::spawn_blocking(handle_pick_audio_file)
            .await
            .map_err(|e| format!("Audio picker worker panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_pick_onnx_file() -> Result<Option<String>, String> {
        tauri::async_runtime::spawn_blocking(handle_pick_onnx_file)
            .await
            .map_err(|e| format!("ONNX picker worker panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_load_track(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        file_path: String,
    ) -> Result<LoadedTrackSummary, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || handle_load_track(&state_arc, &file_path))
            .await
            .map_err(|e| format!("Track loader panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_load_audio_bytes_b64(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        file_name: String,
        audio_b64: String,
    ) -> Result<LoadedTrackSummary, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || {
            handle_load_audio_bytes_b64(&state_arc, &file_name, &audio_b64)
        })
        .await
        .map_err(|e| format!("Audio bytes loader panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_read_file_base64(file_path: String) -> Result<String, String> {
        tauri::async_runtime::spawn_blocking(move || handle_read_file_base64(&file_path))
            .await
            .map_err(|e| format!("File reader panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_prepare_preview_wav_b64(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
    ) -> Result<String, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || handle_prepare_preview_wav_b64(&state_arc))
            .await
            .map_err(|e| format!("Preview encoder panicked: {}", e))?
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
    pub async fn ipc_import_local_onnx(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        source_path: String,
        target_model_id: Option<String>,
    ) -> Result<CachedModelStatus, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || {
            handle_import_local_onnx(&state_arc, &source_path, target_model_id)
        })
        .await
        .map_err(|e| format!("ONNX import worker panicked: {}", e))?
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
    pub async fn ipc_run_separation(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        mode: SeparationMode,
        use_directml: bool,
    ) -> Result<SeparationStemPeaksResponse, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || {
            handle_run_separation(&state_arc, mode, use_directml)
        })
        .await
        .map_err(|e| format!("Separation worker panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_export_separated_stems_b64(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
    ) -> Result<SeparatedStemsWavB64Response, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || handle_export_separated_stems_b64(&state_arc))
            .await
            .map_err(|e| format!("Separated stem encoder panicked: {}", e))?
    }

    #[tauri::command]
    pub async fn ipc_export_audio(
        state: tauri::State<'_, Arc<Mutex<StudioAppState>>>,
        output_dir: String,
        base_name: String,
        target: ExportTargetMode,
        format: AudioExportFormat,
        stem_params: Vec<StemChannelParams>,
        limiter_enabled: bool,
    ) -> Result<ExportAudioResponse, String> {
        let state_arc = Arc::clone(state.inner());
        tauri::async_runtime::spawn_blocking(move || {
            handle_export_audio(
                &state_arc,
                &output_dir,
                &base_name,
                target,
                format,
                stem_params,
                limiter_enabled,
            )
        })
        .await
        .map_err(|e| format!("Export worker panicked: {}", e))?
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

    #[test]
    fn test_base64_roundtrip_and_real_wav_load_and_missing_onnx_guard() {
        let sample = b"MusicStudioAI-Real-Audio-Test";
        let b64 = encode_base64(sample);
        assert_eq!(decode_base64(&b64).expect("decode b64"), sample);

        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        assert_eq!(std::mem::size_of::<OpenFileNameW>(), 152);

        let temp_models = std::env::temp_dir().join("music_studio_ipc_real_wav_test");
        let state = Arc::new(Mutex::new(
            StudioAppState::new(&temp_models).expect("init state"),
        ));

        let buf = StereoAudioBuffer::new(
            44_100,
            vec![0.25_f32; 44_100],
            vec![-0.25_f32; 44_100],
        );
        let wav_bytes = encode_wav_16bit_bytes(&buf).expect("encode preview wav");
        let wav_b64 = encode_base64(&wav_bytes);

        let summary = handle_load_audio_bytes_b64(&state, "real_test.wav", &wav_b64)
            .expect("load dropped wav into Rust state");
        assert_eq!(summary.file_name, "real_test.wav");
        assert!(summary.duration_sec > 0.9);

        let preview_b64 = handle_prepare_preview_wav_b64(&state).expect("prepare preview wav");
        assert!(!preview_b64.is_empty());

        let sep_err = handle_run_separation(&state, SeparationMode::FastMdx, false)
            .expect_err("should fail clearly when .onnx models are missing");
        assert!(sep_err.starts_with("ONNX_PREREQ_MISSING:"));

        let _ = std::fs::remove_dir_all(&temp_models);
    }
}
