//! Stage 4: Local AI Model Manager (`model_manager`).
//! Handles catalog management, resumable HuggingFace HTTPS downloads (via `ureq` + Windows SChannel),
//! streaming SHA-256 integrity verification (`sha2`), local `.onnx` file import, and `onnxruntime.dll` discovery.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ureq::config::Config as HttpConfig;
use ureq::tls::{TlsConfig, TlsProvider};
use ureq::Agent;

use crate::ai_separator::StftModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRole {
    InstrumentalSeparator,
    KaraokeBackingSeparator,
    DeReverb,
    VocalPitchEstimator,
    VocalNeuralSynthesizer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub file_name: String,
    pub role: ModelRole,
    pub task_title: String,
    pub architecture: String,
    pub sdr_label: String,
    pub approx_size_mb: u64,
    pub download_url: String,
    pub expected_sha256: Option<String>,
    pub stft_spec: StftModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModelStatus {
    pub id: String,
    pub file_name: String,
    pub installed: bool,
    pub local_path: Option<String>,
    pub size_bytes: u64,
    pub partial_bytes: u64,
    pub sha256_hex: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgressEvent {
    pub model_id: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub percent: f32,
    pub speed_kbps: f32,
    pub status: DownloadStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadStage {
    Connecting,
    Downloading,
    VerifyingSha256,
    Paused,
    Completed,
}

/// Returns the writable local directory for caching `.onnx` models (`%LOCALAPPDATA%\com.musicstudio.ai\models` on Windows).
pub fn resolve_default_models_dir() -> PathBuf {
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local_app_data)
            .join("com.musicstudio.ai")
            .join("models");
    }
    if let Some(xdg_data) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(xdg_data)
            .join("com.musicstudio.ai")
            .join("models");
    }
    PathBuf::from("models")
}

/// Returns the default open-source ONNX model catalog for Music Studio AI.
pub fn default_model_catalog() -> Vec<ModelCatalogEntry> {
    vec![
        ModelCatalogEntry {
            id: "uvr_mdx_inst_hq1".to_string(),
            file_name: "UVR-MDX-NET-Inst_HQ_1.onnx".to_string(),
            role: ModelRole::InstrumentalSeparator,
            task_title: "Stage 1: Instrumental & All Vocals (Fast MDX-Net)".to_string(),
            architecture: "MDX-Net (Hybrid Time-Frequency U-Net)".to_string(),
            sdr_label: "12.0 dB".to_string(),
            approx_size_mb: 64,
            download_url: "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/UVR-MDX-NET-Inst_HQ_1.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "UVR-MDX-NET-Inst_HQ_1.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 6144,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 256,
                compensate: 1.019,
                outputs_vocals: false,
            },
        },
        ModelCatalogEntry {
            id: "mel_band_roformer_vocals".to_string(),
            file_name: "mel-roformer.fp16.onnx".to_string(),
            role: ModelRole::InstrumentalSeparator,
            task_title: "Stage 1: Instrumental & All Vocals (Pro Mel-Band RoFormer)".to_string(),
            architecture: "Mel-Band RoFormer FP16 (Rotary Transformer)".to_string(),
            sdr_label: "13.1 dB".to_string(),
            approx_size_mb: 443,
            download_url: "https://huggingface.co/phuongnse/mel-roformer-kim-onnx/resolve/main/mel-roformer.fp16.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "mel-roformer.fp16.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 2048,
                hop_length: 441,
                dim_f: 1025,
                dim_t: 801,
                compensate: 1.0,
                outputs_vocals: true,
            },
        },
        ModelCatalogEntry {
            id: "htdemucs_4stems".to_string(),
            file_name: "htdemucs.onnx".to_string(),
            role: ModelRole::InstrumentalSeparator,
            task_title: "Stage 1: Hybrid Transformer Demucs (4-Stem Beat + Vocal)".to_string(),
            architecture: "HTDemucs (Hybrid Time-Frequency Transformer)".to_string(),
            sdr_label: "11.6 dB".to_string(),
            approx_size_mb: 302,
            download_url: "https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "htdemucs.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 4096,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 256,
                compensate: 1.0,
                outputs_vocals: false,
            },
        },
        ModelCatalogEntry {
            id: "uvr_mdx_kara_2".to_string(),
            file_name: "UVR_MDXNET_KARA_2.onnx".to_string(),
            role: ModelRole::KaraokeBackingSeparator,
            task_title: "Stage 2: Lead Vocal & Backing Harmony (Fast MDX-Net)".to_string(),
            architecture: "MDX-Net Karaoke / Backing Extractor".to_string(),
            sdr_label: "10.4 dB".to_string(),
            approx_size_mb: 50,
            download_url: "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/UVR_MDXNET_KARA_2.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "UVR_MDXNET_KARA_2.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 5120,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 256,
                compensate: 1.065,
                outputs_vocals: false,
            },
        },
        ModelCatalogEntry {
            id: "mel_band_roformer_karaoke".to_string(),
            file_name: "mel_band_roformer_karaoke_aufr33_viperx.onnx".to_string(),
            role: ModelRole::KaraokeBackingSeparator,
            task_title: "Stage 2: Lead Vocal & Backing Harmony (Pro RoFormer Karaoke)".to_string(),
            architecture: "Mel-Band RoFormer Karaoke ViperX".to_string(),
            sdr_label: "10.8 dB".to_string(),
            approx_size_mb: 876,
            download_url: "https://huggingface.co/bdsqlsz/mel_band_roformer_karaoke_aufr33-ONNX/resolve/main/model.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "mel_band_roformer_karaoke_aufr33_viperx.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 4096,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 352,
                compensate: 1.0,
                outputs_vocals: false,
            },
        },
        ModelCatalogEntry {
            id: "reverb_hq_foxjoy".to_string(),
            file_name: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
            role: ModelRole::DeReverb,
            task_title: "Stage 3: Dry Lead Vocal & Reverb Tail De-Reverb".to_string(),
            architecture: "MDX-Net De-Reverb HQ".to_string(),
            sdr_label: "11.4 dB".to_string(),
            approx_size_mb: 64,
            download_url: "https://huggingface.co/seanghay/uvr_models/resolve/main/Reverb_HQ_By_FoxJoy.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 512,
                compensate: 1.028,
                outputs_vocals: false,
            },
        },
        ModelCatalogEntry {
            id: "rmvpe_f0".to_string(),
            file_name: "rmvpe.onnx".to_string(),
            role: ModelRole::VocalPitchEstimator,
            task_title: "Neural Pitch Shifting: RMVPE F0 Estimator".to_string(),
            architecture: "Deep U-Net Pitch Estimator".to_string(),
            sdr_label: "Artifact-Free".to_string(),
            approx_size_mb: 345,
            download_url: "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx".to_string(),
            expected_sha256: None,
            stft_spec: StftModelConfig {
                model_filename: "rmvpe.onnx".to_string(),
                sample_rate: 16_000,
                n_fft: 1024,
                hop_length: 160,
                dim_f: 512,
                dim_t: 128,
                compensate: 1.0,
                outputs_vocals: false,
            },
        },
    ]
}

/// Manages local `.onnx` model storage, resumable HTTPS downloads, SHA-256 checks, and local imports.
#[derive(Clone)]
pub struct ModelManager {
    models_dir: PathBuf,
    catalog: Vec<ModelCatalogEntry>,
    http_agent: Agent,
}

impl ModelManager {
    /// Creates a new `ModelManager` rooted at `models_dir` (creating it if missing).
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Result<Self, String> {
        let dir = models_dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create models directory '{}': {}", dir.display(), e))?;

        let tls_config = TlsConfig::builder()
            .provider(TlsProvider::NativeTls)
            .build();
        let http_config = HttpConfig::builder().tls_config(tls_config).build();
        let http_agent = http_config.new_agent();

        Ok(Self {
            models_dir: dir,
            catalog: default_model_catalog(),
            http_agent,
        })
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn catalog(&self) -> &[ModelCatalogEntry] {
        &self.catalog
    }

    pub fn model_path(&self, file_name: &str) -> PathBuf {
        self.models_dir.join(file_name)
    }

    fn partial_path(&self, file_name: &str) -> PathBuf {
        self.models_dir.join(format!("{}.part", file_name))
    }

    /// Scans the local `./models/` cache directory for installed and partially downloaded `.onnx` files.
    pub fn scan_cached_models(&self) -> Vec<CachedModelStatus> {
        self.catalog
            .iter()
            .map(|entry| {
                let target = self.model_path(&entry.file_name);
                let partial = self.partial_path(&entry.file_name);

                let installed_meta = fs::metadata(&target).ok();
                let partial_bytes = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

                match installed_meta {
                    Some(meta) if meta.is_file() && meta.len() > 0 => CachedModelStatus {
                        id: entry.id.clone(),
                        file_name: entry.file_name.clone(),
                        installed: true,
                        local_path: Some(target.display().to_string()),
                        size_bytes: meta.len(),
                        partial_bytes: 0,
                        sha256_hex: None,
                    },
                    _ => CachedModelStatus {
                        id: entry.id.clone(),
                        file_name: entry.file_name.clone(),
                        installed: false,
                        local_path: None,
                        size_bytes: 0,
                        partial_bytes,
                        sha256_hex: None,
                    },
                }
            })
            .collect()
    }

    /// Downloads a model from HuggingFace with HTTP `Range` resume support, pause/cancel flag,
    /// streaming progress updates, and SHA-256 verification.
    pub fn download_model_resumable<F>(
        &self,
        model_id: &str,
        pause_or_cancel: Arc<AtomicBool>,
        mut on_progress: F,
    ) -> Result<CachedModelStatus, String>
    where
        F: FnMut(DownloadProgressEvent),
    {
        let entry = self
            .catalog
            .iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| format!("Unknown model id: {}", model_id))?
            .clone();

        let final_path = self.model_path(&entry.file_name);
        if let Ok(meta) = fs::metadata(&final_path) {
            if meta.is_file() && meta.len() > 0 {
                let sha = compute_file_sha256(&final_path)?;
                return Ok(CachedModelStatus {
                    id: entry.id,
                    file_name: entry.file_name,
                    installed: true,
                    local_path: Some(final_path.display().to_string()),
                    size_bytes: meta.len(),
                    partial_bytes: 0,
                    sha256_hex: Some(sha),
                });
            }
        }

        let part_path = self.partial_path(&entry.file_name);
        let existing_bytes = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);

        on_progress(DownloadProgressEvent {
            model_id: entry.id.clone(),
            downloaded_bytes: existing_bytes,
            total_bytes: None,
            percent: 0.0,
            speed_kbps: 0.0,
            status: DownloadStage::Connecting,
        });

        let mut request = self.http_agent.get(&entry.download_url);
        if existing_bytes > 0 {
            request = request.header("Range", &format!("bytes={}-", existing_bytes));
        }

        let mut response = request
            .call()
            .map_err(|e| format!("Failed to download '{}': {}", entry.file_name, e))?;

        let status_code = response.status().as_u16();
        let is_resumed = status_code == 206 && existing_bytes > 0;
        let content_length = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        let total_bytes = content_length.map(|len| if is_resumed { existing_bytes + len } else { len });

        let mut file = if is_resumed {
            OpenOptions::new()
                .append(true)
                .open(&part_path)
                .map_err(|e| format!("Failed to open partial file for resume: {}", e))?
        } else {
            File::create(&part_path)
                .map_err(|e| format!("Failed to create partial file '{}': {}", part_path.display(), e))?
        };

        let mut downloaded = if is_resumed { existing_bytes } else { 0 };
        let mut reader = response.body_mut().as_reader();
        let mut buf = vec![0_u8; 64 * 1024];
        let started_at = Instant::now();
        let mut bytes_this_session = 0_u64;

        loop {
            if pause_or_cancel.load(Ordering::Relaxed) {
                file.flush().map_err(|e| e.to_string())?;
                let percent = total_bytes
                    .filter(|&t| t > 0)
                    .map(|t| (downloaded as f32 / t as f32) * 100.0)
                    .unwrap_or(0.0);
                on_progress(DownloadProgressEvent {
                    model_id: entry.id.clone(),
                    downloaded_bytes: downloaded,
                    total_bytes,
                    percent,
                    speed_kbps: 0.0,
                    status: DownloadStage::Paused,
                });
                return Ok(CachedModelStatus {
                    id: entry.id,
                    file_name: entry.file_name,
                    installed: false,
                    local_path: None,
                    size_bytes: 0,
                    partial_bytes: downloaded,
                    sha256_hex: None,
                });
            }

            let read_n = reader
                .read(&mut buf)
                .map_err(|e| format!("Network read error while downloading '{}': {}", entry.file_name, e))?;
            if read_n == 0 {
                break;
            }

            file.write_all(&buf[..read_n])
                .map_err(|e| format!("Disk write error: {}", e))?;

            downloaded += read_n as u64;
            bytes_this_session += read_n as u64;

            let elapsed = started_at.elapsed().as_secs_f32().max(0.05);
            let speed_kbps = (bytes_this_session as f32 / 1024.0) / elapsed;
            let percent = total_bytes
                .filter(|&t| t > 0)
                .map(|t| (downloaded as f32 / t as f32) * 100.0)
                .unwrap_or(0.0);

            on_progress(DownloadProgressEvent {
                model_id: entry.id.clone(),
                downloaded_bytes: downloaded,
                total_bytes,
                percent: percent.clamp(0.0, 99.9),
                speed_kbps,
                status: DownloadStage::Downloading,
            });
        }

        file.flush().map_err(|e| e.to_string())?;
        drop(file);

        on_progress(DownloadProgressEvent {
            model_id: entry.id.clone(),
            downloaded_bytes: downloaded,
            total_bytes: Some(downloaded),
            percent: 99.9,
            speed_kbps: 0.0,
            status: DownloadStage::VerifyingSha256,
        });

        validate_onnx_binary_file(&part_path)?;
        let sha_hex = compute_file_sha256(&part_path)?;

        if let Some(expected) = &entry.expected_sha256 {
            if !sha_hex.eq_ignore_ascii_case(expected) {
                let _ = fs::remove_file(&part_path);
                return Err(format!(
                    "SHA-256 verification failed for '{}': expected {}, got {}",
                    entry.file_name, expected, sha_hex
                ));
            }
        }

        fs::rename(&part_path, &final_path)
            .map_err(|e| format!("Failed to finalize model '{}': {}", final_path.display(), e))?;

        on_progress(DownloadProgressEvent {
            model_id: entry.id.clone(),
            downloaded_bytes: downloaded,
            total_bytes: Some(downloaded),
            percent: 100.0,
            speed_kbps: 0.0,
            status: DownloadStage::Completed,
        });

        Ok(CachedModelStatus {
            id: entry.id,
            file_name: entry.file_name,
            installed: true,
            local_path: Some(final_path.display().to_string()),
            size_bytes: downloaded,
            partial_bytes: 0,
            sha256_hex: Some(sha_hex),
        })
    }

    /// Imports an existing `.onnx` model file from the user's local filesystem into the offline cache.
    pub fn import_local_onnx_model<P: AsRef<Path>>(
        &self,
        source_path: P,
        target_model_id: Option<&str>,
    ) -> Result<CachedModelStatus, String> {
        let src = source_path.as_ref();
        if !src.is_file() {
            return Err(format!("Model file '{}' does not exist", src.display()));
        }

        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if ext != "onnx" {
            return Err("Only `.onnx` model files are supported for local import".to_string());
        }

        validate_onnx_binary_file(src)?;
        let sha_hex = compute_file_sha256(src)?;

        let (resolved_id, target_filename) = match target_model_id {
            Some(id) => {
                let entry = self
                    .catalog
                    .iter()
                    .find(|m| m.id == id)
                    .ok_or_else(|| format!("Unknown catalog model id: {}", id))?;
                (entry.id.clone(), entry.file_name.clone())
            }
            None => {
                let file_name = src
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| "Invalid file name".to_string())?
                    .to_string();
                let matched_id = self
                    .catalog
                    .iter()
                    .find(|m| m.file_name.eq_ignore_ascii_case(&file_name))
                    .map(|m| m.id.clone())
                    .unwrap_or_else(|| file_name.trim_end_matches(".onnx").to_string());
                (matched_id, file_name)
            }
        };

        let dest = self.model_path(&target_filename);
        let same_file = match (fs::canonicalize(src), fs::canonicalize(&dest)) {
            (Ok(s), Ok(d)) => s == d,
            _ => false,
        };
        let copied_bytes = if same_file {
            fs::metadata(&dest)
                .map(|m| m.len())
                .map_err(|e| format!("Failed to read model metadata '{}': {}", dest.display(), e))?
        } else {
            fs::copy(src, &dest)
                .map_err(|e| format!("Failed to copy model to cache '{}': {}", dest.display(), e))?
        };

        Ok(CachedModelStatus {
            id: resolved_id,
            file_name: target_filename,
            installed: true,
            local_path: Some(dest.display().to_string()),
            size_bytes: copied_bytes,
            partial_bytes: 0,
            sha256_hex: Some(sha_hex),
        })
    }

    /// Removes a cached `.onnx` or `.part` file from the local cache directory.
    pub fn delete_cached_model(&self, file_name: &str) -> Result<(), String> {
        let target = self.model_path(file_name);
        let partial = self.partial_path(file_name);
        if target.exists() {
            fs::remove_file(&target)
                .map_err(|e| format!("Failed to remove '{}': {}", target.display(), e))?;
        }
        if partial.exists() {
            fs::remove_file(&partial)
                .map_err(|e| format!("Failed to remove '{}': {}", partial.display(), e))?;
        }
        Ok(())
    }
}

/// Computes the streaming lowercase hex SHA-256 digest of a file without loading it entirely into RAM.
pub fn compute_file_sha256<P: AsRef<Path>>(path: P) -> Result<String, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open file for SHA-256 '{}': {}", path.as_ref().display(), e))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Failed to read file during SHA-256: {}", e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut hex, "{:02x}", byte);
    }
    Ok(hex)
}

/// Performs lightweight structural checks to ensure a file is a binary ONNX protobuf (not an HTML error page).
pub fn validate_onnx_binary_file<P: AsRef<Path>>(path: P) -> Result<(), String> {
    let mut file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open ONNX candidate '{}': {}", path.as_ref().display(), e))?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if meta.len() < 16 {
        return Err("File is too small to be a valid ONNX model".to_string());
    }

    let mut header = [0_u8; 16];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    if header.starts_with(b"<!DOCTYPE") || header.starts_with(b"<html") || header.starts_with(b"version https://") {
        return Err("Invalid ONNX file: appears to be an HTML error or Git LFS pointer rather than model weights".to_string());
    }

    Ok(())
}

/// Searches standard local paths (`ORT_DYLIB_PATH`, executable folder, `./runtime/`) for `onnxruntime.dll`.
pub fn discover_onnxruntime_library() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ORT_DYLIB_PATH") {
        let candidate = PathBuf::from(env_path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let dll_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let direct = parent.join(dll_name);
            if direct.is_file() {
                return Some(direct);
            }
            let runtime_sub = parent.join("runtime").join(dll_name);
            if runtime_sub.is_file() {
                return Some(runtime_sub);
            }
        }
    }

    let cwd_runtime = PathBuf::from("runtime").join(dll_name);
    if cwd_runtime.is_file() {
        return Some(cwd_runtime);
    }

    let models_dir_dll = resolve_default_models_dir().join(dll_name);
    if models_dir_dll.is_file() {
        return Some(models_dir_dll);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sha256_and_import_validation() {
        let temp_dir = std::env::temp_dir().join("music_studio_test_models");
        let _ = fs::remove_dir_all(&temp_dir);
        let manager = ModelManager::new(&temp_dir).expect("create manager");

        let sample_src = temp_dir.join("sample_custom.onnx");
        let payload = vec![0x08_u8, 0x09, 0x12, 0x07, b'p', b'y', b't', b'o', b'r', b'c', b'h', 0x3A, 0x10, 0x01, 0x02, 0x03, 0x04];
        fs::write(&sample_src, &payload).expect("write dummy onnx");

        let status = manager
            .import_local_onnx_model(&sample_src, Some("uvr_mdx_inst_hq1"))
            .expect("import valid onnx");
        assert!(status.installed);
        assert_eq!(status.file_name, "UVR-MDX-NET-Inst_HQ_1.onnx");
        assert_eq!(status.size_bytes, payload.len() as u64);
        assert_eq!(status.sha256_hex.as_ref().map(|s| s.len()), Some(64));

        // Reject HTML error payload disguised as .onnx
        let html_fake = temp_dir.join("bad_model.onnx");
        fs::write(&html_fake, b"<!DOCTYPE html><html>404 Not Found</html>").expect("write fake html");
        assert!(manager.import_local_onnx_model(&html_fake, None).is_err());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
