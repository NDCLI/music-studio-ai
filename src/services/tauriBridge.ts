import { PitchSpeedSettings } from '../types/studio';

interface TauriInternals {
  invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: TauriInternals;
  }
}

/**
 * Returns true when running inside a native Tauri v2 desktop container
 * with the Rust single-binary IPC backend active.
 */
export function isTauriDesktopRuntime(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.__TAURI_INTERNALS__?.invoke === 'function'
  );
}

/**
 * Invokes a registered Rust `#[tauri::command]` when running inside Tauri Desktop.
 */
export async function invokeRustCommand<T>(
  commandName: string,
  args?: Record<string, unknown>
): Promise<T> {
  if (!isTauriDesktopRuntime()) {
    throw new Error(
      `Command '${commandName}' requires the Tauri Desktop runtime (--features desktop-tauri).`
    );
  }
  return window.__TAURI_INTERNALS__!.invoke<T>(commandName, args);
}

/**
 * Opens an external HTTPS URL in the user's system browser (Edge/Chrome),
 * routing through Win32 ShellExecuteW inside Tauri Desktop where target="_blank" is blocked.
 */
export async function openExternalUrl(url: string): Promise<void> {
  if (isTauriDesktopRuntime()) {
    await invokeRustCommand<void>('ipc_open_external_url', { url });
    return;
  }
  window.open(url, '_blank', 'noopener,noreferrer');
}

export interface RustCachedModelStatus {
  id: string;
  file_name: string;
  installed: boolean;
  local_path?: string;
  size_bytes: number;
  partial_bytes: number;
  sha256_hex?: string;
}

export async function listModelsViaRustBackend(): Promise<{
  models_dir: string;
  cached_status: RustCachedModelStatus[];
} | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand('ipc_list_models');
}

export async function downloadModelViaRustBackend(
  modelId: string
): Promise<RustCachedModelStatus | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand('ipc_download_model', { modelId });
}

export async function deleteModelViaRustBackend(
  fileName: string
): Promise<void> {
  if (!isTauriDesktopRuntime()) return;
  await invokeRustCommand('ipc_delete_model', { fileName });
}

export interface RustPitchSpeedPayload {
  semitones: number;
  cents: number;
  speed_ratio: number;
  preserve_formants: boolean;
  stem_aware_mode: boolean;
  transient_crisp_lock: boolean;
  engine: 'SignalsmithR3' | 'WorldVocoder' | 'RmvpeHiFiGan';
}

export function mapPitchSettingsToRust(
  settings: PitchSpeedSettings
): RustPitchSpeedPayload {
  const engineMap: Record<
    PitchSpeedSettings['pitchEngine'],
    RustPitchSpeedPayload['engine']
  > = {
    signalsmith_r3: 'SignalsmithR3',
    world_vocoder: 'WorldVocoder',
    rmvpe_hifigan: 'RmvpeHiFiGan',
  };

  return {
    semitones: settings.semitones,
    cents: settings.cents,
    speed_ratio: settings.speed,
    preserve_formants: settings.formantPreservation,
    stem_aware_mode: settings.stemAwareProcessing,
    transient_crisp_lock: settings.transientCrispLock,
    engine: engineMap[settings.pitchEngine],
  };
}

/**
 * Synchronizes Pitch & Speed DSP parameters with the Rust backend when inside Tauri Desktop.
 */
export async function downloadYouTubeAudioViaRust(rawUrl: string): Promise<string | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<string>('ipc_download_youtube_audio', { rawUrl });
}

export async function downloadAndInstallUpdateViaRust(
  installerUrl: string,
  fileName: string
): Promise<void> {
  if (!isTauriDesktopRuntime()) {
    throw new Error('Cập nhật trực tiếp chỉ khả dụng trên bản Desktop (.exe).');
  }
  await invokeRustCommand<void>('ipc_download_and_install_update', {
    installerUrl,
    fileName,
  });
}

export async function syncPitchSpeedToRustBackend(
  settings: PitchSpeedSettings
): Promise<number | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<number>('ipc_update_pitch_speed', {
    config: mapPitchSettingsToRust(settings),
  });
}

export async function parseYouTubeViaRustBackend(rawUrl: string): Promise<{
  video_id: string;
  canonical_watch_url: string;
  was_in_playlist: boolean;
  ignored_playlist_id?: string;
  playlist_index?: number;
  start_time_sec?: number;
} | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand('ipc_parse_youtube_url', { rawUrl });
}

/**
 * Resolves the default local export directory (%USERPROFILE%\Music\Music Studio AI).
 */
export async function getDefaultExportDirViaRust(): Promise<string | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<string>('ipc_get_default_export_dir');
}

/**
 * Opens a native Windows folder picker dialog without spawning child .exe processes.
 */
export async function pickExportFolderViaRust(): Promise<string | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<string | null>('ipc_pick_export_folder');
}

/**
 * Writes rendered audio bytes directly to the chosen Windows directory.
 */
export async function saveExportedFileViaRust(
  outputDir: string,
  fileName: string,
  bytes: Uint8Array
): Promise<string> {
  return invokeRustCommand<string>('ipc_save_exported_file', {
    outputDir,
    fileName,
    bytes: Array.from(bytes),
  });
}

/**
 * Opens a local folder in Windows Explorer via Win32 ShellExecuteW.
 */
export async function openLocalFolderViaRust(folderPath: string): Promise<void> {
  if (!isTauriDesktopRuntime()) return;
  await invokeRustCommand<void>('ipc_open_local_folder', { folderPath });
}

export interface RustLoadedTrackSummary {
  file_name: string;
  is_video_source: boolean;
  sample_rate: number;
  channels: number;
  duration_sec: number;
  detected_key: string;
  detected_bpm: number;
  waveform_peaks: number[];
}

export interface RustSeparationStemsPeaks {
  lead_vocal: number[];
  backing_vocal: number[];
  instrumental: number[];
  reverb_tail: number[];
  execution_backend: string;
}

export interface RustSeparatedStemsWavB64 {
  lead_vocal_b64: string;
  backing_vocal_b64: string;
  instrumental_b64: string;
  reverb_tail_b64: string;
}

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes.buffer;
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

export async function pickAudioFileViaRust(): Promise<string | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<string | null>('ipc_pick_audio_file');
}

export async function pickOnnxFileViaRust(): Promise<string | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<string | null>('ipc_pick_onnx_file');
}

export async function importLocalOnnxViaRust(
  sourcePath: string,
  targetModelId?: string
): Promise<RustCachedModelStatus | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<RustCachedModelStatus>('ipc_import_local_onnx', {
    sourcePath,
    targetModelId: targetModelId ?? null,
  });
}

export async function loadTrackViaRustBackend(
  filePath: string
): Promise<RustLoadedTrackSummary | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<RustLoadedTrackSummary>('ipc_load_track', { filePath });
}

export async function loadAudioBytesViaRustBackend(
  fileName: string,
  arrayBuffer: ArrayBuffer
): Promise<RustLoadedTrackSummary | null> {
  if (!isTauriDesktopRuntime()) return null;
  return invokeRustCommand<RustLoadedTrackSummary>('ipc_load_audio_bytes_b64', {
    fileName,
    audioB64: arrayBufferToBase64(arrayBuffer),
  });
}

export async function readFileBytesViaRust(filePath: string): Promise<ArrayBuffer> {
  const b64 = await invokeRustCommand<string>('ipc_read_file_base64', { filePath });
  return base64ToArrayBuffer(b64);
}

export async function preparePreviewWavViaRust(): Promise<ArrayBuffer> {
  const b64 = await invokeRustCommand<string>('ipc_prepare_preview_wav_b64');
  return base64ToArrayBuffer(b64);
}

export async function runSeparationViaRustBackend(
  mode: 'FastMdx' | 'ProRoformer' | 'HtDemucs',
  useDirectml = true
): Promise<RustSeparationStemsPeaks> {
  return invokeRustCommand<RustSeparationStemsPeaks>('ipc_run_separation', {
    mode,
    useDirectml,
  });
}

export async function getSeparatedStemsWavViaRust(): Promise<RustSeparatedStemsWavB64> {
  return invokeRustCommand<RustSeparatedStemsWavB64>('ipc_export_separated_stems_b64');
}

export interface RustExportStemChannelParam {
  role: 'LeadVocal' | 'BackingVocal' | 'Instrumental' | 'ReverbTail';
  volume_percent: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  formant_locked: boolean;
}

export async function exportAudioViaRustBackend(params: {
  outputDir: string;
  baseName: string;
  target: 'MasterMix' | 'AllStems' | 'KaraokeBacking';
  format: 'Wav16Bit' | 'Wav24Bit' | 'Wav32Float' | 'Flac24BitLossless';
  stemParams: RustExportStemChannelParam[];
  limiterEnabled: boolean;
}): Promise<{ exported_paths: string[] }> {
  return invokeRustCommand<{ exported_paths: string[] }>('ipc_export_audio', {
    outputDir: params.outputDir,
    baseName: params.baseName,
    target: params.target,
    format: params.format,
    stemParams: params.stemParams,
    limiterEnabled: params.limiterEnabled,
  });
}
