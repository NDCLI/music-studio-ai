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
