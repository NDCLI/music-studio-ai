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
