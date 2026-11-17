export type StemId = 'lead_vocal' | 'backing_vocal' | 'instrumental' | 'reverb_tail';

export type PitchEngineId = 'signalsmith_r3' | 'world_vocoder' | 'rmvpe_hifigan';

export type SeparationPresetId =
  | 'full_mix'
  | 'karaoke_with_backing'
  | 'instrumental_only'
  | 'practice_backing'
  | 'acapella';

export type SeparationQualityMode = 'fast_mdx' | 'pro_roformer';

export type TrackSourceType =
  | 'studio_demo'
  | 'audio_file'
  | 'video_file'
  | 'youtube_single';

export interface StemTrackState {
  id: StemId;
  name: string;
  subtitle: string;
  modelSource: string;
  colorHex: string;
  badgeClass: string;
  volume: number; // 0 to 150 (%)
  pan: number; // -50 (L) to +50 (R)
  muted: boolean;
  solo: boolean;
  formantLocked: boolean;
  peaks: number[];
}

export interface StereoStemBuffer {
  sampleRate: number;
  left: Float32Array;
  right: Float32Array;
}

export type SeparatedStemsMap = Record<StemId, StereoStemBuffer>;

export interface PitchSpeedSettings {
  semitones: number; // -12 to +12
  cents: number; // -50 to +50
  speed: number; // 0.25 to 2.00
  formantPreservation: boolean;
  stemAwareProcessing: boolean;
  transientCrispLock: boolean;
  pitchEngine: PitchEngineId;
}

export interface SongMetadata {
  title: string;
  artist: string;
  fileName: string;
  sampleRate: number;
  channels: number;
  durationSec: number;
  originalBpm: number;
  originalKey: string; // e.g. 'Am'
  isCustomFile: boolean;
  sourceType: TrackSourceType;
  youtubeInfo?: {
    videoId: string;
    canonicalUrl: string;
    wasInPlaylist: boolean;
    ignoredPlaylistId?: string;
    playlistIndex?: number;
    thumbnailUrl?: string;
  };
}

export interface LocalAiModelSpec {
  id: string;
  fileName: string;
  taskTitle: string;
  architecture: string;
  sdrDb: string;
  sizeMb: number;
  downloadUrl: string;
  fallbackUrls?: string[];
  sourceRepo: string;
  stftSpec: {
    nFft: number;
    hopLength: number;
    dimF: number;
    dimT: number;
    compensate: number;
  };
  recommendedDevice: string;
  installed: boolean;
  downloading?: boolean;
  paused?: boolean;
  downloadProgress?: number;
  downloadedBytes?: number;
  totalBytes?: number;
  sha256Hex?: string;
  cachedSizeBytes?: number;
  cachedSource?: 'huggingface' | 'local_import';
  downloadError?: string;
}

export interface SeparationStageStatus {
  isRunning: boolean;
  isSeparated: boolean;
  currentStage: 0 | 1 | 2 | 3;
  progressPercent: number;
  stageMessage: string;
  executionBackend?: string;
}
