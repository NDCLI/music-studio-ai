import { LocalAiModelSpec, PitchEngineId } from '../types/studio';

export const MUSICAL_KEYS = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
] as const;

/**
 * Computes the transposed musical key given an original key (e.g., "Am" or "C") and semitone shift.
 */
export function computeTransposedKey(originalKey: string, semitones: number): string {
  const isMinor = originalKey.endsWith('m');
  const rootNote = isMinor ? originalKey.slice(0, -1) : originalKey;
  const startIndex = MUSICAL_KEYS.indexOf(rootNote as (typeof MUSICAL_KEYS)[number]);
  if (startIndex === -1) return originalKey;

  const shiftedIndex = (((startIndex + semitones) % 12) + 12) % 12;
  return `${MUSICAL_KEYS[shiftedIndex]}${isMinor ? 'm' : ''}`;
}

export interface PitchEngineInfo {
  id: PitchEngineId;
  name: string;
  badge: string;
  license: string;
  description: string;
}

export const PITCH_ENGINES: PitchEngineInfo[] = [
  {
    id: 'signalsmith_r3',
    name: 'Signalsmith Stretch + RubberBand R3',
    badge: 'Real-time 0ms',
    license: 'MIT / GPLv2',
    description:
      'Bảo toàn Formant thanh quản bằng Cepstral Envelope & khóa điểm gõ trống (Transient Lock) thời gian thực.',
  },
  {
    id: 'world_vocoder',
    name: 'WORLD Vocoder (Harvest + CheapTrick)',
    badge: 'Vocal Chuyên Sâu',
    license: 'Modified BSD',
    description:
      'Tách riêng tần số cơ bản (F0), bao phổ thanh quản (SP) và hơi thở (AP). Giữ nguyên 100% âm sắc ca sĩ khi dịch ±6 nửa cung.',
  },
  {
    id: 'rmvpe_hifigan',
    name: 'AI RMVPE + NSF-HiFiGAN (ONNX)',
    badge: 'Studio Render',
    license: 'MIT Open-Source',
    description:
      'Trích xuất cao độ bằng mạng nơ-ron RMVPE và tái tạo sóng âm bằng Neural Source-Filter HiFi-GAN không méo pha.',
  },
];

export const LOCAL_AI_MODELS: LocalAiModelSpec[] = [
  {
    id: 'uvr_mdx_inst_hq4',
    fileName: 'UVR-MDX-NET-Inst_HQ_4.onnx',
    taskTitle: 'Tầng 1: Tách Nhạc Nền & Tổng Vocal (Chế độ Nhanh)',
    architecture: 'MDX-Net (Hybrid Time-Frequency U-Net)',
    sdrDb: '11.8 dB',
    sizeMb: 56,
    stftSpec: {
      nFft: 5120,
      hopLength: 1024,
      dimF: 2560,
      dimT: 256,
      compensate: 1.019,
    },
    recommendedDevice: 'CPU Đa Luồng / DirectML GPU',
    installed: true,
  },
  {
    id: 'mel_band_roformer_vocals',
    fileName: 'mel_band_roformer_kim_ft_unwa.onnx',
    taskTitle: 'Tầng 1: Tách Nhạc Nền & Tổng Vocal (Chế độ Siêu Nét)',
    architecture: 'Mel-Band RoFormer (Rotary Transformer)',
    sdrDb: '13.1 dB',
    sizeMb: 228,
    stftSpec: {
      nFft: 4096,
      hopLength: 1024,
      dimF: 2048,
      dimT: 352,
      compensate: 1.0,
    },
    recommendedDevice: 'DirectML / CUDA GPU (VRAM ≥ 4GB)',
    installed: true,
  },
  {
    id: 'uvr_mdx_kara_2',
    fileName: 'UVR_MDXNET_KARA_2.onnx',
    taskTitle: 'Tầng 2: Tách Vocal Chính (Lead) & Giọng Bè (Harmony)',
    architecture: 'MDX-Net Karaoke / Backing Extractor',
    sdrDb: '10.4 dB',
    sizeMb: 55,
    stftSpec: {
      nFft: 5120,
      hopLength: 1024,
      dimF: 2048,
      dimT: 256,
      compensate: 1.065,
    },
    recommendedDevice: 'CPU Đa Luồng / DirectML GPU',
    installed: true,
  },
  {
    id: 'mel_band_roformer_karaoke',
    fileName: 'mel_band_roformer_karaoke_aufr33_viperx.onnx',
    taskTitle: 'Tầng 2: Tách Vocal Chính & Giọng Bè (Chế độ Pro)',
    architecture: 'Mel-Band RoFormer Karaoke ViperX',
    sdrDb: '10.8 dB',
    sizeMb: 228,
    stftSpec: {
      nFft: 4096,
      hopLength: 1024,
      dimF: 2048,
      dimT: 352,
      compensate: 1.0,
    },
    recommendedDevice: 'DirectML / CUDA GPU',
    installed: false,
  },
  {
    id: 'reverb_hq_foxjoy',
    fileName: 'Reverb_HQ_By_FoxJoy.onnx',
    taskTitle: 'Tầng 3: Tách Vocal Khô (Dry) & Đuôi Vang (Reverb Tail)',
    architecture: 'MDX-Net De-Reverb HQ',
    sdrDb: '11.4 dB',
    sizeMb: 67,
    stftSpec: {
      nFft: 7680,
      hopLength: 1024,
      dimF: 3072,
      dimT: 256,
      compensate: 1.028,
    },
    recommendedDevice: 'CPU Đa Luồng / DirectML GPU',
    installed: true,
  },
  {
    id: 'rmvpe_nsf_hifigan',
    fileName: 'rmvpe.onnx + nsf_hifigan.onnx',
    taskTitle: 'Nâng Hạ Tone AI: Dò Cao Độ F0 & Tái Tạo Sóng Âm Vocal',
    architecture: 'Deep Pitch Estimator + Neural Source-Filter',
    sdrDb: 'Artifact-Free',
    sizeMb: 90,
    stftSpec: {
      nFft: 2048,
      hopLength: 512,
      dimF: 1024,
      dimT: 128,
      compensate: 1.0,
    },
    recommendedDevice: 'CPU / DirectML GPU',
    installed: true,
  },
];
