import React, { useEffect, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, FolderOpen, Upload, X } from 'lucide-react';
import { StudioHeader } from './components/StudioHeader';
import { StudioToolbar } from './components/StudioToolbar';
import { TransportBar } from './components/TransportBar';
import { PitchSpeedDeck } from './components/PitchSpeedDeck';
import { StemMixerTimeline } from './components/StemMixerTimeline';
import { WorkspaceTabs, type StudioWorkspaceTab } from './components/WorkspaceTabs';
import { ModelManagerModal } from './components/ModelManagerModal';
import { UpdateModal } from './components/UpdateModal';
import { ExportModal, ExportFormat, ExportOutcome, ExportTarget } from './components/ExportModal';
import { YouTubeImportModal } from './components/YouTubeImportModal';
import { LOCAL_AI_MODELS } from './data/modelsCatalog';
import { StudioPreviewAudioEngine } from './services/audioSynthesizer';
import {
  deleteCachedModelFromDb,
  downloadModelFromHuggingFace,
  importLocalOnnxModel,
  listCachedModelsFromDb,
} from './services/modelStorage';
import {
  clearBrowserExportDirHandle,
  ensureDirectoryWritePermission,
  isEmbeddedFrame,
  loadBrowserExportDirHandle,
  pickBrowserDirectoryHandle,
  pickSaveFileHandle,
  supportsSaveFilePicker,
  type BrowserDirectoryHandle,
  type BrowserFileHandle,
} from './services/exportStorage';
import { separateAudioBuffer3Stage } from './services/stemSeparator';
import {
  base64ToArrayBuffer,
  deleteModelViaRustBackend,
  downloadModelViaRustBackend,
  exportAudioViaRustBackend,
  getDefaultExportDirViaRust,
  getSeparatedStemsWavViaRust,
  importLocalOnnxViaRust,
  isTauriDesktopRuntime,
  listModelsViaRustBackend,
  loadAudioBytesViaRustBackend,
  loadTrackViaRustBackend,
  openLocalFolderViaRust,
  pickAudioFileViaRust,
  pickExportFolderViaRust,
  pickOnnxFileViaRust,
  preparePreviewWavViaRust,
  readFileBytesViaRust,
  runSeparationViaRustBackend,
  saveExportedFileViaRust,
  syncPitchSpeedToRustBackend,
} from './services/tauriBridge';
import {
  checkForAppUpdates,
  type AppUpdateCheckResult,
} from './services/updateChecker';
import {
  ComputeDeviceMode,
  LocalAiModelSpec,
  PitchSpeedSettings,
  SeparationPresetId,
  SeparationQualityMode,
  SeparationStageStatus,
  SongMetadata,
  StemId,
  StemTrackState,
  TrackSourceType,
} from './types/studio';

const EXPORT_DIR_STORAGE_KEY = 'music_studio_export_dir';
const EXPORT_MODE_STORAGE_KEY = 'music_studio_browser_export_mode';
const ACTIVE_TAB_STORAGE_KEY = 'music_studio_active_workspace_tab';
const COMPUTE_DEVICE_STORAGE_KEY = 'music_studio_compute_device';

const VIDEO_EXTENSIONS = ['mp4', 'mkv', 'webm', 'mov', 'avi', 'm4v'];

function detectFileSourceType(file: File): TrackSourceType {
  if (file.type.startsWith('video/')) return 'video_file';
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  if (VIDEO_EXTENSIONS.includes(ext)) return 'video_file';
  return 'audio_file';
}

const INITIAL_STEMS: StemTrackState[] = [
  {
    id: 'lead_vocal',
    name: 'Vocal Chính (Dry Lead Vocal)',
    subtitle: 'Giọng hát chính đã tách bè & khử đuôi vang',
    modelSource: '',
    colorHex: '#EC4899',
    badgeClass: 'text-pink-400',
    volume: 100,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: true,
    peaks: [],
  },
  {
    id: 'backing_vocal',
    name: 'Giọng Bè (Backing Vocal / Harmony)',
    subtitle: 'Bè phối âm, hợp xướng & điệp khúc phụ họa',
    modelSource: '',
    colorHex: '#F59E0B',
    badgeClass: 'text-amber-400',
    volume: 90,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: true,
    peaks: [],
  },
  {
    id: 'instrumental',
    name: 'Nhạc Nền (Instrumental Beat)',
    subtitle: 'Trống, Bass, Guitar, Piano & Synth giữ nguyên độ nảy',
    modelSource: '',
    colorHex: '#10B981',
    badgeClass: 'text-emerald-400',
    volume: 100,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: false,
    peaks: [],
  },
  {
    id: 'reverb_tail',
    name: 'Đuôi Vang (Vocal Reverb / Echo Tail)',
    subtitle: 'Không gian vang phòng thu tách rời để nâng tone không méo pha',
    modelSource: '',
    colorHex: '#06B6D4',
    badgeClass: 'text-cyan-400',
    volume: 55,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: false,
    peaks: [],
  },
];

export const App: React.FC = () => {
  const [song, setSong] = useState<SongMetadata | null>(null);

  const [pitchSpeed, setPitchSpeed] = useState<PitchSpeedSettings>({
    semitones: 0,
    cents: 0,
    speed: 1.0,
    formantPreservation: true,
    stemAwareProcessing: true,
    transientCrispLock: true,
    pitchEngine: 'signalsmith_r3',
  });

  const [stems, setStems] = useState<StemTrackState[]>(INITIAL_STEMS);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [peakLevelDb, setPeakLevelDb] = useState(-60);
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [loopStartSec] = useState(30);
  const [loopEndSec] = useState(75);
  const [masterVolume, setMasterVolume] = useState(100);
  const [limiterEnabled, setLimiterEnabled] = useState(true);
  const [isLoadingTrack, setIsLoadingTrack] = useState(false);
  const [trackLoadError, setTrackLoadError] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [separationToast, setSeparationToast] = useState<string | null>(null);
  const [separationBackendType, setSeparationBackendType] = useState<
    'none' | 'stft_dsp' | 'onnx_rust'
  >('none');

  const [qualityMode, setQualityMode] =
    useState<SeparationQualityMode>('fast_mdx');
  const [activePreset, setActivePreset] =
    useState<SeparationPresetId>('full_mix');
  const [separationStatus, setSeparationStatus] =
    useState<SeparationStageStatus>({
      isRunning: false,
      isSeparated: false,
      currentStage: 0,
      progressPercent: 0,
      stageMessage: 'Hãy mở file Nhạc hoặc Video từ máy để xử lý thực tế.',
    });

  const [models, setModels] = useState<LocalAiModelSpec[]>(LOCAL_AI_MODELS);
  const [modelsDir, setModelsDir] = useState<string | null>(null);
  const [computeDevice, setComputeDevice] = useState<ComputeDeviceMode>(() => {
    try {
      const saved = localStorage.getItem(COMPUTE_DEVICE_STORAGE_KEY);
      if (saved === 'auto' || saved === 'gpu' || saved === 'cpu') return saved;
    } catch {
      // Ignore storage read errors
    }
    return 'auto';
  });

  const handleChangeComputeDevice = (device: ComputeDeviceMode) => {
    setComputeDevice(device);
    try {
      localStorage.setItem(COMPUTE_DEVICE_STORAGE_KEY, device);
    } catch {
      // Ignore storage write errors
    }
  };

  const [isYouTubeModalOpen, setIsYouTubeModalOpen] = useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
  const [updateCheckResult, setUpdateCheckResult] = useState<AppUpdateCheckResult | null>(null);
  const [updateCheckError, setUpdateCheckError] = useState<string | null>(null);

  const handleCheckForUpdates = async () => {
    setIsUpdateModalOpen(true);
    setIsCheckingUpdate(true);
    setUpdateCheckError(null);
    try {
      const result = await checkForAppUpdates();
      setUpdateCheckResult(result);
    } catch (err) {
      setUpdateCheckError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsCheckingUpdate(false);
    }
  };

  const handleOpenModelsFolder = () => {
    if (modelsDir && isTauriDesktopRuntime()) {
      void openLocalFolderViaRust(modelsDir).catch(() => {});
    }
  };
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState<StudioWorkspaceTab>(() => {
    try {
      const saved = localStorage.getItem(ACTIVE_TAB_STORAGE_KEY);
      if (saved === 'pitch_speed' || saved === 'separation') return saved;
    } catch {
      // Ignore storage read errors
    }
    return 'separation';
  });

  const handleChangeWorkspaceTab = (tab: StudioWorkspaceTab) => {
    setActiveWorkspaceTab(tab);
    try {
      localStorage.setItem(ACTIVE_TAB_STORAGE_KEY, tab);
    } catch {
      // Ignore storage write errors
    }
  };

  const [defaultExportDir, setDefaultExportDir] = useState<string | null>(null);
  const [customExportDir, setCustomExportDir] = useState<string | null>(() => {
    try {
      return localStorage.getItem(EXPORT_DIR_STORAGE_KEY) || null;
    } catch {
      return null;
    }
  });
  const [browserDirHandle, setBrowserDirHandle] =
    useState<BrowserDirectoryHandle | null>(null);
  const [browserExportMode, setBrowserExportMode] = useState<
    'downloads' | 'directory' | 'save_as'
  >(() => {
    try {
      const saved = localStorage.getItem(EXPORT_MODE_STORAGE_KEY);
      if (saved === 'save_as' && supportsSaveFilePicker()) return 'save_as';
    } catch {
      // Ignore storage access errors
    }
    return 'downloads';
  });

  const audioEngineRef = useRef<StudioPreviewAudioEngine | null>(null);
  const dropzoneFileInputRef = useRef<HTMLInputElement | null>(null);
  const separationToastTimerRef = useRef<number | null>(null);
  const downloadAbortControllersRef = useRef<Record<string, AbortController>>({});

  const showSeparationSuccessToast = (text: string) => {
    setSeparationToast(text);
    if (separationToastTimerRef.current !== null) {
      window.clearTimeout(separationToastTimerRef.current);
    }
    separationToastTimerRef.current = window.setTimeout(() => {
      setSeparationToast(null);
      separationToastTimerRef.current = null;
    }, 6000);
  };
  const stemsRef = useRef(stems);
  stemsRef.current = stems;
  const pitchSpeedRef = useRef(pitchSpeed);
  pitchSpeedRef.current = pitchSpeed;

  useEffect(() => {
    audioEngineRef.current = new StudioPreviewAudioEngine();

    if (isTauriDesktopRuntime()) {
      void getDefaultExportDirViaRust()
        .then((dir) => {
          if (dir) setDefaultExportDir(dir);
        })
        .catch(() => {});
    } else {
      void loadBrowserExportDirHandle().then((restoredHandle) => {
        if (restoredHandle) {
          setBrowserDirHandle(restoredHandle);
          setBrowserExportMode((prev) => (prev === 'save_as' ? 'save_as' : 'directory'));
        }
      });
    }

    void Promise.all([
      listCachedModelsFromDb().catch(() => ({} as Record<string, { sizeBytes: number; sha256Hex: string; source: 'huggingface' | 'local_import' }>)),
      listModelsViaRustBackend().catch(() => null),
    ]).then(([dbMap, rustOverview]) => {
      if (rustOverview?.models_dir) {
        setModelsDir(rustOverview.models_dir);
      }
      const rustMap = new Map(
        (rustOverview?.cached_status ?? []).map((item) => [item.id, item])
      );
      setModels((prev) =>
        prev.map((m) => {
          const rustHit = rustMap.get(m.id);
          if (rustHit?.installed) {
            return {
              ...m,
              installed: true,
              cachedSizeBytes: rustHit.size_bytes,
              sha256Hex: rustHit.sha256_hex ?? m.sha256Hex,
              cachedSource: 'huggingface',
            };
          }
          const hit = dbMap[m.id];
          if (!hit) return m;
          return {
            ...m,
            installed: true,
            cachedSizeBytes: hit.sizeBytes,
            sha256Hex: hit.sha256Hex,
            cachedSource: hit.source,
          };
        })
      );
    });

    return () => {
      audioEngineRef.current?.stop();
    };
  }, []);

  const handleTogglePlay = () => {
    const engine = audioEngineRef.current;
    if (!engine || !song) return;

    const totalDuration = song.durationSec;

    if (isPlaying) {
      engine.stop();
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      engine.start(
        () => stemsRef.current,
        () => pitchSpeedRef.current,
        (deltaSec, livePeakDb) => {
          setPeakLevelDb(livePeakDb);
          setCurrentTimeSec((prev) => {
            const next = prev + deltaSec;
            if (loopEnabled && next >= loopEndSec) {
              engine.seekToSeconds(loopStartSec);
              return loopStartSec;
            }
            if (totalDuration > 0 && next >= totalDuration) {
              engine.seekToSeconds(0);
              return 0;
            }
            return next;
          });
        }
      );
    }
  };

  const handleStop = () => {
    audioEngineRef.current?.stop();
    setIsPlaying(false);
    setCurrentTimeSec(0);
    audioEngineRef.current?.seekToSeconds(0);
  };

  const handleSeek = (timeSec: number) => {
    setCurrentTimeSec(timeSec);
    audioEngineRef.current?.seekToSeconds(timeSec);
  };

  const handleSelectAudioFile = async (file: File) => {
    const engine = audioEngineRef.current;
    if (!engine) return;

    if (isPlaying) {
      engine.stop();
      setIsPlaying(false);
    }

    setIsLoadingTrack(true);
    setTrackLoadError(null);
    const sourceType = detectFileSourceType(file);

    try {
      const rawArrayBuffer = await file.arrayBuffer();
      const decoded = await engine.loadAudioFromArrayBuffer(rawArrayBuffer);
      const cleanTitle = file.name.replace(/\.[^/.]+$/, '');

      if (isTauriDesktopRuntime() && rawArrayBuffer.byteLength <= 80 * 1024 * 1024) {
        void loadAudioBytesViaRustBackend(file.name, rawArrayBuffer).catch(() => null);
      }

      setSong({
        title: cleanTitle,
        artist: sourceType === 'video_file' ? 'Video Audio Track' : 'Local Audio Track',
        fileName: file.name,
        sampleRate: decoded.sampleRate,
        channels: decoded.channels,
        durationSec: Math.max(1, Math.round(decoded.durationSec)),
        originalBpm: decoded.detectedBpm,
        originalKey: decoded.detectedKey,
        isCustomFile: true,
        sourceType,
      });

      setStems((prev) =>
        prev.map((s) => ({
          ...s,
          muted: false,
          solo: false,
          volume: s.id === 'reverb_tail' ? 55 : 100,
          peaks: [],
        }))
      );
      setActivePreset('full_mix');
      setSeparationToast(null);
      setSeparationBackendType('none');
      setSeparationStatus({
        isRunning: false,
        isSeparated: false,
        currentStage: 0,
        progressPercent: 0,
        stageMessage: `Chưa tách nhạc — Bấm "Tách Nhạc AI (3 Tầng)" để tách "${file.name}" và hiển thị sóng âm 4 kênh.`,
      });
      setCurrentTimeSec(0);
    } catch (err) {
      setTrackLoadError(
        `Không thể giải mã "${file.name}": ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    } finally {
      setIsLoadingTrack(false);
    }
  };

  const handleDesktopOpenTrack = async () => {
    const engine = audioEngineRef.current;
    if (!engine) return;

    const pickedPath = await pickAudioFileViaRust().catch(() => null);
    if (!pickedPath) return;

    if (isPlaying) {
      engine.stop();
      setIsPlaying(false);
    }

    setIsLoadingTrack(true);
    setTrackLoadError(null);

    try {
      const rustSummary = await loadTrackViaRustBackend(pickedPath);
      if (!rustSummary) {
        throw new Error('Không đọc được file qua bộ giải mã cục bộ.');
      }

      try {
        const rawBytes = await readFileBytesViaRust(pickedPath);
        await engine.loadAudioFromArrayBuffer(rawBytes);
      } catch {
        const previewWavBytes = await preparePreviewWavViaRust();
        await engine.loadAudioFromArrayBuffer(previewWavBytes);
      }

      const cleanTitle = rustSummary.file_name.replace(/\.[^/.]+$/, '');
      setSong({
        title: cleanTitle,
        artist: rustSummary.is_video_source ? 'Video Audio Track' : 'Local Audio Track',
        fileName: rustSummary.file_name,
        sampleRate: rustSummary.sample_rate,
        channels: rustSummary.channels,
        durationSec: Math.max(1, Math.round(rustSummary.duration_sec)),
        originalBpm: Math.round(rustSummary.detected_bpm),
        originalKey: rustSummary.detected_key,
        isCustomFile: true,
        sourceType: rustSummary.is_video_source ? 'video_file' : 'audio_file',
      });

      setStems((prev) =>
        prev.map((s) => ({
          ...s,
          muted: false,
          solo: false,
          volume: s.id === 'reverb_tail' ? 55 : 100,
          peaks: [],
        }))
      );
      setActivePreset('full_mix');
      setSeparationToast(null);
      setSeparationBackendType('none');
      setSeparationStatus({
        isRunning: false,
        isSeparated: false,
        currentStage: 0,
        progressPercent: 0,
        stageMessage: `Chưa tách nhạc — Bấm "Tách Nhạc AI (3 Tầng)" để tách "${rustSummary.file_name}" và hiển thị sóng âm 4 kênh.`,
      });
      setCurrentTimeSec(0);
    } catch (err) {
      setTrackLoadError(
        `Không thể nạp file "${pickedPath}": ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    } finally {
      setIsLoadingTrack(false);
    }
  };

  const activeDesktopExportDir = customExportDir ?? defaultExportDir;
  const isDesktopRuntime = isTauriDesktopRuntime();
  const browserSaveAsCapable = !isDesktopRuntime && supportsSaveFilePicker();
  const embeddedFrame = !isDesktopRuntime && isEmbeddedFrame();

  const exportFolderLabel = isDesktopRuntime
    ? activeDesktopExportDir ?? '%USERPROFILE%\\Music\\Music Studio AI'
    : browserExportMode === 'save_as'
      ? 'Mở hộp thoại Save As để chọn nơi lưu mỗi lần xuất'
      : browserDirHandle
        ? browserDirHandle.name
        : 'Thư mục Downloads (Mặc định của trình duyệt)';
  const isCustomExportFolder = isDesktopRuntime
    ? Boolean(customExportDir)
    : browserExportMode === 'save_as' || Boolean(browserDirHandle);
  const canOpenExportFolder = isDesktopRuntime && Boolean(activeDesktopExportDir);

  const handlePickExportFolder = async (): Promise<string | null> => {
    if (isTauriDesktopRuntime()) {
      const picked = await pickExportFolderViaRust();
      if (picked) {
        setCustomExportDir(picked);
        try {
          localStorage.setItem(EXPORT_DIR_STORAGE_KEY, picked);
        } catch {
          // Ignore storage quota errors
        }
      }
      return null;
    }

    try {
      const handle = await pickBrowserDirectoryHandle();
      setBrowserDirHandle(handle);
      setBrowserExportMode('directory');
      try {
        localStorage.setItem(EXPORT_MODE_STORAGE_KEY, 'directory');
      } catch {
        // Ignore storage errors
      }
      return null;
    } catch (err) {
      if (err instanceof Error && err.message === 'UNSUPPORTED_DIRECTORY_PICKER') {
        if (embeddedFrame) {
          return 'Trang đang mở trong khung xem trước nên trình duyệt chặn chọn thư mục trên ổ đĩa (file xuất vẫn vào Downloads). Hãy bấm "Mở Tab Riêng" ở bên phải hoặc bật app Desktop.';
        }
        if (supportsSaveFilePicker()) {
          return 'Trình duyệt không hỗ trợ chọn thư mục cố định. Hãy bấm "Lưu Qua Save As" bên cạnh để tự chọn thư mục khi xuất!';
        }
        return 'Trình duyệt hiện tại không hỗ trợ chọn thư mục trực tiếp (chỉ dùng được trên Chrome/Edge); file sẽ tải về Downloads.';
      }
      if (err instanceof Error && err.message === 'PERMISSION_DENIED') {
        return 'Bạn chưa cấp quyền ghi file vào thư mục vừa chọn, nên chưa thể lưu vào đó.';
      }
      if (err instanceof DOMException && err.name === 'SecurityError') {
        return embeddedFrame
          ? 'Khung xem trước (iframe) chặn quyền chọn thư mục trên ổ đĩa. Hãy bấm "Mở Tab Riêng" để mở trực tiếp trên Chrome/Edge.'
          : 'Trình duyệt chặn chọn thư mục ở chế độ hiện tại. Hãy thử nút "Lưu Qua Save As".';
      }
      if (
        (err instanceof DOMException && err.name === 'InvalidStateError') ||
        (err instanceof Error && err.message.includes('already active'))
      ) {
        return 'Trình duyệt đang có một hộp thoại chọn tệp/thư mục mở ẩn ở nền (kiểm tra Taskbar / Alt+Tab để đóng nó hoặc bấm F5 tải lại trang), sau đó ưu tiên chọn "Lưu Qua Save As" nếu muốn lưu ra Desktop.';
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        if (!browserDirHandle && browserExportMode !== 'save_as') {
          return 'Chưa chọn được thư mục mới. Lưu ý: Chrome/Edge không cho chọn trực tiếp Desktop/Documents bằng nút Chọn Thư Mục — hãy tạo 1 thư mục con riêng (vd: D:\\NhacXuat) hoặc bấm "Lưu Qua Save As".';
        }
        return null;
      }
      return err instanceof Error ? err.message : 'Không thể truy cập thư mục đã chọn.';
    }
  };

  const handleEnableSaveAsMode = () => {
    setBrowserExportMode('save_as');
    try {
      localStorage.setItem(EXPORT_MODE_STORAGE_KEY, 'save_as');
    } catch {
      // Ignore storage errors
    }
  };

  const handleResetExportFolder = () => {
    if (isTauriDesktopRuntime()) {
      setCustomExportDir(null);
      try {
        localStorage.removeItem(EXPORT_DIR_STORAGE_KEY);
      } catch {
        // Ignore storage errors
      }
      return;
    }
    setBrowserDirHandle(null);
    setBrowserExportMode('downloads');
    void clearBrowserExportDirHandle();
    try {
      localStorage.removeItem(EXPORT_MODE_STORAGE_KEY);
    } catch {
      // Ignore storage errors
    }
  };

  const handleOpenExportFolder = () => {
    if (isTauriDesktopRuntime() && activeDesktopExportDir) {
      void openLocalFolderViaRust(activeDesktopExportDir);
    }
  };

  const handleTriggerExport = async (
    baseName: string,
    target: ExportTarget,
    format: ExportFormat
  ): Promise<ExportOutcome> => {
    const engine = audioEngineRef.current;
    if (!engine || !song) {
      return {
        files: [],
        destinationLabel: exportFolderLabel,
        savedToLocalFolder: false,
        errorMessage: 'Vui lòng mở file Nhạc hoặc Video trước khi xuất.',
      };
    }

    if (
      isTauriDesktopRuntime() &&
      activeDesktopExportDir &&
      separationBackendType !== 'stft_dsp'
    ) {
      const formatMap: Record<
        ExportFormat,
        'Wav16Bit' | 'Wav24Bit' | 'Wav32Float' | 'Flac24BitLossless'
      > = {
        wav16: 'Wav16Bit',
        wav24: 'Wav24Bit',
        wav32f: 'Wav32Float',
        flac: 'Flac24BitLossless',
      };
      const targetMap: Record<
        ExportTarget,
        'MasterMix' | 'AllStems' | 'KaraokeBacking'
      > = {
        master_mix: 'MasterMix',
        all_stems: 'AllStems',
        karaoke_backing: 'KaraokeBacking',
      };
      const roleMap: Record<
        StemId,
        'LeadVocal' | 'BackingVocal' | 'Instrumental' | 'ReverbTail'
      > = {
        lead_vocal: 'LeadVocal',
        backing_vocal: 'BackingVocal',
        instrumental: 'Instrumental',
        reverb_tail: 'ReverbTail',
      };

      const res = await exportAudioViaRustBackend({
        outputDir: activeDesktopExportDir,
        baseName,
        target: targetMap[target],
        format: formatMap[format],
        stemParams: stems.map((s) => ({
          role: roleMap[s.id],
          volume_percent: s.volume,
          pan: s.pan / 50,
          muted: s.muted,
          solo: s.solo,
          formant_locked: s.formantLocked,
        })),
        limiterEnabled,
      });

      return {
        files: res.exported_paths.map((p) => p.split(/[\\/]/).pop() ?? p),
        destinationLabel: activeDesktopExportDir,
        savedToLocalFolder: true,
      };
    }

    // Acquire browser directory permission or single-file Save As handle immediately while user activation is active
    let prePickedSaveHandle: BrowserFileHandle | null = null;
    let selectedDirHandle: BrowserDirectoryHandle | null = browserDirHandle;
    if (!isDesktopRuntime && browserExportMode === 'save_as') {
      try {
        if (target === 'all_stems') {
          if (!selectedDirHandle) {
            selectedDirHandle = await pickBrowserDirectoryHandle();
            setBrowserDirHandle(selectedDirHandle);
          }
          const ok = await ensureDirectoryWritePermission(selectedDirHandle, true);
          if (!ok) {
            throw new Error('Trình duyệt từ chối quyền ghi vào thư mục đã chọn.');
          }
        } else {
          const ext = format === 'flac' ? 'flac' : 'wav';
          const expectedName =
            target === 'karaoke_backing'
              ? `${baseName}_karaoke_with_backing.${ext}`
              : `${baseName}.${ext}`;
          prePickedSaveHandle = await pickSaveFileHandle(expectedName, format);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return {
            files: [],
            destinationLabel: exportFolderLabel,
            savedToLocalFolder: false,
            errorMessage: 'Đã hủy hộp thoại chọn nơi lưu — chưa ghi file.',
          };
        }
        if (
          (err instanceof DOMException && err.name === 'InvalidStateError') ||
          (err instanceof Error && err.message.includes('already active'))
        ) {
          return {
            files: [],
            destinationLabel: exportFolderLabel,
            savedToLocalFolder: false,
            errorMessage:
              'Hộp thoại chọn thư mục trước đó vẫn đang mở ẩn trong trình duyệt (`File picker already active`). Hãy nhấn Alt+Tab đóng hộp thoại cũ hoặc nhấn F5 tải lại trang, rồi bấm trực tiếp "Lưu Qua Save As".',
          };
        }
        throw err;
      }
    } else if (!isDesktopRuntime && browserExportMode === 'directory') {
      if (!selectedDirHandle) {
        throw new Error('Không tìm thấy quyền thư mục đã chọn. Vui lòng bấm "Chọn Thư Mục..." lại.');
      }
      const permitted = await ensureDirectoryWritePermission(selectedDirHandle, true);
      if (!permitted) {
        throw new Error(
          `Chưa được cấp quyền ghi vào thư mục "${selectedDirHandle.name}". Hãy bấm "Chọn Thư Mục..." lại để cấp quyền.`
        );
      }
    }

    const outputs = engine.renderExportAudioFiles(
      baseName,
      stems,
      pitchSpeed,
      target,
      format
    );

    if (isTauriDesktopRuntime() && activeDesktopExportDir) {
      const savedNames: string[] = [];
      for (const item of outputs) {
        const bytes = new Uint8Array(await item.blob.arrayBuffer());
        await saveExportedFileViaRust(activeDesktopExportDir, item.fileName, bytes);
        savedNames.push(item.fileName);
      }
      return {
        files: savedNames,
        destinationLabel: activeDesktopExportDir,
        savedToLocalFolder: true,
      };
    }

    if (prePickedSaveHandle && outputs.length === 1) {
      const writable = await prePickedSaveHandle.createWritable();
      await writable.write(outputs[0].blob);
      await writable.close();
      return {
        files: [prePickedSaveHandle.name],
        destinationLabel: 'vị trí Save As đã chọn',
        savedToLocalFolder: true,
      };
    }

    if (selectedDirHandle) {
      const savedNames: string[] = [];
      for (const item of outputs) {
        const fileHandle = await selectedDirHandle.getFileHandle(item.fileName, {
          create: true,
        });
        const writable = await fileHandle.createWritable();
        await writable.write(item.blob);
        await writable.close();
        savedNames.push(item.fileName);
      }
      return {
        files: savedNames,
        destinationLabel: `thư mục "${selectedDirHandle.name}"`,
        savedToLocalFolder: true,
      };
    }

    const names: string[] = [];
    for (const item of outputs) {
      const url = URL.createObjectURL(item.blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = item.fileName;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      names.push(item.fileName);
    }

    return {
      files: names,
      destinationLabel: 'thư mục Downloads mặc định',
      savedToLocalFolder: false,
    };
  };

  const handleUpdatePitchSpeed = (partial: Partial<PitchSpeedSettings>) => {
    setPitchSpeed((prev) => {
      const next = { ...prev, ...partial };
      void syncPitchSpeedToRustBackend(next);
      return next;
    });
  };

  const handleResetPitchSpeed = () => {
    setPitchSpeed((prev) => {
      const next = {
        ...prev,
        semitones: 0,
        cents: 0,
        speed: 1.0,
      };
      void syncPitchSpeedToRustBackend(next);
      return next;
    });
  };

  const runSpectralSeparation = async () => {
    const engine = audioEngineRef.current;
    const customBuffer = engine?.getCustomAudioBuffer();
    if (!customBuffer || !engine) return;

    const startedAt = performance.now();
    setSeparationToast(null);
    setSeparationStatus({
      isRunning: true,
      isSeparated: false,
      currentStage: 1,
      progressPercent: 1,
      stageMessage: 'Đang phân tích tín hiệu & tách phổ STFT 3 tầng trên file thực tế...',
    });

    try {
      const result = await separateAudioBuffer3Stage(
        customBuffer,
        qualityMode,
        (info) => {
          setSeparationStatus({
            isRunning: true,
            isSeparated: false,
            currentStage: info.stage,
            progressPercent: info.percent,
            stageMessage: info.message,
          });
        }
      );

      const elapsedSec = ((performance.now() - startedAt) / 1000).toFixed(1);
      engine.setSeparatedStems(result.stems);
      setStems((prev) =>
        prev.map((s) => ({
          ...s,
          modelSource: 'Tách Phổ STFT (DSP)',
          peaks: result.peaks[s.id] ?? s.peaks,
        }))
      );
      setSeparationBackendType('stft_dsp');
      const doneMessage = `Đã tách xong 4 kênh độc lập trong ${elapsedSec}s (Vocal chính, Giọng bè, Nhạc nền, Đuôi vang).`;
      setSeparationStatus({
        isRunning: false,
        isSeparated: true,
        currentStage: 3,
        progressPercent: 100,
        stageMessage: doneMessage,
        executionBackend: 'Tách Phổ STFT (DSP)',
      });
      showSeparationSuccessToast(`Tách nhạc thành công (${elapsedSec}s)! Sóng âm 4 kênh đã hiển thị.`);
    } catch (err) {
      setSeparationStatus({
        isRunning: false,
        isSeparated: false,
        currentStage: 0,
        progressPercent: 0,
        stageMessage: `Lỗi khi tách phổ: ${
          err instanceof Error ? err.message : String(err)
        }`,
      });
    }
  };

  const handleRunSeparation = async () => {
    if (separationStatus.isRunning) return;
    const engine = audioEngineRef.current;
    const customBuffer = engine?.getCustomAudioBuffer();
    if (!customBuffer || !engine || !song) return;

    if (isTauriDesktopRuntime()) {
      const startedAt = performance.now();
      const useGpu = computeDevice !== 'cpu';
      const deviceLabel =
        computeDevice === 'cpu'
          ? 'CPU đa luồng'
          : computeDevice === 'gpu'
            ? 'GPU (DirectML)'
            : 'AUTO (Ưu tiên GPU)';
      setSeparationToast(null);
      setSeparationStatus({
        isRunning: true,
        isSeparated: false,
        currentStage: 1,
        progressPercent: 15,
        stageMessage: `Đang chạy suy luận 3 tầng AI ONNX (${deviceLabel})...`,
      });

      try {
        const rustMode = qualityMode === 'pro_roformer' ? 'ProRoformer' : 'FastMdx';
        const res = await runSeparationViaRustBackend(rustMode, useGpu);
        const wavRes = await getSeparatedStemsWavViaRust();

        const [lead, backing, inst, rev] = await Promise.all([
          engine.decodeWavArrayBufferToStereo(base64ToArrayBuffer(wavRes.lead_vocal_b64)),
          engine.decodeWavArrayBufferToStereo(base64ToArrayBuffer(wavRes.backing_vocal_b64)),
          engine.decodeWavArrayBufferToStereo(base64ToArrayBuffer(wavRes.instrumental_b64)),
          engine.decodeWavArrayBufferToStereo(base64ToArrayBuffer(wavRes.reverb_tail_b64)),
        ]);

        engine.setSeparatedStems({
          lead_vocal: lead,
          backing_vocal: backing,
          instrumental: inst,
          reverb_tail: rev,
        });

        const elapsedSec = ((performance.now() - startedAt) / 1000).toFixed(1);
        setStems((prev) =>
          prev.map((s) => ({
            ...s,
            peaks: res[s.id] ?? s.peaks,
          }))
        );
        setSeparationBackendType('onnx_rust');
        setSeparationStatus({
          isRunning: false,
          isSeparated: true,
          currentStage: 3,
          progressPercent: 100,
          stageMessage: `Đã tách xong 4 kênh bằng AI ONNX cục bộ trong ${elapsedSec}s.`,
          executionBackend: res.execution_backend,
        });
        showSeparationSuccessToast(`Tách nhạc AI ONNX thành công (${elapsedSec}s)!`);
        return;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes('ONNX_PREREQ_MISSING:')) {
          const cleanReason = msg.replace(/^.*ONNX_PREREQ_MISSING:\s*/, '');
          setSeparationStatus({
            isRunning: false,
            isSeparated: false,
            currentStage: 0,
            progressPercent: 0,
            stageMessage: 'Chưa thể chạy mô hình AI ONNX trên máy.',
            onnxPrereqMissing: cleanReason,
          });
          return;
        }
        setSeparationStatus({
          isRunning: false,
          isSeparated: false,
          currentStage: 0,
          progressPercent: 0,
          stageMessage: `Lỗi AI ONNX: ${msg}`,
        });
        return;
      }
    }

    await runSpectralSeparation();
  };

  const handleApplyPreset = (preset: SeparationPresetId) => {
    setActivePreset(preset);
    setStems((prev) =>
      prev.map((s) => {
        switch (preset) {
          case 'full_mix':
            return {
              ...s,
              muted: false,
              solo: false,
              volume: s.id === 'reverb_tail' ? 55 : 100,
            };
          case 'karaoke_with_backing':
            return {
              ...s,
              solo: false,
              muted: s.id === 'lead_vocal' || s.id === 'reverb_tail',
              volume: s.id === 'backing_vocal' ? 105 : 100,
            };
          case 'instrumental_only':
            return {
              ...s,
              solo: false,
              muted: s.id !== 'instrumental',
              volume: 100,
            };
          case 'practice_backing':
            return {
              ...s,
              solo: false,
              muted: s.id === 'lead_vocal' || s.id === 'reverb_tail',
              volume: s.id === 'backing_vocal' ? 130 : 35,
            };
          case 'acapella':
            return {
              ...s,
              solo: false,
              muted: s.id === 'instrumental',
              volume: 100,
            };
        }
      })
    );
  };

  const handleToggleMute = (id: StemId) => {
    setStems((prev) =>
      prev.map((s) => (s.id === id ? { ...s, muted: !s.muted } : s))
    );
  };

  const handleToggleSolo = (id: StemId) => {
    setStems((prev) =>
      prev.map((s) => (s.id === id ? { ...s, solo: !s.solo } : s))
    );
  };

  const handleChangeStemVolume = (id: StemId, volume: number) => {
    setStems((prev) =>
      prev.map((s) => (s.id === id ? { ...s, volume } : s))
    );
  };

  const handleDownloadModel = async (modelId: string) => {
    const spec = models.find((m) => m.id === modelId);
    if (!spec) return;

    const controller = new AbortController();
    downloadAbortControllersRef.current[modelId] = controller;

    setModels((prev) =>
      prev.map((m) =>
        m.id === modelId
          ? {
              ...m,
              downloading: true,
              paused: false,
              downloadProgress: 0,
              downloadedBytes: 0,
              downloadError: undefined,
            }
          : m
      )
    );

    try {
      if (isTauriDesktopRuntime() && spec.sizeMb > 200) {
        const rustSaved = await downloadModelViaRustBackend(modelId);
        if (rustSaved?.installed) {
          setModels((prev) =>
            prev.map((m) =>
              m.id === modelId
                ? {
                    ...m,
                    downloading: false,
                    installed: true,
                    downloadProgress: 100,
                    cachedSizeBytes: rustSaved.size_bytes,
                    sha256Hex: rustSaved.sha256_hex,
                    cachedSource: 'huggingface',
                    downloadError: undefined,
                  }
                : m
            )
          );
          return;
        }
      }

      const saved = await downloadModelFromHuggingFace(
        spec,
        controller.signal,
        (downloadedBytes, totalBytes, percent) => {
          setModels((prev) =>
            prev.map((m) =>
              m.id === modelId
                ? {
                    ...m,
                    downloadedBytes,
                    totalBytes,
                    downloadProgress: percent,
                  }
                : m
            )
          );
        }
      );

      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? {
                ...m,
                downloading: false,
                installed: true,
                downloadProgress: 100,
                cachedSizeBytes: saved.sizeBytes,
                sha256Hex: saved.sha256Hex,
                cachedSource: saved.source,
                downloadError: undefined,
              }
            : m
        )
      );
    } catch (err) {
      const isAborted = err instanceof DOMException && err.name === 'AbortError';
      if (!isAborted && isTauriDesktopRuntime()) {
        try {
          const rustSaved = await downloadModelViaRustBackend(modelId);
          if (rustSaved?.installed) {
            setModels((prev) =>
              prev.map((m) =>
                m.id === modelId
                  ? {
                      ...m,
                      downloading: false,
                      installed: true,
                      downloadProgress: 100,
                      cachedSizeBytes: rustSaved.size_bytes,
                      sha256Hex: rustSaved.sha256_hex,
                      cachedSource: 'huggingface',
                      downloadError: undefined,
                    }
                  : m
              )
            );
            return;
          }
        } catch {
          // Fall through to detailed error banner below
        }
      }

      const errMessage = isAborted
        ? 'Đã tạm dừng tải model. Bấm Tải HuggingFace để tải lại hoặc Nhập .ONNX Có Sẵn.'
        : `Không thể tải trực tiếp (${
            err instanceof Error ? err.message : String(err)
          }). Hãy bấm nút "Copy Link Tải" hoặc mở Repo trên trình duyệt rồi chọn "Nhập .ONNX Có Sẵn".`;

      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? {
                ...m,
                downloading: false,
                paused: isAborted,
                downloadError: errMessage,
              }
            : m
        )
      );
    } finally {
      delete downloadAbortControllersRef.current[modelId];
    }
  };

  const handlePauseDownload = (modelId: string) => {
    const controller = downloadAbortControllersRef.current[modelId];
    if (controller) {
      controller.abort();
    }
  };

  const handleImportLocalModel = async (modelId: string, file: File) => {
    const spec = models.find((m) => m.id === modelId);
    if (!spec) return;

    try {
      const saved = await importLocalOnnxModel(file, spec);
      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? {
                ...m,
                installed: true,
                downloading: false,
                downloadProgress: 100,
                cachedSizeBytes: saved.sizeBytes,
                sha256Hex: saved.sha256Hex,
                cachedSource: saved.source,
                downloadError: undefined,
              }
            : m
        )
      );
    } catch (err) {
      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? {
                ...m,
                downloadError: err instanceof Error ? err.message : String(err),
              }
            : m
        )
      );
    }
  };

  const handleImportDesktopOnnx = async (modelId: string) => {
    const pickedPath = await pickOnnxFileViaRust().catch(() => null);
    if (!pickedPath) return;

    try {
      const saved = await importLocalOnnxViaRust(pickedPath, modelId);
      if (!saved) return;
      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? {
                ...m,
                installed: true,
                downloading: false,
                downloadProgress: 100,
                cachedSizeBytes: saved.size_bytes,
                sha256Hex: saved.sha256_hex,
                cachedSource: 'local_import',
                downloadError: undefined,
              }
            : m
        )
      );
    } catch (err) {
      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? {
                ...m,
                downloadError: err instanceof Error ? err.message : String(err),
              }
            : m
        )
      );
    }
  };

  const handleDeleteCachedModel = async (modelId: string) => {
    const spec = models.find((m) => m.id === modelId);
    await deleteCachedModelFromDb(modelId);
    if (spec) {
      await deleteModelViaRustBackend(spec.fileName).catch(() => {});
    }
    setModels((prev) =>
      prev.map((m) =>
        m.id === modelId
          ? {
              ...m,
              installed: false,
              cachedSizeBytes: undefined,
              sha256Hex: undefined,
              cachedSource: undefined,
              downloadProgress: 0,
            }
          : m
      )
    );
  };

  const installedCount = models.filter((m) => m.installed).length;
  const hasLoadedTrack = Boolean(song);

  const handleOpenFileClick = () => {
    if (isTauriDesktopRuntime()) {
      void handleDesktopOpenTrack();
    } else {
      dropzoneFileInputRef.current?.click();
    }
  };

  const handleDropFile = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void handleSelectAudioFile(file);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!isDraggingFile) setIsDraggingFile(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setIsDraggingFile(false);
      }}
      onDrop={handleDropFile}
      className="min-h-screen bg-[#080B14] text-slate-100 flex flex-col"
    >
      <input
        ref={dropzoneFileInputRef}
        type="file"
        accept="audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.opus,.mp4,.mkv,.webm,.mov,.avi,.m4v"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleSelectAudioFile(file);
          e.target.value = '';
        }}
        className="hidden"
      />

      {/* 1. Studio Top Header */}
      <StudioHeader
        song={song}
        semitones={pitchSpeed.semitones}
        speed={pitchSpeed.speed}
        onSelectAudioFile={handleSelectAudioFile}
        onRequestOpenTrack={isTauriDesktopRuntime() ? () => void handleDesktopOpenTrack() : undefined}
        onOpenYouTubeModal={() => setIsYouTubeModalOpen(true)}
        onOpenExportModal={() => setIsExportModalOpen(true)}
      />

      {/* 1B. Studio Toolbar: Compute Device, Model & App Settings/Update */}
      <StudioToolbar
        computeDevice={computeDevice}
        onChangeComputeDevice={handleChangeComputeDevice}
        qualityMode={qualityMode}
        onChangeQualityMode={setQualityMode}
        models={models}
        installedModelCount={installedCount}
        isCheckingUpdate={isCheckingUpdate}
        updateAvailableVersion={
          updateCheckResult?.hasUpdate ? updateCheckResult.latestVersion : null
        }
        onOpenModelManager={() => setIsModelModalOpen(true)}
        onCheckForUpdates={() => void handleCheckForUpdates()}
      />

      {/* Real Track Loader Dropzone when empty or error occurred */}
      {(!hasLoadedTrack || trackLoadError || isLoadingTrack || isDraggingFile) && (
        <div
          className={`mx-4 mt-3 rounded-2xl border-2 border-dashed p-5 transition-colors ${
            isDraggingFile
              ? 'border-indigo-400 bg-indigo-500/15'
              : 'border-white/15 bg-[#0F1424]/80'
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3.5">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-300">
                <Upload className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-white">
                  {isLoadingTrack
                    ? 'Đang giải mã âm thanh từ file...'
                    : isDraggingFile
                      ? 'Thả file nhạc hoặc video vào đây để mở'
                      : 'Chưa mở file âm thanh — Chọn file thực tế từ máy để test'}
                </h2>
                <p className="mt-0.5 text-xs text-slate-400">
                  Hỗ trợ trực tiếp file nhạc (MP3, WAV, FLAC, M4A, OGG) và Video (MP4, MKV, WEBM, MOV).
                </p>
              </div>
            </div>

            <button
              type="button"
              disabled={isLoadingTrack}
              onClick={handleOpenFileClick}
              className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-bold text-white shadow-lg shadow-indigo-600/25 transition-colors hover:bg-indigo-500 disabled:opacity-60"
            >
              <FolderOpen className="h-4 w-4" />
              <span>{isLoadingTrack ? 'Đang Mở...' : 'Chọn File Nhạc / Video'}</span>
            </button>
          </div>

          {trackLoadError && (
            <div className="mt-3 flex items-start gap-2 rounded-xl border border-rose-500/40 bg-rose-500/10 px-3.5 py-2.5 text-xs text-rose-200">
              <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{trackLoadError}</span>
            </div>
          )}
        </div>
      )}

      {/* 2. Master Transport Bar */}
      <TransportBar
        disabled={!hasLoadedTrack}
        isPlaying={isPlaying}
        currentTimeSec={currentTimeSec}
        durationSec={song?.durationSec ?? 0}
        loopEnabled={loopEnabled}
        loopStartSec={loopStartSec}
        loopEndSec={loopEndSec}
        masterVolume={masterVolume}
        limiterEnabled={limiterEnabled}
        peakLevelDb={peakLevelDb}
        onTogglePlay={handleTogglePlay}
        onStop={handleStop}
        onSeek={handleSeek}
        onToggleLoop={() => setLoopEnabled(!loopEnabled)}
        onChangeMasterVolume={setMasterVolume}
        onToggleLimiter={() => setLimiterEnabled(!limiterEnabled)}
      />

      {/* 3. Studio Workspace Tabs & Active Feature Panel */}
      <main className="flex-1">
        <WorkspaceTabs
          activeTab={activeWorkspaceTab}
          onChangeTab={handleChangeWorkspaceTab}
          hasLoadedTrack={hasLoadedTrack}
          separationStatus={separationStatus}
          pitchSpeed={pitchSpeed}
        />

        {activeWorkspaceTab === 'separation' ? (
          <div
            id="panel-separation"
            role="tabpanel"
            aria-labelledby="tab-separation"
          >
            <StemMixerTimeline
              hasLoadedTrack={hasLoadedTrack}
              stems={stems}
              currentTimeSec={currentTimeSec}
              durationSec={song?.durationSec ?? 0}
              qualityMode={qualityMode}
              activePreset={activePreset}
              separationStatus={separationStatus}
              onChangeQualityMode={setQualityMode}
              onRunSeparation={() => void handleRunSeparation()}
              onRunSpectralFallback={() => void runSpectralSeparation()}
              onOpenModelManager={() => setIsModelModalOpen(true)}
              onApplyPreset={handleApplyPreset}
              onToggleMute={handleToggleMute}
              onToggleSolo={handleToggleSolo}
              onChangeStemVolume={handleChangeStemVolume}
              onSeek={handleSeek}
            />
          </div>
        ) : (
          <div
            id="panel-pitch-speed"
            role="tabpanel"
            aria-labelledby="tab-pitch-speed"
          >
            <PitchSpeedDeck
              settings={pitchSpeed}
              originalKey={song?.originalKey ?? '--'}
              originalBpm={song?.originalBpm ?? 0}
              onUpdateSettings={handleUpdatePitchSpeed}
              onResetPitchSpeed={handleResetPitchSpeed}
            />
          </div>
        )}
      </main>

      {/* Modals */}
      <YouTubeImportModal
        isOpen={isYouTubeModalOpen}
        onClose={() => setIsYouTubeModalOpen(false)}
      />

      <ModelManagerModal
        isOpen={isModelModalOpen}
        models={models}
        computeDevice={computeDevice}
        onChangeComputeDevice={handleChangeComputeDevice}
        modelsDir={modelsDir}
        onOpenModelsFolder={
          isTauriDesktopRuntime() && modelsDir ? handleOpenModelsFolder : undefined
        }
        onOpenUpdateCheck={() => {
          setIsModelModalOpen(false);
          void handleCheckForUpdates();
        }}
        onClose={() => setIsModelModalOpen(false)}
        onDownloadModel={handleDownloadModel}
        onPauseDownload={handlePauseDownload}
        onImportLocalModel={handleImportLocalModel}
        onImportDesktopOnnx={
          isTauriDesktopRuntime()
            ? (modelId) => void handleImportDesktopOnnx(modelId)
            : undefined
        }
        onDeleteCachedModel={handleDeleteCachedModel}
      />

      <UpdateModal
        isOpen={isUpdateModalOpen}
        isChecking={isCheckingUpdate}
        result={updateCheckResult}
        errorMsg={updateCheckError}
        onCheckAgain={() => void handleCheckForUpdates()}
        onClose={() => setIsUpdateModalOpen(false)}
      />

      {song && (
        <ExportModal
          isOpen={isExportModalOpen}
          song={song}
          settings={pitchSpeed}
          exportFolderLabel={exportFolderLabel}
          isCustomExportFolder={isCustomExportFolder}
          isSaveAsMode={!isDesktopRuntime && browserExportMode === 'save_as'}
          supportsSaveAs={browserSaveAsCapable}
          isInIframe={embeddedFrame}
          canOpenExportFolder={canOpenExportFolder}
          onPickExportFolder={handlePickExportFolder}
          onEnableSaveAsMode={browserSaveAsCapable ? handleEnableSaveAsMode : undefined}
          onResetExportFolder={handleResetExportFolder}
          onOpenExportFolder={handleOpenExportFolder}
          onClose={() => setIsExportModalOpen(false)}
          onTriggerExport={handleTriggerExport}
        />
      )}

      {separationToast && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-5 right-5 z-50 flex items-center gap-3 rounded-xl border border-emerald-400/50 bg-[#0F1424]/95 px-4 py-3 text-xs font-semibold text-emerald-200 shadow-2xl shadow-emerald-950/50 backdrop-blur-xl"
        >
          <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
          <span>{separationToast}</span>
          <button
            type="button"
            onClick={() => setSeparationToast(null)}
            aria-label="Đóng thông báo"
            className="cursor-pointer rounded p-0.5 text-slate-400 hover:bg-white/10 hover:text-white"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};
