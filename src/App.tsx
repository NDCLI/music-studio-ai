import React, { useEffect, useRef, useState } from 'react';
import { StudioHeader } from './components/StudioHeader';
import { TransportBar } from './components/TransportBar';
import { PitchSpeedDeck } from './components/PitchSpeedDeck';
import { StemMixerTimeline } from './components/StemMixerTimeline';
import { ModelManagerModal } from './components/ModelManagerModal';
import { ExportModal, ExportFormat, ExportTarget } from './components/ExportModal';
import { YouTubeImportModal } from './components/YouTubeImportModal';
import { LOCAL_AI_MODELS } from './data/modelsCatalog';
import {
  generateStemPeaks,
  StudioPreviewAudioEngine,
} from './services/audioSynthesizer';
import {
  deleteCachedModelFromDb,
  downloadModelFromHuggingFace,
  importLocalOnnxModel,
  listCachedModelsFromDb,
} from './services/modelStorage';
import { separateAudioBuffer3Stage } from './services/stemSeparator';
import { syncPitchSpeedToRustBackend } from './services/tauriBridge';
import { YouTubeTrackResolvedInfo } from './services/youtubeParser';
import {
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
    modelSource: 'UVR_MDXNET_KARA_2.onnx',
    colorHex: '#EC4899',
    badgeClass: 'text-pink-400',
    volume: 100,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: true,
    peaks: generateStemPeaks('lead_vocal', 140),
  },
  {
    id: 'backing_vocal',
    name: 'Giọng Bè (Backing Vocal / Harmony)',
    subtitle: 'Bè phối âm, hợp xướng & điệp khúc phụ họa',
    modelSource: 'UVR_MDXNET_KARA_2.onnx',
    colorHex: '#F59E0B',
    badgeClass: 'text-amber-400',
    volume: 90,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: true,
    peaks: generateStemPeaks('backing_vocal', 140),
  },
  {
    id: 'instrumental',
    name: 'Nhạc Nền (Instrumental Beat)',
    subtitle: 'Trống, Bass, Guitar, Piano & Synth giữ nguyên độ nảy',
    modelSource: 'UVR-MDX-NET-Inst_HQ_4.onnx',
    colorHex: '#10B981',
    badgeClass: 'text-emerald-400',
    volume: 100,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: false,
    peaks: generateStemPeaks('instrumental', 140),
  },
  {
    id: 'reverb_tail',
    name: 'Đuôi Vang (Vocal Reverb / Echo Tail)',
    subtitle: 'Không gian vang phòng thu tách rời để nâng tone không méo pha',
    modelSource: 'Reverb_HQ_By_FoxJoy.onnx',
    colorHex: '#06B6D4',
    badgeClass: 'text-cyan-400',
    volume: 55,
    pan: 0,
    muted: false,
    solo: false,
    formantLocked: false,
    peaks: generateStemPeaks('reverb_tail', 140),
  },
];

export const App: React.FC = () => {
  const [song, setSong] = useState<SongMetadata>({
    title: 'Bản Ballad Phòng Thu (Demo Đa Kênh)',
    artist: 'Music Studio AI',
    fileName: 'Studio_Ballad_Master_44k.wav',
    sampleRate: 44100,
    channels: 2,
    durationSec: 215,
    originalBpm: 124,
    originalKey: 'Am',
    isCustomFile: false,
    sourceType: 'studio_demo',
  });

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
  const [currentTimeSec, setCurrentTimeSec] = useState(32.4);
  const [peakLevelDb, setPeakLevelDb] = useState(-14.2);
  const [loopEnabled, setLoopEnabled] = useState(true);
  const [loopStartSec] = useState(30);
  const [loopEndSec] = useState(75);
  const [masterVolume, setMasterVolume] = useState(100);
  const [limiterEnabled, setLimiterEnabled] = useState(true);

  const [qualityMode, setQualityMode] =
    useState<SeparationQualityMode>('fast_mdx');
  const [activePreset, setActivePreset] =
    useState<SeparationPresetId>('full_mix');
  const [separationStatus, setSeparationStatus] =
    useState<SeparationStageStatus>({
      isRunning: false,
      isSeparated: true,
      currentStage: 3,
      progressPercent: 100,
      stageMessage:
        'Sẵn sàng 4 rãnh Demo đa kênh (Mở file Nhạc/Video để tách phổ thực tế)',
      executionBackend: '4 Stems Active',
    });

  const [models, setModels] = useState<LocalAiModelSpec[]>(LOCAL_AI_MODELS);
  const [isYouTubeModalOpen, setIsYouTubeModalOpen] = useState(false);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);

  const audioEngineRef = useRef<StudioPreviewAudioEngine | null>(null);
  const downloadAbortControllersRef = useRef<Record<string, AbortController>>({});
  const stemsRef = useRef(stems);
  stemsRef.current = stems;
  const pitchSpeedRef = useRef(pitchSpeed);
  pitchSpeedRef.current = pitchSpeed;

  useEffect(() => {
    audioEngineRef.current = new StudioPreviewAudioEngine();

    void listCachedModelsFromDb()
      .then((cachedMap) => {
        setModels((prev) =>
          prev.map((m) => {
            const hit = cachedMap[m.id];
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
      })
      .catch(() => {
        // Ignore IndexedDB unavailability in restricted environments
      });

    return () => {
      audioEngineRef.current?.stop();
    };
  }, []);

  const handleTogglePlay = () => {
    const engine = audioEngineRef.current;
    if (!engine) return;

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
            if (next >= song.durationSec) {
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

    const sourceType = detectFileSourceType(file);

    try {
      const decoded = await engine.loadAudioFile(file);
      const cleanTitle = file.name.replace(/\.[^/.]+$/, '');
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
          peaks: decoded.peaks[s.id] ?? s.peaks,
        }))
      );
      setSeparationStatus({
        isRunning: false,
        isSeparated: false,
        currentStage: 0,
        progressPercent: 0,
        stageMessage: `Đã nạp "${file.name}" — Bấm "Tách Nhạc AI Local (3 Tầng)" để tách riêng Vocal, Bè & Nhạc nền.`,
      });
      setCurrentTimeSec(0);
    } catch {
      const cleanTitle = file.name.replace(/\.[^/.]+$/, '');
      setSong((prev) => ({
        ...prev,
        title: cleanTitle,
        fileName: file.name,
        isCustomFile: true,
        sourceType,
      }));
    }
  };

  const handleImportYouTubeTrack = (info: YouTubeTrackResolvedInfo) => {
    const engine = audioEngineRef.current;
    if (engine && isPlaying) {
      engine.stop();
      setIsPlaying(false);
    }

    setSong({
      title: info.title,
      artist: info.authorName,
      fileName: `yt_${info.parsed.videoId}.m4a`,
      sampleRate: 44100,
      channels: 2,
      durationSec: info.durationSec,
      originalBpm: info.detectedBpm,
      originalKey: info.detectedKey,
      isCustomFile: false,
      sourceType: 'youtube_single',
      youtubeInfo: {
        videoId: info.parsed.videoId,
        canonicalUrl: info.parsed.canonicalWatchUrl,
        wasInPlaylist: info.parsed.wasInPlaylist,
        ignoredPlaylistId: info.parsed.ignoredPlaylistId,
        playlistIndex: info.parsed.playlistIndex,
        thumbnailUrl: info.thumbnailUrl,
      },
    });

    setStems((prev) =>
      prev.map((s) => ({
        ...s,
        peaks: info.peaks[s.id] ?? s.peaks,
      }))
    );
    setCurrentTimeSec(info.parsed.startTimeSec ?? 0);
  };

  const handleTriggerExport = (
    baseName: string,
    target: ExportTarget,
    format: ExportFormat
  ): string[] => {
    const engine = audioEngineRef.current;
    if (!engine) return [];

    const outputs = engine.renderExportAudioFiles(
      baseName,
      stems,
      pitchSpeed,
      target,
      format
    );
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

    return names;
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

  const handleRunSeparation = async () => {
    if (separationStatus.isRunning) return;
    const engine = audioEngineRef.current;
    const customBuffer = engine?.getCustomAudioBuffer();

    const getModelLabel = (stemId: StemId) => {
      if (stemId === 'instrumental') {
        return qualityMode === 'fast_mdx'
          ? 'UVR-MDX-NET-Inst_HQ_4.onnx'
          : 'mel_band_roformer_kim_ft_unwa.onnx';
      }
      if (stemId === 'reverb_tail') {
        return 'Reverb_HQ_By_FoxJoy.onnx';
      }
      return qualityMode === 'fast_mdx'
        ? 'UVR_MDXNET_KARA_2.onnx'
        : 'mel_band_roformer_karaoke_aufr33_viperx.onnx';
    };

    if (customBuffer && engine) {
      setSeparationStatus({
        isRunning: true,
        isSeparated: false,
        currentStage: 1,
        progressPercent: 1,
        stageMessage: 'Đang khởi tạo bộ biến đổi STFT 3 tầng & Overlap-Add 50%...',
      });

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

      engine.setSeparatedStems(result.stems);
      setStems((prev) =>
        prev.map((s) => ({
          ...s,
          modelSource: getModelLabel(s.id),
          peaks: result.peaks[s.id] ?? s.peaks,
        }))
      );

      setSeparationStatus({
        isRunning: false,
        isSeparated: true,
        currentStage: 3,
        progressPercent: 100,
        stageMessage:
          'Hoàn tất tách 4 kênh (Lead Vocal, Bè, Nhạc nền, Reverb) bằng STFT Overlap-Add 50%!',
        executionBackend: 'STFT 3-Stage Separated',
      });
      return;
    }

    const stage1Model =
      qualityMode === 'fast_mdx'
        ? 'UVR-MDX-NET-Inst_HQ_4.onnx (n_fft=5120, dim_f=2560)'
        : 'mel_band_roformer_kim_ft_unwa.onnx (SDR 13.1 dB)';
    const stage2Model =
      qualityMode === 'fast_mdx'
        ? 'UVR_MDXNET_KARA_2.onnx (n_fft=5120, dim_f=2048)'
        : 'mel_band_roformer_karaoke_aufr33_viperx.onnx';

    setSeparationStatus({
      isRunning: true,
      isSeparated: false,
      currentStage: 1,
      progressPercent: 15,
      stageMessage: `Tầng 1/3 (Demo): Đang tách Nhạc Nền & Tổng Vocal bằng ${stage1Model}...`,
    });

    window.setTimeout(() => {
      setSeparationStatus({
        isRunning: true,
        isSeparated: false,
        currentStage: 2,
        progressPercent: 56,
        stageMessage: `Tầng 2/3 (Demo): Đang tách Vocal Chính & Giọng Bè bằng ${stage2Model}...`,
      });
    }, 650);

    window.setTimeout(() => {
      setSeparationStatus({
        isRunning: true,
        isSeparated: false,
        currentStage: 3,
        progressPercent: 86,
        stageMessage:
          'Tầng 3/3 (Demo): Đang tách Vocal Khô & Đuôi Vang bằng Reverb_HQ_By_FoxJoy.onnx (n_fft=7680)...',
      });
    }, 1300);

    window.setTimeout(() => {
      setSeparationStatus({
        isRunning: false,
        isSeparated: true,
        currentStage: 3,
        progressPercent: 100,
        stageMessage:
          'Hoàn tất tách 4 rãnh trên bản Demo (Mở file MP3/WAV/MP4 để chạy tách phổ thực tế)!',
        executionBackend: 'Demo 4-Stem Mix',
      });
      setStems((prev) =>
        prev.map((s) => ({
          ...s,
          modelSource: getModelLabel(s.id),
          peaks: generateStemPeaks(s.id, 140, Math.random() * 0.4 + 0.8),
        }))
      );
    }, 1900);
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
      const errMessage = isAborted
        ? 'Đã tạm dừng tải model. Bấm Tải HuggingFace để tải lại hoặc Nhập .ONNX Có Sẵn.'
        : `Không thể tải trực tiếp qua trình duyệt (có thể do CORS/Firewall mạng công ty): ${
            err instanceof Error ? err.message : String(err)
          }. Hãy bấm link Repo để tải bằng trình duyệt rồi chọn "Nhập .ONNX Có Sẵn".`;

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

  const handleDeleteCachedModel = async (modelId: string) => {
    await deleteCachedModelFromDb(modelId);
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

  return (
    <div className="min-h-screen bg-[#080B14] text-slate-100 flex flex-col">
      {/* 1. Studio Top Header */}
      <StudioHeader
        song={song}
        semitones={pitchSpeed.semitones}
        speed={pitchSpeed.speed}
        installedModelCount={installedCount}
        totalModelCount={models.length}
        onSelectAudioFile={handleSelectAudioFile}
        onOpenYouTubeModal={() => setIsYouTubeModalOpen(true)}
        onOpenModelManager={() => setIsModelModalOpen(true)}
        onOpenExportModal={() => setIsExportModalOpen(true)}
      />

      {/* 2. Master Transport Bar */}
      <TransportBar
        isPlaying={isPlaying}
        currentTimeSec={currentTimeSec}
        durationSec={song.durationSec}
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

      {/* 3. Main Pitch Shifting & Speed Control Deck */}
      <main className="flex-1">
        <PitchSpeedDeck
          settings={pitchSpeed}
          originalKey={song.originalKey}
          originalBpm={song.originalBpm}
          onUpdateSettings={handleUpdatePitchSpeed}
          onResetPitchSpeed={handleResetPitchSpeed}
        />

        {/* 4. Local AI Stem Separation & 4-Track Waveform Mixer */}
        <StemMixerTimeline
          stems={stems}
          currentTimeSec={currentTimeSec}
          durationSec={song.durationSec}
          qualityMode={qualityMode}
          activePreset={activePreset}
          separationStatus={separationStatus}
          onChangeQualityMode={setQualityMode}
          onRunSeparation={handleRunSeparation}
          onApplyPreset={handleApplyPreset}
          onToggleMute={handleToggleMute}
          onToggleSolo={handleToggleSolo}
          onChangeStemVolume={handleChangeStemVolume}
          onSeek={handleSeek}
        />
      </main>

      {/* Modals */}
      <YouTubeImportModal
        isOpen={isYouTubeModalOpen}
        onClose={() => setIsYouTubeModalOpen(false)}
        onImportYouTubeTrack={handleImportYouTubeTrack}
      />

      <ModelManagerModal
        isOpen={isModelModalOpen}
        models={models}
        onClose={() => setIsModelModalOpen(false)}
        onDownloadModel={handleDownloadModel}
        onPauseDownload={handlePauseDownload}
        onImportLocalModel={handleImportLocalModel}
        onDeleteCachedModel={handleDeleteCachedModel}
      />

      <ExportModal
        isOpen={isExportModalOpen}
        song={song}
        settings={pitchSpeed}
        onClose={() => setIsExportModalOpen(false)}
        onTriggerExport={handleTriggerExport}
      />
    </div>
  );
};
