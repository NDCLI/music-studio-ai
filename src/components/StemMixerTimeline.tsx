import React, { useEffect, useRef } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Headphones,
  Layers,
  Loader2,
  Mic,
  Music,
  Radio,
  Sparkles,
  Volume2,
  VolumeX,
  Waves,
} from 'lucide-react';
import {
  SeparationPresetId,
  SeparationQualityMode,
  SeparationStageStatus,
  StemId,
  StemTrackState,
} from '../types/studio';

interface WaveformCanvasProps {
  peaks: number[];
  colorHex: string;
  progressRatio: number;
  isMuted: boolean;
  onSeekRatio: (ratio: number) => void;
}

const WaveformCanvas: React.FC<WaveformCanvasProps> = ({
  peaks,
  colorHex,
  progressRatio,
  isMuted,
  onSeekRatio,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const barCount = peaks.length;
    if (barCount === 0) {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    const gap = 2;
    const barWidth = Math.max(1.5, (width - (barCount - 1) * gap) / barCount);
    const centerY = height / 2;

    for (let i = 0; i < barCount; i++) {
      const ratio = i / barCount;
      const amp = peaks[i] ?? 0.2;
      const barHeight = Math.max(4, amp * (height * 0.84));
      const x = i * (barWidth + gap);
      const y = centerY - barHeight / 2;

      if (isMuted) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.22)';
      } else if (ratio <= progressRatio) {
        ctx.fillStyle = colorHex;
      } else {
        ctx.fillStyle = `${colorHex}45`;
      }

      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }

    // Draw Playhead Line
    const playheadX = Math.max(0, Math.min(width, progressRatio * width));
    ctx.strokeStyle = '#F8FAFC';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();
  }, [peaks, colorHex, progressRatio, isMuted]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    onSeekRatio(ratio);
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      className="cursor-pointer h-14 w-full rounded-lg bg-[#080B14]/90"
      title="Click vào sóng âm để tua đến vị trí bất kỳ"
    />
  );
};

interface StemMixerTimelineProps {
  hasLoadedTrack: boolean;
  stems: StemTrackState[];
  currentTimeSec: number;
  durationSec: number;
  qualityMode: SeparationQualityMode;
  activePreset: SeparationPresetId;
  separationStatus: SeparationStageStatus;
  onChangeQualityMode: (mode: SeparationQualityMode) => void;
  onRunSeparation: () => void;
  onRunSpectralFallback?: () => void;
  onOpenModelManager?: () => void;
  onApplyPreset: (preset: SeparationPresetId) => void;
  onToggleMute: (id: StemId) => void;
  onToggleSolo: (id: StemId) => void;
  onChangeStemVolume: (id: StemId, volume: number) => void;
  onSeek: (timeSec: number) => void;
}

const PRESET_BUTTONS: { id: SeparationPresetId; label: string; desc: string }[] = [
  { id: 'full_mix', label: 'Full Mix (Đầy Đủ)', desc: 'Nghe tất cả các kênh' },
  {
    id: 'karaoke_with_backing',
    label: 'Karaoke Có Bè',
    desc: 'Tắt Vocal chính, giữ Nhạc nền + Giọng bè',
  },
  {
    id: 'instrumental_only',
    label: 'Beat Thuần (Instrumental)',
    desc: 'Tắt cả Vocal chính và Bè, chỉ giữ Nhạc nền',
  },
  {
    id: 'practice_backing',
    label: 'Luyện Hát Bè',
    desc: 'Làm nổi bật Giọng bè + Nhạc nền nhỏ 35%',
  },
  {
    id: 'acapella',
    label: 'Acapella (Vocal + Bè)',
    desc: 'Chỉ nghe Vocal chính và Giọng bè',
  },
];

export const StemMixerTimeline: React.FC<StemMixerTimelineProps> = ({
  hasLoadedTrack,
  stems,
  currentTimeSec,
  durationSec,
  qualityMode,
  activePreset,
  separationStatus,
  onChangeQualityMode,
  onRunSeparation,
  onRunSpectralFallback,
  onOpenModelManager,
  onApplyPreset,
  onToggleMute,
  onToggleSolo,
  onChangeStemVolume,
  onSeek,
}) => {
  const progressRatio = durationSec > 0 ? currentTimeSec / durationSec : 0;
  const anySolo = stems.some((s) => s.solo);

  const getStemIcon = (id: StemId) => {
    switch (id) {
      case 'lead_vocal':
        return <Mic className="h-4 w-4 text-pink-400" />;
      case 'backing_vocal':
        return <Headphones className="h-4 w-4 text-amber-400" />;
      case 'instrumental':
        return <Music className="h-4 w-4 text-emerald-400" />;
      case 'reverb_tail':
        return <Waves className="h-4 w-4 text-cyan-400" />;
    }
  };

  return (
    <section
      aria-label="AI Stem Separation and Multi-Stem Mixer"
      className="mx-4 mt-3 mb-6 rounded-2xl border border-white/10 bg-[#0F1424]/90 p-5 shadow-2xl backdrop-blur-xl"
    >
      {/* Top Bar: AI Stem Separation Pipeline Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-pink-500/15 text-pink-400">
            <Layers className="h-5 w-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-sm font-bold text-white">
                TÁCH NHẠC NỀN, VOCAL CHÍNH &amp; GIỌNG BÈ
              </h2>
              {separationStatus.isSeparated && (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-emerald-300">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Đã tách xong &bull; {separationStatus.executionBackend ?? '4 Kênh'}
                </span>
              )}
              {hasLoadedTrack && !separationStatus.isSeparated && !separationStatus.isRunning && (
                <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/35 bg-amber-500/15 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-amber-300">
                  Chưa tách nhạc
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400">
              {separationStatus.isSeparated || separationStatus.isRunning || hasLoadedTrack
                ? separationStatus.stageMessage
                : 'Tách riêng giọng hát chính, giọng bè, nhạc nền và tiếng vang'}
            </p>
          </div>
        </div>

        {/* AI Quality Selector & Trigger Button */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="inline-flex flex-wrap rounded-xl border border-white/10 bg-[#080B14] p-1 gap-1">
            <button
              type="button"
              onClick={() => onChangeQualityMode('htdemucs')}
              className={`cursor-pointer rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors duration-150 ${
                qualityMode === 'htdemucs'
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Combo 1: HTDemucs + RoFormer (Chuẩn Nhất)
            </button>
            <button
              type="button"
              onClick={() => onChangeQualityMode('pro_roformer')}
              className={`cursor-pointer rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors duration-150 ${
                qualityMode === 'pro_roformer'
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Combo 2: Inst_HQ + RoFormer
            </button>
            <button
              type="button"
              onClick={() => onChangeQualityMode('fast_mdx')}
              className={`cursor-pointer rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors duration-150 ${
                qualityMode === 'fast_mdx'
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Combo 3: Inst_HQ + MDX Karaoke 2 (Máy Yếu)
            </button>
          </div>

          <button
            type="button"
            disabled={!hasLoadedTrack || separationStatus.isRunning}
            onClick={onRunSeparation}
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-pink-600 px-4 py-2.5 text-xs font-bold text-white shadow-lg shadow-indigo-500/25 transition-opacity duration-200 hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-45"
          >
            {separationStatus.isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Đang Tách ({separationStatus.progressPercent}%)...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Tách Nhạc AI (3 Tầng)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Missing ONNX Dependencies Guidance (Desktop Mode) */}
      {separationStatus.onnxPrereqMissing && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-500/35 bg-amber-500/10 p-3.5 text-xs text-amber-200">
          <div className="flex items-start gap-2.5 flex-1 min-w-[240px]">
            <AlertCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-amber-300">Chưa đủ tệp để chạy AI ONNX trên máy</div>
              <p className="mt-0.5 text-[11px] text-amber-200/90">
                {separationStatus.onnxPrereqMissing}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {onOpenModelManager && (
              <button
                type="button"
                onClick={onOpenModelManager}
                className="cursor-pointer rounded-lg border border-indigo-400/40 bg-indigo-500/20 px-3 py-1.5 text-xs font-semibold text-indigo-200 hover:bg-indigo-500/30"
              >
                Mở Quản Lý Model AI
              </button>
            )}
          </div>
        </div>
      )}

      {/* Prominent Error Banner when separation fails */}
      {!separationStatus.isRunning &&
        !separationStatus.isSeparated &&
        !separationStatus.onnxPrereqMissing &&
        separationStatus.stageMessage.startsWith('Lỗi') && (
          <div
            role="alert"
            className="mt-4 flex items-start gap-2.5 rounded-xl border border-rose-500/45 bg-rose-500/15 p-3.5 text-xs text-rose-100"
          >
            <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-rose-300">Tách nhạc thất bại</div>
              <p className="mt-0.5 text-[11px] text-rose-200/90 break-words">
                {separationStatus.stageMessage}
              </p>
            </div>
          </div>
        )}

      {/* Live 3-Stage AI Separation Progress Bar (when running) */}
      {separationStatus.isRunning && (
        <div className="mt-4 rounded-xl border border-indigo-500/30 bg-indigo-950/30 p-3.5">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="inline-flex items-center gap-2 font-semibold text-indigo-200">
              <Cpu className="h-4 w-4 text-indigo-400 animate-pulse" />
              {separationStatus.stageMessage}
            </span>
            <span className="font-mono font-bold text-indigo-300">
              Tầng {separationStatus.currentStage}/3 &bull; {separationStatus.progressPercent}%
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full bg-gradient-to-r from-indigo-500 via-pink-500 to-emerald-400 transition-all duration-200"
              style={{ width: `${separationStatus.progressPercent}%` }}
            />
          </div>
        </div>
      )}

      {/* Prominent Success Status Banner after separation completes */}
      {separationStatus.isSeparated && !separationStatus.isRunning && (
        <div
          role="status"
          className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-500/40 bg-emerald-500/15 px-4 py-3 text-xs text-emerald-100"
        >
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
            <div>
              <div className="font-bold text-emerald-300">
                Tách nhạc thành công!
              </div>
              <p className="mt-0.5 text-[11px] text-emerald-200/90">
                {separationStatus.stageMessage}
              </p>
            </div>
          </div>
          <span className="rounded-lg border border-emerald-400/30 bg-emerald-500/20 px-2.5 py-1 font-mono text-[11px] font-semibold text-emerald-200">
            4 Kênh Đã Tách
          </span>
        </div>
      )}

      {/* 1-Click Karaoke / Harmony Presets */}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 inline-flex items-center gap-1 text-xs font-semibold text-slate-400">
            <Radio className="h-3.5 w-3.5 text-indigo-400" />
            Chế độ nghe nhanh:
          </span>
          {PRESET_BUTTONS.map((p) => {
            const active = activePreset === p.id;
            const controlsLocked = !hasLoadedTrack || !separationStatus.isSeparated;
            return (
              <button
                key={p.id}
                type="button"
                disabled={controlsLocked}
                title={controlsLocked ? 'Hãy tách nhạc trước để chọn chế độ nghe từng kênh' : p.desc}
                onClick={() => onApplyPreset(p.id)}
                className={`cursor-pointer rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45 ${
                  active && !controlsLocked
                    ? 'border-indigo-400/60 bg-indigo-500/20 text-white'
                    : 'border-white/10 bg-[#080B14]/80 text-slate-300 hover:border-white/25 hover:text-white'
                }`}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* 4 Multi-Stem Tracks */}
      <div className="mt-4 space-y-3">
        {stems.map((stem) => {
          const isSilenced = stem.muted || (anySolo && !stem.solo);
          return (
            <div
              key={stem.id}
              className={`grid grid-cols-1 gap-3 rounded-xl border p-3.5 transition-colors duration-200 lg:grid-cols-12 items-center ${
                isSilenced
                  ? 'border-white/5 bg-[#080B14]/40 opacity-65'
                  : 'border-white/10 bg-[#080B14]/85'
              }`}
            >
              {/* Left 3 cols: Stem Name, Model Badge & Formant Status */}
              <div className="lg:col-span-3 flex items-start gap-3">
                <div
                  className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/10"
                  style={{ backgroundColor: `${stem.colorHex}18` }}
                >
                  {getStemIcon(stem.id)}
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white truncate">
                      {stem.name}
                    </span>
                    {stem.formantLocked && (
                      <span className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                        Formant Lock
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400 truncate">{stem.subtitle}</p>
                </div>
              </div>

              {/* Middle 3 cols: Mute, Solo & Volume Fader */}
              <div className="lg:col-span-3 flex items-center gap-2.5">
                <button
                  type="button"
                  disabled={!hasLoadedTrack || !separationStatus.isSeparated}
                  onClick={() => onToggleMute(stem.id)}
                  title="Tắt tiếng kênh này (Mute)"
                  className={`cursor-pointer flex h-8 w-9 items-center justify-center rounded-lg border font-mono text-xs font-bold transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45 ${
                    stem.muted
                      ? 'border-rose-500 bg-rose-500 text-white'
                      : 'border-white/15 bg-slate-800/80 text-slate-300 hover:border-white/30'
                  }`}
                >
                  M
                </button>

                <button
                  type="button"
                  disabled={!hasLoadedTrack || !separationStatus.isSeparated}
                  onClick={() => onToggleSolo(stem.id)}
                  title="Chỉ nghe kênh này (Solo)"
                  className={`cursor-pointer flex h-8 w-9 items-center justify-center rounded-lg border font-mono text-xs font-bold transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45 ${
                    stem.solo
                      ? 'border-amber-400 bg-amber-400 text-slate-950'
                      : 'border-white/15 bg-slate-800/80 text-slate-300 hover:border-white/30'
                  }`}
                >
                  S
                </button>

                <div className="flex flex-1 items-center gap-2 rounded-lg border border-white/10 bg-slate-900/80 px-2.5 py-1.5">
                  {isSilenced ? (
                    <VolumeX className="h-3.5 w-3.5 text-rose-400 shrink-0" />
                  ) : (
                    <Volume2 className="h-3.5 w-3.5 text-slate-400 shrink-0" />
                  )}
                  <input
                    type="range"
                    disabled={!hasLoadedTrack || !separationStatus.isSeparated}
                    min={0}
                    max={150}
                    value={stem.volume}
                    onChange={(e) =>
                      onChangeStemVolume(stem.id, parseInt(e.target.value, 10))
                    }
                    aria-label={`Âm lượng ${stem.name}`}
                    className="studio-slider w-full disabled:opacity-45"
                  />
                  <span className="w-10 text-right font-mono text-xs text-slate-200">
                    {stem.volume}%
                  </span>
                </div>
              </div>

              {/* Right 6 cols: Interactive High-DPI Canvas Waveform (hidden until separated) */}
              <div className="relative lg:col-span-6">
                <WaveformCanvas
                  peaks={separationStatus.isSeparated ? stem.peaks : []}
                  colorHex={stem.colorHex}
                  progressRatio={progressRatio}
                  isMuted={isSilenced}
                  onSeekRatio={(ratio) => onSeek(ratio * durationSec)}
                />
                {!separationStatus.isSeparated && (
                  <div className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-lg px-3 text-center">
                    <span className="rounded-md bg-[#080B14]/85 px-2.5 py-1 text-[11px] font-medium text-slate-400">
                      {!hasLoadedTrack
                        ? 'Chưa nạp file âm thanh'
                        : separationStatus.isRunning
                          ? `Đang tách sóng âm... (${separationStatus.progressPercent}%)`
                          : 'Chưa tách — Sóng âm sẽ hiển thị sau khi tách nhạc xong'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};
