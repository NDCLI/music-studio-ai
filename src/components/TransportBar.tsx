import React from 'react';
import {
  Pause,
  Play,
  Repeat,
  RotateCcw,
  ShieldAlert,
  Square,
  Volume2,
} from 'lucide-react';

interface TransportBarProps {
  isPlaying: boolean;
  currentTimeSec: number;
  durationSec: number;
  loopEnabled: boolean;
  loopStartSec: number;
  loopEndSec: number;
  masterVolume: number;
  limiterEnabled: boolean;
  onTogglePlay: () => void;
  onStop: () => void;
  onSeek: (timeSec: number) => void;
  onToggleLoop: () => void;
  onChangeMasterVolume: (vol: number) => void;
  onToggleLimiter: () => void;
}

export function formatTimecode(sec: number): string {
  const safe = Math.max(0, sec);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const hundredths = Math.floor((safe % 1) * 100);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(
    hundredths
  ).padStart(2, '0')}`;
}

export const TransportBar: React.FC<TransportBarProps> = ({
  isPlaying,
  currentTimeSec,
  durationSec,
  loopEnabled,
  loopStartSec,
  loopEndSec,
  masterVolume,
  limiterEnabled,
  onTogglePlay,
  onStop,
  onSeek,
  onToggleLoop,
  onChangeMasterVolume,
  onToggleLimiter,
}) => {
  const progressPercent =
    durationSec > 0 ? Math.min(100, (currentTimeSec / durationSec) * 100) : 0;

  return (
    <section
      aria-label="Master Transport Controls"
      className="mx-4 mt-3 rounded-2xl border border-white/10 bg-[#0F1424]/90 px-5 py-3.5 shadow-xl backdrop-blur-xl"
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Left: Playback Transport Buttons */}
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() => onSeek(0)}
            title="Về đầu bài (Rewind)"
            className="cursor-pointer flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-slate-800/80 text-slate-300 transition-colors duration-200 hover:border-white/25 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          <button
            type="button"
            onClick={onTogglePlay}
            className={`cursor-pointer inline-flex h-10 items-center gap-2 rounded-xl px-5 text-xs font-bold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
              isPlaying
                ? 'bg-amber-500 text-slate-950 hover:bg-amber-400'
                : 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/20 hover:bg-emerald-400'
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4 fill-current" />
                <span>Tạm Dừng</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>Phát Nhạc (Nghe Thử Tức Thì)</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={onStop}
            title="Dừng hẳn (Stop)"
            className="cursor-pointer flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-slate-800/80 text-slate-300 transition-colors duration-200 hover:border-rose-500/40 hover:text-rose-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
          >
            <Square className="h-4 w-4" />
          </button>

          {/* Timecode Readout */}
          <div className="ml-2 rounded-xl border border-white/10 bg-[#080B14] px-3.5 py-1.5 font-mono text-sm font-semibold tracking-wider">
            <span className="text-emerald-400">{formatTimecode(currentTimeSec)}</span>
            <span className="mx-1.5 text-slate-600">/</span>
            <span className="text-slate-400">{formatTimecode(durationSec)}</span>
          </div>
        </div>

        {/* Center: Master Timeline Scrubber */}
        <div className="flex flex-1 min-w-[220px] flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Tiến trình phát (Click hoặc kéo để tua)</span>
            <span className="font-mono text-slate-300">{progressPercent.toFixed(1)}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={durationSec}
            step={0.05}
            value={currentTimeSec}
            onChange={(e) => onSeek( parseFloat(e.target.value) )}
            aria-label="Thanh tua thời gian bài hát"
            className="studio-slider w-full"
          />
        </div>

        {/* Right: A/B Practice Loop & Master Bus Limiter */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={onToggleLoop}
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 font-mono text-xs font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              loopEnabled
                ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
                : 'border-white/10 bg-slate-800/60 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Repeat className="h-3.5 w-3.5" />
            <span>
              A/B Loop: {formatTimecode(loopStartSec).slice(0, 5)} -{' '}
              {formatTimecode(loopEndSec).slice(0, 5)}
            </span>
          </button>

          <button
            type="button"
            onClick={onToggleLimiter}
            title="True-Peak Soft-Knee Limiter (-0.3 dBFS) chống rè vỡ tiếng khi nâng tone"
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
              limiterEnabled
                ? 'border-emerald-500/40 bg-emerald-500/15 text-emerald-300'
                : 'border-white/10 bg-slate-800/60 text-slate-400'
            }`}
          >
            <ShieldAlert className="h-3.5 w-3.5" />
            <span>Chống Vỡ Tiếng (-0.3 dBFS)</span>
          </button>

          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-[#080B14] px-3 py-1.5">
            <Volume2 className="h-4 w-4 text-slate-400" />
            <input
              type="range"
              min={0}
              max={120}
              value={masterVolume}
              onChange={(e) => onChangeMasterVolume(Number(e.target.value))}
              aria-label="Master Volume"
              className="studio-slider w-20"
            />
            <span className="w-9 text-right font-mono text-xs text-slate-200">
              {masterVolume}%
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};
