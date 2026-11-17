import React from 'react';
import { CheckCircle2, Layers, Loader2, Sliders } from 'lucide-react';
import { PitchSpeedSettings, SeparationStageStatus } from '../types/studio';

export type StudioWorkspaceTab = 'separation' | 'pitch_speed';

interface WorkspaceTabsProps {
  activeTab: StudioWorkspaceTab;
  onChangeTab: (tab: StudioWorkspaceTab) => void;
  hasLoadedTrack: boolean;
  separationStatus: SeparationStageStatus;
  pitchSpeed: PitchSpeedSettings;
}

/**
 * Accessible two-tab studio switcher isolating AI Stem Separation and Pitch/Tempo shifting
 * while displaying live cross-tab status badges.
 */
export const WorkspaceTabs: React.FC<WorkspaceTabsProps> = ({
  activeTab,
  onChangeTab,
  hasLoadedTrack,
  separationStatus,
  pitchSpeed,
}) => {
  const isPitchOrSpeedModified =
    pitchSpeed.semitones !== 0 ||
    pitchSpeed.cents !== 0 ||
    Math.abs(pitchSpeed.speed - 1.0) >= 0.005;

  const pitchSummaryLabel = isPitchOrSpeedModified
    ? `${pitchSpeed.semitones > 0 ? `+${pitchSpeed.semitones}` : pitchSpeed.semitones} st • ${pitchSpeed.speed.toFixed(2)}x`
    : '0 st • 1.00x';

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      onChangeTab(activeTab === 'separation' ? 'pitch_speed' : 'separation');
    }
  };

  return (
    <div className="mx-4 mt-3">
      <div
        role="tablist"
        aria-label="Không gian làm việc chính"
        onKeyDown={handleKeyDown}
        className="grid grid-cols-1 gap-2 rounded-2xl border border-white/10 bg-[#0F1424]/90 p-1.5 shadow-xl backdrop-blur-xl sm:grid-cols-2"
      >
        <button
          id="tab-separation"
          type="button"
          role="tab"
          aria-selected={activeTab === 'separation'}
          aria-controls="panel-separation"
          tabIndex={activeTab === 'separation' ? 0 : -1}
          onClick={() => onChangeTab('separation')}
          className={`cursor-pointer flex items-center justify-between gap-2 md:gap-3 rounded-xl border px-3 py-2 md:px-4 md:py-2.5 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-pink-400 ${
            activeTab === 'separation'
              ? 'border-pink-500/45 bg-gradient-to-r from-indigo-600/25 via-pink-600/20 to-pink-500/15 text-white shadow-lg shadow-pink-950/25'
              : 'border-transparent bg-transparent text-slate-400 hover:bg-white/5 hover:text-slate-200'
          }`}
        >
          <div className="flex items-center gap-2.5 md:gap-3 min-w-0">
            <div
              className={`flex h-8 w-8 md:h-9 md:w-9 shrink-0 items-center justify-center rounded-xl transition-colors duration-200 ${
                activeTab === 'separation'
                  ? 'bg-pink-500/25 text-pink-300'
                  : 'bg-slate-800/80 text-slate-400'
              }`}
            >
              <Layers className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <div className="text-xs font-bold tracking-wide truncate">
                TÁCH NHẠC (4 KÊNH)
              </div>
              <div className="hidden lg:block text-[11px] text-slate-400 truncate">
                Tách Vocal chính, Giọng bè, Nhạc nền &amp; Đuôi vang
              </div>
            </div>
          </div>

          <div className="shrink-0">
            {separationStatus.isRunning ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/25 px-2.5 py-1 font-mono text-[10px] font-semibold text-indigo-200">
                <Loader2 className="h-3 w-3 animate-spin" />
                {separationStatus.progressPercent}%
              </span>
            ) : separationStatus.isSeparated ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2.5 py-1 font-mono text-[10px] font-semibold text-emerald-300">
                <CheckCircle2 className="h-3 w-3" />
                Đã tách 4 kênh
              </span>
            ) : hasLoadedTrack ? (
              <span className="rounded-full border border-amber-500/35 bg-amber-500/15 px-2.5 py-1 font-mono text-[10px] font-semibold text-amber-300">
                Chưa tách
              </span>
            ) : null}
          </div>
        </button>

        <button
          id="tab-pitch-speed"
          type="button"
          role="tab"
          aria-selected={activeTab === 'pitch_speed'}
          aria-controls="panel-pitch-speed"
          tabIndex={activeTab === 'pitch_speed' ? 0 : -1}
          onClick={() => onChangeTab('pitch_speed')}
          className={`cursor-pointer flex items-center justify-between gap-2 md:gap-3 rounded-xl border px-3 py-2 md:px-4 md:py-2.5 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
            activeTab === 'pitch_speed'
              ? 'border-indigo-500/50 bg-gradient-to-r from-indigo-600/25 via-indigo-500/20 to-amber-500/10 text-white shadow-lg shadow-indigo-950/25'
              : 'border-transparent bg-transparent text-slate-400 hover:bg-white/5 hover:text-slate-200'
          }`}
        >
          <div className="flex items-center gap-2.5 md:gap-3 min-w-0">
            <div
              className={`flex h-8 w-8 md:h-9 md:w-9 shrink-0 items-center justify-center rounded-xl transition-colors duration-200 ${
                activeTab === 'pitch_speed'
                  ? 'bg-indigo-500/25 text-indigo-300'
                  : 'bg-slate-800/80 text-slate-400'
              }`}
            >
              <Sliders className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <div className="text-xs font-bold tracking-wide truncate">
                NÂNG TONE &amp; TEMPO
              </div>
              <div className="hidden lg:block text-[11px] text-slate-400 truncate">
                Đổi cao độ ±12 nửa cung bảo toàn âm sắc &amp; tốc độ
              </div>
            </div>
          </div>

          <div className="shrink-0">
            <span
              className={`rounded-full px-2.5 py-1 font-mono text-[10px] font-semibold ${
                isPitchOrSpeedModified
                  ? 'border border-indigo-400/40 bg-indigo-500/25 text-indigo-200'
                  : 'border border-white/10 bg-slate-800/80 text-slate-400'
              }`}
            >
              {pitchSummaryLabel}
            </span>
          </div>
        </button>
      </div>
    </div>
  );
};
