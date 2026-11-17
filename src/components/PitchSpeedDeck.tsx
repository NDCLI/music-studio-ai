import React from 'react';
import {
  Activity,
  CheckCircle2,
  Gauge,
  Minus,
  Plus,
  RotateCcw,
  Sliders,
  Sparkles,
  Wand2,
} from 'lucide-react';
import { PitchEngineId, PitchSpeedSettings } from '../types/studio';
import { computeTransposedKey, PITCH_ENGINES } from '../data/modelsCatalog';

interface PitchSpeedDeckProps {
  settings: PitchSpeedSettings;
  originalKey: string;
  originalBpm: number;
  onUpdateSettings: (partial: Partial<PitchSpeedSettings>) => void;
  onResetPitchSpeed: () => void;
}

const SPEED_PRESETS = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5];

export const PitchSpeedDeck: React.FC<PitchSpeedDeckProps> = ({
  settings,
  originalKey,
  originalBpm,
  onUpdateSettings,
  onResetPitchSpeed,
}) => {
  const targetKey = computeTransposedKey(originalKey, settings.semitones);
  const currentBpm = (originalBpm * settings.speed).toFixed(1);
  const activeEngine =
    PITCH_ENGINES.find((e) => e.id === settings.pitchEngine) ?? PITCH_ENGINES[0];

  const adjustSemitones = (delta: number) => {
    const next = Math.max(-12, Math.min(12, settings.semitones + delta));
    onUpdateSettings({ semitones: next });
  };

  return (
    <section
      aria-label="Pitch Shifting and Time Stretching Studio Deck"
      className="mx-4 mt-3 grid grid-cols-1 gap-4 lg:grid-cols-12"
    >
      {/* LEFT DECK (7 cols): PITCH SHIFTER & FORMANT PRESERVATION */}
      <div className="lg:col-span-7 rounded-2xl border border-white/10 bg-[#0F1424]/90 p-5 shadow-xl backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400">
              <Sliders className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">
                NÂNG HẠ TONE (PITCH SHIFTER)
              </h2>
              <p className="text-xs text-slate-400">
                Dịch chuyển cao độ thời gian thực giữ nguyên âm sắc giọng hát tự nhiên
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => onUpdateSettings({ semitones: 0, cents: 0 })}
            className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-800/80 px-2.5 py-1.5 text-xs font-medium text-slate-300 transition-colors duration-200 hover:border-white/25 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Về Tone Gốc (0 st)
          </button>
        </div>

        {/* Main Semitones Control Row */}
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-12 items-center">
          {/* Semitone Step Counter & Key Badge */}
          <div className="sm:col-span-5 flex flex-col justify-between rounded-xl border border-indigo-500/25 bg-[#080B14] p-3.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Nửa Cung (Semitones)
              </span>
              <span className="rounded bg-indigo-500/20 px-2 py-0.5 font-mono text-xs font-bold text-indigo-300">
                {originalKey} &rarr; {targetKey}
              </span>
            </div>

            <div className="my-3 flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => adjustSemitones(-1)}
                aria-label="Hạ 1 nửa cung"
                className="cursor-pointer flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-slate-800 text-white transition-colors duration-200 hover:bg-indigo-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
              >
                <Minus className="h-4 w-4" />
              </button>

              <div className="text-center font-mono">
                <span className="text-2xl font-bold text-white">
                  {settings.semitones > 0
                    ? `+${settings.semitones}`
                    : settings.semitones}
                </span>
                <span className="ml-1 text-xs text-slate-400">st</span>
              </div>

              <button
                type="button"
                onClick={() => adjustSemitones(1)}
                aria-label="Nâng 1 nửa cung"
                className="cursor-pointer flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-slate-800 text-white transition-colors duration-200 hover:bg-indigo-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
              >
                <Plus className="h-4 w-4" />
              </button>
            </div>

            {/* Quick Semitone Jump Pills */}
            <div className="grid grid-cols-5 gap-1">
              {[-4, -2, 0, 2, 4].map((val) => (
                <button
                  key={val}
                  type="button"
                  onClick={() => onUpdateSettings({ semitones: val })}
                  className={`cursor-pointer rounded-md py-1 font-mono text-[11px] font-semibold transition-colors duration-150 ${
                    settings.semitones === val
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800/70 text-slate-400 hover:bg-slate-700 hover:text-white'
                  }`}
                >
                  {val > 0 ? `+${val}` : val}
                </button>
              ))}
            </div>
          </div>

          {/* Sliders: Semitones Range & Fine-tune Cents */}
          <div className="sm:col-span-7 flex flex-col gap-3.5 rounded-xl border border-white/5 bg-[#080B14]/60 p-3.5">
            <div>
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">
                  Thanh trượt Nâng / Hạ Tone (-12 đến +12 st)
                </span>
                <span className="font-mono font-semibold text-indigo-400">
                  {settings.semitones > 0
                    ? `+${settings.semitones}`
                    : settings.semitones}{' '}
                  semitones
                </span>
              </div>
              <input
                type="range"
                min={-12}
                max={12}
                step={1}
                value={settings.semitones}
                onChange={(e) =>
                  onUpdateSettings({ semitones: parseInt(e.target.value, 10) })
                }
                aria-label="Thanh trượt nâng hạ tone theo nửa cung"
                className="studio-slider w-full"
              />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
                <span>-12 st (1 Quãng 8)</span>
                <span>0 st (Gốc)</span>
                <span>+12 st (1 Quãng 8)</span>
              </div>
            </div>

            <div className="border-t border-white/5 pt-2.5">
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">
                  Tinh chỉnh Vi Cao Độ (Fine-tune Cents)
                </span>
                <span className="font-mono font-semibold text-emerald-400">
                  {settings.cents > 0 ? `+${settings.cents}` : settings.cents} cents
                </span>
              </div>
              <input
                type="range"
                min={-50}
                max={50}
                step={1}
                value={settings.cents}
                onChange={(e) =>
                  onUpdateSettings({ cents: parseInt(e.target.value, 10) })
                }
                aria-label="Tinh chỉnh vi cao độ Cents"
                className="studio-slider w-full"
              />
            </div>
          </div>
        </div>

        {/* Anti-Distortion Formant & Open-Source Model Selector */}
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() =>
              onUpdateSettings({
                formantPreservation: !settings.formantPreservation,
              })
            }
            className={`cursor-pointer flex items-start gap-3 rounded-xl border p-3 text-left transition-colors duration-200 ${
              settings.formantPreservation
                ? 'border-emerald-500/40 bg-emerald-500/10'
                : 'border-white/10 bg-slate-900/50 hover:border-white/20'
            }`}
          >
            <CheckCircle2
              className={`mt-0.5 h-4 w-4 shrink-0 ${
                settings.formantPreservation ? 'text-emerald-400' : 'text-slate-500'
              }`}
            />
            <div>
              <div className="text-xs font-bold text-white">
                Bảo Toàn Âm Sắc (Formant Preservation)
              </div>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Giữ nguyên đỉnh cộng hưởng thanh quản, chống hiện tượng méo giọng sóc chuột (chipmunk).
              </p>
            </div>
          </button>

          <button
            type="button"
            onClick={() =>
              onUpdateSettings({
                stemAwareProcessing: !settings.stemAwareProcessing,
              })
            }
            className={`cursor-pointer flex items-start gap-3 rounded-xl border p-3 text-left transition-colors duration-200 ${
              settings.stemAwareProcessing
                ? 'border-indigo-500/40 bg-indigo-500/10'
                : 'border-white/10 bg-slate-900/50 hover:border-white/20'
            }`}
          >
            <Wand2
              className={`mt-0.5 h-4 w-4 shrink-0 ${
                settings.stemAwareProcessing ? 'text-indigo-400' : 'text-slate-500'
              }`}
            />
            <div>
              <div className="text-xs font-bold text-white">
                Stem-Aware Pitch Shift (Đa Kênh Thông Minh)
              </div>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Xử lý riêng Vocal khô (Formant Lock) và Nhạc nền (Transient Lock) trước khi gộp Reverb.
              </p>
            </div>
          </button>
        </div>

        {/* Pitch Engine Selector */}
        <div className="mt-3.5 rounded-xl border border-white/10 bg-[#080B14]/90 p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-300">
              <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
              Chế Độ Xử Lý Cao Độ:
            </span>
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {PITCH_ENGINES.map((engine) => {
              const selected = settings.pitchEngine === (engine.id as PitchEngineId);
              return (
                <button
                  key={engine.id}
                  type="button"
                  onClick={() => onUpdateSettings({ pitchEngine: engine.id })}
                  className={`cursor-pointer rounded-lg border p-2.5 text-left transition-colors duration-200 ${
                    selected
                      ? 'border-indigo-500 bg-indigo-500/20 text-white'
                      : 'border-white/10 bg-slate-900/70 text-slate-300 hover:border-white/25'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-xs font-bold truncate">{engine.name}</span>
                  </div>
                  <span className="mt-1 inline-block rounded bg-white/10 px-1.5 py-0.5 font-mono text-[10px] text-indigo-300">
                    {engine.badge}
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-2 text-[11px] text-slate-400">{activeEngine.description}</p>
        </div>
      </div>

      {/* RIGHT DECK (5 cols): TIME STRETCHING / SPEED CONTROL */}
      <div className="lg:col-span-5 flex flex-col justify-between rounded-2xl border border-white/10 bg-[#0F1424]/90 p-5 shadow-xl backdrop-blur-xl">
        <div>
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3.5">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-amber-500/20 text-amber-400">
                <Gauge className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-white">
                  CHỈNH TỐC ĐỘ BÀI HÁT
                </h2>
                <p className="text-xs text-slate-400">
                  Tăng/giảm nhịp độ độc lập không làm thay đổi cao độ
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={onResetPitchSpeed}
              className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-slate-800/80 px-2.5 py-1.5 text-xs font-medium text-slate-300 transition-colors duration-200 hover:border-white/25 hover:text-white"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset Tất Cả
            </button>
          </div>

          {/* Speed Big Readout & BPM Box */}
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-amber-500/25 bg-[#080B14] p-3.5">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Tốc Độ Phát (Speed)
              </span>
              <div className="mt-1 font-mono text-2xl font-bold text-amber-400">
                {(settings.speed * 100).toFixed(0)}%
                <span className="ml-1.5 text-xs font-normal text-slate-400">
                  ({settings.speed.toFixed(2)}x)
                </span>
              </div>
            </div>

            <div className="rounded-xl border border-white/10 bg-[#080B14] p-3.5">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Nhịp Độ Thực Tế (BPM)
              </span>
              <div className="mt-1 font-mono text-2xl font-bold text-white">
                {currentBpm}
                <span className="ml-1.5 text-xs font-normal text-slate-400">
                  BPM (Gốc: {originalBpm})
                </span>
              </div>
            </div>
          </div>

          {/* Speed Slider */}
          <div className="mt-4 rounded-xl border border-white/5 bg-[#080B14]/60 p-3.5">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-medium text-slate-300">
                Thanh trượt Tốc độ (0.25x &ndash; 2.00x)
              </span>
              <span className="font-mono font-semibold text-amber-400">
                {settings.speed.toFixed(2)}x
              </span>
            </div>
            <input
              type="range"
              min={0.25}
              max={2.0}
              step={0.01}
              value={settings.speed}
              onChange={(e) =>
                onUpdateSettings({ speed: parseFloat(e.target.value) })
              }
              aria-label="Thanh trượt chỉnh tốc độ bài hát"
              className="studio-slider w-full"
            />

            {/* Quick Speed Preset Buttons */}
            <div className="mt-3 grid grid-cols-7 gap-1.5">
              {SPEED_PRESETS.map((preset) => {
                const isSelected = Math.abs(settings.speed - preset) < 0.005;
                return (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => onUpdateSettings({ speed: preset })}
                    className={`cursor-pointer rounded-lg py-1.5 font-mono text-xs font-semibold transition-colors duration-150 ${
                      isSelected
                        ? 'bg-amber-500 text-slate-950 font-bold'
                        : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700 hover:text-white'
                    }`}
                  >
                    {preset}x
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Transient Crisp Lock Toggle */}
        <div className="mt-4">
          <button
            type="button"
            onClick={() =>
              onUpdateSettings({
                transientCrispLock: !settings.transientCrispLock,
              })
            }
            className={`cursor-pointer w-full flex items-start gap-3 rounded-xl border p-3.5 text-left transition-colors duration-200 ${
              settings.transientCrispLock
                ? 'border-amber-500/40 bg-amber-500/10'
                : 'border-white/10 bg-slate-900/50 hover:border-white/20'
            }`}
          >
            <Activity
              className={`mt-0.5 h-4 w-4 shrink-0 ${
                settings.transientCrispLock ? 'text-amber-400' : 'text-slate-500'
              }`}
            />
            <div>
              <div className="text-xs font-bold text-white">
                Khóa Độ Nảy Nhạc Cụ (Transient Phase Reset)
              </div>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Giữ tiếng trống Kick/Snare và tiếng gảy đàn Guitar sắc nét, không bị nhòe pha khi tua chậm 0.5x &ndash; 0.75x.
              </p>
            </div>
          </button>
        </div>
      </div>
    </section>
  );
};
