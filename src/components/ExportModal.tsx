import React, { useState } from 'react';
import {
  CheckCircle2,
  Download,
  FileAudio,
  Sparkles,
  X,
} from 'lucide-react';
import { PitchSpeedSettings, SongMetadata } from '../types/studio';
import { computeTransposedKey } from '../data/modelsCatalog';

export type ExportFormat = 'wav24' | 'wav32f' | 'flac' | 'mp3_320';
export type ExportTarget = 'master_mix' | 'all_stems' | 'karaoke_backing';

interface ExportModalProps {
  isOpen: boolean;
  song: SongMetadata;
  settings: PitchSpeedSettings;
  onClose: () => void;
  onTriggerExport: (baseName: string, target: ExportTarget) => string[];
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  song,
  settings,
  onClose,
  onTriggerExport,
}) => {
  const [format, setFormat] = useState<ExportFormat>('wav24');
  const [target, setTarget] = useState<ExportTarget>('master_mix');
  const [useAiVocalRender, setUseAiVocalRender] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportedFiles, setExportedFiles] = useState<string[]>([]);

  if (!isOpen) return null;

  const targetKey = computeTransposedKey(song.originalKey, settings.semitones);
  const baseName = `${song.title.replace(/\s+/g, '_')}_${targetKey}_${Math.round(
    settings.speed * 100
  )}pct`;

  const handleStartExport = () => {
    setIsExporting(true);
    setExportedFiles([]);
    window.setTimeout(() => {
      const files = onTriggerExport(baseName, target);
      setIsExporting(false);
      setExportedFiles(files);
    }, 350);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
    >
      <div className="w-full max-w-xl overflow-hidden rounded-2xl border border-white/15 bg-[#0F1424] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/20 text-emerald-400">
              <FileAudio className="h-5 w-5" />
            </div>
            <div>
              <h2 id="export-modal-title" className="text-base font-bold text-white">
                XUẤT FILE ÂM THANH PHÒNG THU (STUDIO EXPORT)
              </h2>
              <p className="text-xs text-slate-400">
                Tone đích: {targetKey} ({settings.semitones >= 0 ? `+${settings.semitones}` : settings.semitones} st) &bull; Tốc độ: {(settings.speed * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-xl border border-white/10 bg-slate-800/80 p-2 text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4">
          {/* Target Stems Selection */}
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-300 mb-2">
              1. Chọn kênh xuất ra
            </label>
            <div className="grid grid-cols-1 gap-2">
              {[
                {
                  id: 'master_mix',
                  title: 'Bản phối hiện tại (Master Mix theo bộ trộn)',
                  desc: 'Gộp các kênh đang bật với thiết lập Tone & Tốc độ hiện tại',
                },
                {
                  id: 'all_stems',
                  title: 'Xuất rời từng kênh (4 Files: Vocal chính, Bè, Nhạc nền, Reverb)',
                  desc: 'Tải xuống 4 file WAV độc lập để đưa vào DAW hoặc mix chuyên sâu',
                },
                {
                  id: 'karaoke_backing',
                  title: 'Bản Karaoke giữ Giọng Bè (Instrumental + Backing Vocal)',
                  desc: 'Tắt Vocal chính nhưng giữ nguyên bè phối âm đã nâng/hạ tone',
                },
              ].map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setTarget(item.id as ExportTarget)}
                  className={`cursor-pointer rounded-xl border p-3 text-left transition-colors ${
                    target === item.id
                      ? 'border-indigo-500 bg-indigo-500/20 text-white'
                      : 'border-white/10 bg-[#080B14] text-slate-300 hover:border-white/25'
                  }`}
                >
                  <div className="text-xs font-bold">{item.title}</div>
                  <div className="mt-0.5 text-[11px] text-slate-400">{item.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Audio Format Selection */}
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-300 mb-2">
              2. Định dạng &amp; Chất lượng (Crate `hound` / Browser WAV Renderer)
            </label>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {[
                { id: 'wav24', label: 'WAV 24-bit PCM' },
                { id: 'wav32f', label: 'WAV 32-bit Float' },
                { id: 'flac', label: 'FLAC Lossless' },
                { id: 'mp3_320', label: 'MP3 320 kbps' },
              ].map((fmt) => (
                <button
                  key={fmt.id}
                  type="button"
                  onClick={() => setFormat(fmt.id as ExportFormat)}
                  className={`cursor-pointer rounded-xl border py-2.5 text-center font-mono text-xs font-semibold transition-colors ${
                    format === fmt.id
                      ? 'border-emerald-400 bg-emerald-500/20 text-emerald-300'
                      : 'border-white/10 bg-[#080B14] text-slate-300 hover:border-white/25'
                  }`}
                >
                  {fmt.label}
                </button>
              ))}
            </div>
          </div>

          {/* AI Neural Pitch Render Option */}
          <button
            type="button"
            onClick={() => setUseAiVocalRender(!useAiVocalRender)}
            className={`cursor-pointer w-full flex items-start gap-3 rounded-xl border p-3.5 text-left transition-colors ${
              useAiVocalRender
                ? 'border-indigo-500/40 bg-indigo-500/15'
                : 'border-white/10 bg-[#080B14]'
            }`}
          >
            <Sparkles
              className={`mt-0.5 h-4 w-4 shrink-0 ${
                useAiVocalRender ? 'text-indigo-400' : 'text-slate-500'
              }`}
            />
            <div>
              <div className="text-xs font-bold text-white">
                Sử dụng AI RMVPE + NSF-HiFiGAN + WORLD khi xuất kênh Vocal
              </div>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Tái tạo sóng âm giọng hát bằng Neural Source-Filter kết hợp bảo toàn Formant, đảm bảo không bị méo tiếng ngay cả khi dịch chuyển &pm;6 semitones.
              </p>
            </div>
          </button>

          {exportedFiles.length > 0 && (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/15 px-4 py-2.5 text-xs font-semibold text-emerald-300 space-y-1">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                <span>Đã xuất và tải xuống {exportedFiles.length} file WAV:</span>
              </div>
              <div className="font-mono text-[11px] text-emerald-200 pl-6">
                {exportedFiles.join(', ')}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2.5 border-t border-white/10 px-6 py-4">
          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-xl border border-white/10 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700"
          >
            Đóng
          </button>
          <button
            type="button"
            disabled={isExporting}
            onClick={handleStartExport}
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-5 py-2 text-xs font-bold text-slate-950 transition-colors hover:bg-emerald-400"
          >
            <Download className="h-4 w-4" />
            {isExporting ? 'Đang Render...' : 'Bắt Đầu Xuất File'}
          </button>
        </div>
      </div>
    </div>
  );
};
