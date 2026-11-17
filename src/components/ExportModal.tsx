import React, { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileAudio,
  FolderOpen,
  RotateCcw,
  Sparkles,
  X,
} from 'lucide-react';
import { PitchSpeedSettings, SongMetadata } from '../types/studio';
import { computeTransposedKey } from '../data/modelsCatalog';
import { BrowserAudioExportFormat } from '../services/dspEngine';

export type ExportFormat = BrowserAudioExportFormat;
export type ExportTarget = 'master_mix' | 'all_stems' | 'karaoke_backing';

export interface ExportOutcome {
  files: string[];
  destinationLabel: string;
  savedToLocalFolder: boolean;
  errorMessage?: string;
}

interface ExportModalProps {
  isOpen: boolean;
  song: SongMetadata;
  settings: PitchSpeedSettings;
  exportFolderLabel: string;
  isCustomExportFolder: boolean;
  canOpenExportFolder: boolean;
  onPickExportFolder: () => Promise<string | null>;
  onResetExportFolder: () => void;
  onOpenExportFolder: () => void;
  onClose: () => void;
  onTriggerExport: (
    baseName: string,
    target: ExportTarget,
    format: ExportFormat
  ) => Promise<ExportOutcome>;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  song,
  settings,
  exportFolderLabel,
  isCustomExportFolder,
  canOpenExportFolder,
  onPickExportFolder,
  onResetExportFolder,
  onOpenExportFolder,
  onClose,
  onTriggerExport,
}) => {
  const [format, setFormat] = useState<ExportFormat>('wav24');
  const [target, setTarget] = useState<ExportTarget>('master_mix');
  const [useAiVocalRender, setUseAiVocalRender] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [isPickingFolder, setIsPickingFolder] = useState(false);
  const [folderWarning, setFolderWarning] = useState<string | null>(null);
  const [exportOutcome, setExportOutcome] = useState<ExportOutcome | null>(null);

  if (!isOpen) return null;

  const targetKey = computeTransposedKey(song.originalKey, settings.semitones);
  const baseName = `${song.title.replace(/\s+/g, '_')}_${targetKey}_${Math.round(
    settings.speed * 100
  )}pct`;

  const handlePickFolder = async () => {
    setFolderWarning(null);
    setIsPickingFolder(true);
    try {
      const warning = await onPickExportFolder();
      if (warning) {
        setFolderWarning(warning);
      }
    } finally {
      setIsPickingFolder(false);
    }
  };

  const handleStartExport = async () => {
    setIsExporting(true);
    setExportOutcome(null);
    setFolderWarning(null);
    try {
      const outcome = await onTriggerExport(baseName, target, format);
      setExportOutcome(outcome);
    } catch (err) {
      setExportOutcome({
        files: [],
        destinationLabel: exportFolderLabel,
        savedToLocalFolder: false,
        errorMessage: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
    >
      <div className="max-h-[90vh] w-full max-w-xl overflow-hidden rounded-2xl border border-white/15 bg-[#0F1424] shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/20 text-emerald-400">
              <FileAudio className="h-5 w-5" />
            </div>
            <div>
              <h2 id="export-modal-title" className="text-base font-bold text-white">
                XUẤT FILE ÂM THANH
              </h2>
              <p className="text-xs text-slate-400">
                Tone đích: {targetKey} ({settings.semitones >= 0 ? `+${settings.semitones}` : settings.semitones} st) &bull; Tốc độ: {(settings.speed * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Đóng cửa sổ"
            className="cursor-pointer rounded-xl border border-white/10 bg-slate-800/80 p-2 text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
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
                  desc: 'Xuất 4 file âm thanh độc lập để dùng cho thu âm hoặc mix riêng',
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
              2. Định dạng &amp; Chất lượng âm thanh
            </label>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {[
                { id: 'wav24', label: 'WAV 24-bit PCM' },
                { id: 'wav32f', label: 'WAV 32-bit Float' },
                { id: 'wav16', label: 'WAV 16-bit CD' },
                { id: 'flac', label: 'FLAC 24-bit Lossless' },
              ].map((fmt) => (
                <button
                  key={fmt.id}
                  type="button"
                  onClick={() => setFormat(fmt.id as ExportFormat)}
                  className={`cursor-pointer rounded-xl border py-2.5 px-2 text-center font-mono text-[11px] font-semibold transition-colors ${
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

          {/* Destination Export Folder Selector */}
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-300 mb-2">
              3. Thư mục lưu file xuất
            </label>
            <div className="rounded-xl border border-white/10 bg-[#080B14] p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="text-[11px] text-slate-400">
                    {isCustomExportFolder ? 'Thư mục đã chọn:' : 'Thư mục mặc định:'}
                  </div>
                  <div
                    title={exportFolderLabel}
                    className="mt-0.5 truncate font-mono text-xs font-semibold text-white"
                  >
                    {exportFolderLabel}
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    disabled={isPickingFolder}
                    onClick={handlePickFolder}
                    className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg border border-indigo-500/40 bg-indigo-500/15 px-3 py-1.5 text-xs font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/25 disabled:opacity-60"
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                    <span>{isPickingFolder ? 'Đang Mở...' : 'Chọn Thư Mục...'}</span>
                  </button>
                  {isCustomExportFolder && (
                    <button
                      type="button"
                      onClick={onResetExportFolder}
                      title="Quay lại thư mục mặc định"
                      className="cursor-pointer inline-flex items-center gap-1 rounded-lg border border-white/10 bg-slate-800/80 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 hover:text-white"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      <span>Mặc Định</span>
                    </button>
                  )}
                </div>
              </div>
              {folderWarning && (
                <div className="mt-2 flex items-start gap-1.5 text-[11px] text-amber-300">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                  <span>{folderWarning}</span>
                </div>
              )}
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
                Tăng cường giữ âm sắc tự nhiên cho giọng hát khi xuất
              </div>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Giữ nguyên độ ấm và chất giọng tự nhiên khi dịch chuyển cao độ hoặc tốc độ bài hát.
              </p>
            </div>
          </button>

          {exportOutcome && exportOutcome.errorMessage && (
            <div className="flex items-start gap-2 rounded-xl border border-rose-500/35 bg-rose-500/15 px-4 py-2.5 text-xs text-rose-200">
              <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <span>Không thể lưu file: {exportOutcome.errorMessage}</span>
            </div>
          )}

          {exportOutcome && exportOutcome.files.length > 0 && (
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/15 px-4 py-2.5 text-xs font-semibold text-emerald-300 space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0" />
                  <span>
                    Đã xuất {exportOutcome.files.length} file ({format.toUpperCase()}) vào{' '}
                    {exportOutcome.destinationLabel}:
                  </span>
                </div>
                {canOpenExportFolder && (
                  <button
                    type="button"
                    onClick={onOpenExportFolder}
                    className="cursor-pointer inline-flex items-center gap-1 rounded-lg border border-emerald-400/40 bg-emerald-500/20 px-2.5 py-1 text-[11px] font-bold text-emerald-100 hover:bg-emerald-500/30"
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                    <span>Mở Thư Mục</span>
                  </button>
                )}
              </div>
              <div className="font-mono text-[11px] text-emerald-200 pl-6 break-all">
                {exportOutcome.files.join(', ')}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2.5 border-t border-white/10 px-6 py-4 shrink-0">
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
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-5 py-2 text-xs font-bold text-slate-950 transition-colors hover:bg-emerald-400 disabled:opacity-60"
          >
            <Download className="h-4 w-4" />
            {isExporting ? 'Đang Render...' : 'Bắt Đầu Xuất File'}
          </button>
        </div>
      </div>
    </div>
  );
};
