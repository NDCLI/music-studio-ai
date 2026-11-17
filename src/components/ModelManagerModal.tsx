import React, { useRef, useState } from 'react';
import {
  AlertCircle,
  Check,
  CheckCircle2,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  FolderPlus,
  HardDrive,
  Loader2,
  Pause,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import { LocalAiModelSpec } from '../types/studio';
import { openExternalUrl } from '../services/tauriBridge';

interface ModelManagerModalProps {
  isOpen: boolean;
  models: LocalAiModelSpec[];
  onClose: () => void;
  onDownloadModel: (modelId: string) => void;
  onPauseDownload: (modelId: string) => void;
  onImportLocalModel: (modelId: string, file: File) => void;
  onDeleteCachedModel: (modelId: string) => void;
}

function formatBytesMb(bytes?: number): string {
  if (!bytes || bytes <= 0) return '0.0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const ModelManagerModal: React.FC<ModelManagerModalProps> = ({
  isOpen,
  models,
  onClose,
  onDownloadModel,
  onPauseDownload,
  onImportLocalModel,
  onDeleteCachedModel,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [importTargetId, setImportTargetId] = useState<string | null>(null);
  const [copiedModelId, setCopiedModelId] = useState<string | null>(null);

  if (!isOpen) return null;

  const totalCachedBytes = models.reduce(
    (sum, m) => sum + (m.installed ? m.cachedSizeBytes ?? 0 : 0),
    0
  );

  const handleOpenImportPicker = (modelId: string) => {
    setImportTargetId(modelId);
    fileInputRef.current?.click();
  };

  const handleFilePicked = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && importTargetId) {
      onImportLocalModel(importTargetId, file);
    }
    e.target.value = '';
  };

  const handleOpenRepoPage = (sourceRepo: string) => {
    void openExternalUrl(`https://huggingface.co/${sourceRepo}`);
  };

  const handleCopyDirectLink = async (model: LocalAiModelSpec) => {
    try {
      await navigator.clipboard.writeText(model.downloadUrl);
      setCopiedModelId(model.id);
      window.setTimeout(() => setCopiedModelId(null), 1800);
    } catch {
      // Clipboard fallback ignored if blocked
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-manager-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".onnx"
        onChange={handleFilePicked}
        className="hidden"
      />

      <div className="max-h-[90vh] w-full max-w-4xl overflow-hidden rounded-2xl border border-white/15 bg-[#0F1424] shadow-2xl flex flex-col">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <h2 id="model-manager-title" className="text-base font-bold text-white">
                QUẢN LÝ MODEL AI TÁCH NHẠC &amp; PITCH LOCAL (ONNX RUNTIME)
              </h2>
              <p className="text-xs text-slate-400">
                Tải từ HuggingFace hoặc nhập file `.onnx` có sẵn trên máy &bull; Kiểm tra SHA-256 &bull; Lưu Cache Offline
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

        {/* Runtime & Cache Summary Banner */}
        <div className="mx-6 mt-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-500/25 bg-emerald-500/10 px-4 py-2.5 text-xs text-emerald-200">
          <span className="inline-flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
            Rust `ort` (`load-dynamic`, `directml`) dùng `onnxruntime.dll` có chữ ký số Microsoft &mdash; Không tạo `.exe` phụ trợ.
          </span>
          <span className="font-mono text-[11px] text-emerald-300">
            Cache Offline: {formatBytesMb(totalCachedBytes)}
          </span>
        </div>

        {/* Models Cards */}
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {models.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border border-white/10 bg-[#080B14]/90 p-4 transition-colors hover:border-white/20"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs font-bold text-indigo-300">
                      {m.fileName}
                    </span>
                    <span className="rounded bg-white/10 px-2 py-0.5 font-mono text-[11px] text-emerald-400">
                      SDR: {m.sdrDb}
                    </span>
                    <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                      ~{m.sizeMb} MB
                    </span>
                    {m.installed && m.cachedSource && (
                      <span className="rounded bg-indigo-500/20 px-2 py-0.5 font-mono text-[10px] text-indigo-200">
                        {m.cachedSource === 'local_import' ? 'Imported Local' : 'HuggingFace Cache'}
                      </span>
                    )}
                  </div>
                  <h3 className="mt-1 text-sm font-semibold text-white">
                    {m.taskTitle}
                  </h3>
                  <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                    <span>
                      Kiến trúc: {m.architecture} &bull; Phần cứng: {m.recommendedDevice} &bull; Repo:
                    </span>
                    <button
                      type="button"
                      onClick={() => handleOpenRepoPage(m.sourceRepo)}
                      className="cursor-pointer inline-flex items-center gap-1 text-indigo-400 hover:text-indigo-300 hover:underline"
                    >
                      <span>{m.sourceRepo}</span>
                      <ExternalLink className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleCopyDirectLink(m)}
                      title={m.downloadUrl}
                      className="cursor-pointer inline-flex items-center gap-1 rounded border border-white/10 bg-slate-800/80 px-2 py-0.5 text-[10px] font-medium text-slate-300 hover:border-indigo-400/50 hover:text-white"
                    >
                      {copiedModelId === m.id ? (
                        <>
                          <Check className="h-3 w-3 text-emerald-400" />
                          <span className="text-emerald-300">Đã Copy Link</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3 w-3" />
                          <span>Copy Link Tải</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Action Controls */}
                <div className="flex flex-wrap items-center gap-2">
                  {m.installed ? (
                    <>
                      <span className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-500/30 bg-emerald-500/15 px-3 py-1.5 text-xs font-semibold text-emerald-300">
                        <CheckCircle2 className="h-4 w-4" />
                        Sẵn Sàng ({formatBytesMb(m.cachedSizeBytes)})
                      </span>
                      <button
                        type="button"
                        onClick={() => onDeleteCachedModel(m.id)}
                        title="Xóa model khỏi bộ nhớ Cache Offline"
                        className="cursor-pointer rounded-xl border border-rose-500/30 bg-rose-500/10 p-2 text-rose-300 transition-colors hover:bg-rose-500/20"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </>
                  ) : m.downloading ? (
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/40 bg-indigo-500/20 px-3 py-1.5 font-mono text-xs font-semibold text-indigo-200">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        {m.downloadedBytes
                          ? `${formatBytesMb(m.downloadedBytes)} / ${m.sizeMb} MB (${m.downloadProgress ?? 0}%)`
                          : 'Đang tải trực tiếp...'}
                      </span>
                      <button
                        type="button"
                        onClick={() => onPauseDownload(m.id)}
                        className="cursor-pointer inline-flex items-center gap-1 rounded-xl border border-amber-500/40 bg-amber-500/15 px-2.5 py-1.5 text-xs font-semibold text-amber-300 hover:bg-amber-500/25"
                      >
                        <Pause className="h-3.5 w-3.5" />
                        Dừng
                      </button>
                    </div>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => onDownloadModel(m.id)}
                        className="cursor-pointer inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-3.5 py-1.5 text-xs font-bold text-white transition-colors hover:bg-indigo-500"
                      >
                        <Download className="h-3.5 w-3.5" />
                        Tải HuggingFace
                      </button>
                      <button
                        type="button"
                        onClick={() => handleOpenImportPicker(m.id)}
                        title="Nhập file .onnx đã tải sẵn trên máy tính"
                        className="cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-white/15 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 transition-colors hover:border-indigo-400/50 hover:text-white"
                      >
                        <FolderPlus className="h-3.5 w-3.5 text-amber-400" />
                        Nhập .ONNX Có Sẵn
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Download Progress Bar if active */}
              {m.downloading && (
                <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-150"
                    style={{ width: `${Math.max(5, m.downloadProgress ?? 0)}%` }}
                  />
                </div>
              )}

              {/* Error banner if network/CORS blocked download */}
              {m.downloadError && (
                <div className="mt-2.5 flex items-start gap-2 rounded-lg border border-amber-500/35 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
                  <AlertCircle className="h-3.5 w-3.5 text-amber-400 shrink-0 mt-0.5" />
                  <span>{m.downloadError}</span>
                </div>
              )}

              {/* Technical STFT Parameters & SHA-256 Fingerprint */}
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-white/5 pt-2.5 font-mono text-[11px] text-slate-400">
                <div className="flex flex-wrap items-center gap-2">
                  <HardDrive className="h-3.5 w-3.5 text-slate-500" />
                  <span>n_fft: {m.stftSpec.nFft}</span>
                  <span>&bull;</span>
                  <span>hop_length: {m.stftSpec.hopLength}</span>
                  <span>&bull;</span>
                  <span>dim_f: {m.stftSpec.dimF}</span>
                  <span>&bull;</span>
                  <span>dim_t: {m.stftSpec.dimT}</span>
                  <span>&bull;</span>
                  <span>compensate: {m.stftSpec.compensate}</span>
                </div>
                {m.sha256Hex && (
                  <span className="text-emerald-400">
                    SHA-256: {m.sha256Hex.slice(0, 12)}...{m.sha256Hex.slice(-8)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/10 px-6 py-3.5">
          <span className="text-[11px] text-slate-400">
            Bấm <strong>Repo</strong> để mở HuggingFace bằng trình duyệt, hoặc bấm <strong>Copy Link Tải</strong> rồi chọn <strong>Nhập .ONNX Có Sẵn</strong>.
          </span>
          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-xl bg-slate-800 px-5 py-2 text-xs font-bold text-white transition-colors hover:bg-slate-700"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};
