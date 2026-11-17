import React from 'react';
import {
  CheckCircle2,
  Cpu,
  Download,
  HardDrive,
  Loader2,
  ShieldCheck,
  X,
} from 'lucide-react';
import { LocalAiModelSpec } from '../types/studio';

interface ModelManagerModalProps {
  isOpen: boolean;
  models: LocalAiModelSpec[];
  onClose: () => void;
  onDownloadModel: (modelId: string) => void;
}

export const ModelManagerModal: React.FC<ModelManagerModalProps> = ({
  isOpen,
  models,
  onClose,
  onDownloadModel,
}) => {
  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-manager-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
    >
      <div className="max-h-[88vh] w-full max-w-4xl overflow-hidden rounded-2xl border border-white/15 bg-[#0F1424] shadow-2xl flex flex-col">
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
                100% Model mã nguồn mở miễn phí chạy offline qua crate `ort` (load-dynamic)
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

        {/* Info Banner */}
        <div className="mx-6 mt-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-500/25 bg-emerald-500/10 px-4 py-2.5 text-xs text-emerald-200">
          <span className="inline-flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
            Liên kết động `onnxruntime.dll` có chữ ký số Microsoft &mdash; Không sinh file `.exe` phụ trợ, không gây lag AhnLab.
          </span>
          <span className="font-mono text-[11px] text-emerald-300">
            Cache: ./models/*.onnx
          </span>
        </div>

        {/* Models Table / Cards */}
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {models.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border border-white/10 bg-[#080B14]/90 p-4 transition-colors hover:border-white/20"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs font-bold text-indigo-300">
                      {m.fileName}
                    </span>
                    <span className="rounded bg-white/10 px-2 py-0.5 font-mono text-[11px] text-emerald-400">
                      SDR: {m.sdrDb}
                    </span>
                    <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                      {m.sizeMb} MB
                    </span>
                  </div>
                  <h3 className="mt-1 text-sm font-semibold text-white">
                    {m.taskTitle}
                  </h3>
                  <p className="text-xs text-slate-400">
                    Kiến trúc: {m.architecture} &bull; Phần cứng: {m.recommendedDevice}
                  </p>
                </div>

                <div>
                  {m.installed ? (
                    <span className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-500/30 bg-emerald-500/15 px-3 py-1.5 text-xs font-semibold text-emerald-300">
                      <CheckCircle2 className="h-4 w-4" />
                      Sẵn Sàng (Offline)
                    </span>
                  ) : m.downloading ? (
                    <span className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/40 bg-indigo-500/20 px-3 py-1.5 font-mono text-xs font-semibold text-indigo-200">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Đang tải {m.downloadProgress ?? 0}%
                    </span>
                  ) : (
                    <button
                      type="button"
                      onClick={() => onDownloadModel(m.id)}
                      className="cursor-pointer inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-3.5 py-1.5 text-xs font-bold text-white transition-colors hover:bg-indigo-500"
                    >
                      <Download className="h-3.5 w-3.5" />
                      Tải Model Miễn Phí
                    </button>
                  )}
                </div>
              </div>

              {/* Technical STFT Parameters */}
              <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/5 pt-2.5 font-mono text-[11px] text-slate-400">
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
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end border-t border-white/10 px-6 py-3.5">
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
