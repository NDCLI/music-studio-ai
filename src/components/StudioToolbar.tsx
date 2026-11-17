import React from 'react';
import {
  CheckCircle2,
  Cpu,
  Layers,
  RefreshCw,
  Settings,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  ComputeDeviceMode,
  LocalAiModelSpec,
  SeparationQualityMode,
} from '../types/studio';
import { APP_VERSION } from '../services/updateChecker';

interface StudioToolbarProps {
  computeDevice: ComputeDeviceMode;
  onChangeComputeDevice: (device: ComputeDeviceMode) => void;
  qualityMode: SeparationQualityMode;
  onChangeQualityMode: (mode: SeparationQualityMode) => void;
  models: LocalAiModelSpec[];
  installedModelCount: number;
  isCheckingUpdate: boolean;
  updateAvailableVersion?: string | null;
  onOpenModelManager: () => void;
  onCheckForUpdates: () => void;
}

const DEVICE_OPTIONS: {
  id: ComputeDeviceMode;
  label: string;
  title: string;
  icon: 'auto' | 'gpu' | 'cpu';
}[] = [
  {
    id: 'auto',
    label: 'AUTO',
    title: 'Tự động chọn GPU nếu có sẵn, tự chuyển CPU nếu không hỗ trợ',
    icon: 'auto',
  },
  {
    id: 'gpu',
    label: 'GPU',
    title: 'Ưu tiên tăng tốc phần cứng GPU (DirectML)',
    icon: 'gpu',
  },
  {
    id: 'cpu',
    label: 'CPU',
    title: 'Chạy bằng CPU đa luồng',
    icon: 'cpu',
  },
];

export const StudioToolbar: React.FC<StudioToolbarProps> = ({
  computeDevice,
  onChangeComputeDevice,
  qualityMode,
  onChangeQualityMode,
  models,
  installedModelCount,
  isCheckingUpdate,
  updateAvailableVersion,
  onOpenModelManager,
  onCheckForUpdates,
}) => {
  const requiredIds =
    qualityMode === 'fast_mdx'
      ? ['uvr_mdx_inst_hq1', 'uvr_mdx_kara_2']
      : qualityMode === 'htdemucs'
        ? ['htdemucs_4stems', 'mel_band_roformer_karaoke']
        : qualityMode === 'htdemucs_mdx_kara'
          ? ['htdemucs_4stems', 'uvr_mdx_kara_2']
          : ['uvr_mdx_inst_hq1', 'mel_band_roformer_karaoke'];

  const installedRequiredCount = requiredIds.filter(
    (id) => models.find((m) => m.id === id)?.installed
  ).length;
  const isSelectedProfileReady = installedRequiredCount === requiredIds.length;

  return (
    <section
      aria-label="Thanh công cụ thiết bị, model và cập nhật"
      className="mx-4 mt-2.5 rounded-2xl border border-white/10 bg-[#0F1424]/90 px-4 py-2.5 shadow-xl backdrop-blur-xl"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Part 1: Compute Device Selection (AUTO / GPU / CPU) */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-400">
            <Zap className="h-3.5 w-3.5 text-amber-400" />
            <span>Thiết bị:</span>
          </span>
          <div
            role="radiogroup"
            aria-label="Chọn thiết bị xử lý"
            className="inline-flex rounded-xl border border-white/10 bg-[#080B14] p-0.5"
          >
            {DEVICE_OPTIONS.map((opt) => {
              const active = computeDevice === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  title={opt.title}
                  onClick={() => onChangeComputeDevice(opt.id)}
                  className={`cursor-pointer inline-flex items-center gap-1 rounded-lg px-2.5 py-1 font-mono text-[11px] font-bold transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
                    active
                      ? opt.id === 'gpu'
                        ? 'bg-emerald-500/25 text-emerald-300 border border-emerald-400/40'
                        : opt.id === 'cpu'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-400/40'
                          : 'border border-indigo-400/50 bg-indigo-600 text-white'
                      : 'border border-transparent text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {opt.icon === 'auto' && <Sparkles className="h-3 w-3" />}
                  {opt.icon === 'gpu' && <Zap className="h-3 w-3" />}
                  {opt.icon === 'cpu' && <Cpu className="h-3 w-3" />}
                  <span>{opt.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Part 2: Settings (Manage Models) & Check for Updates */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onOpenModelManager}
            title="Cài đặt thư mục Model AI, tải từ HuggingFace hoặc nhập file .ONNX"
            className="cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/35 bg-indigo-500/15 px-3 py-1.5 text-xs font-semibold text-indigo-200 transition-colors duration-200 hover:bg-indigo-500/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
          >
            <Settings className="h-3.5 w-3.5 text-indigo-300" />
            <span>Quản Lý Model</span>
            <span className="rounded bg-indigo-500/30 px-1.5 py-0.5 font-mono text-[10px] text-white">
              {installedModelCount}/{models.length}
            </span>
          </button>

          <button
            type="button"
            disabled={isCheckingUpdate}
            onClick={onCheckForUpdates}
            title={`Kiểm tra cập nhật ứng dụng (hiện tại v${APP_VERSION})`}
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 disabled:opacity-60 ${
              updateAvailableVersion
                ? 'border-amber-400/50 bg-amber-500/20 text-amber-200 hover:bg-amber-500/30'
                : 'border-white/15 bg-slate-800/80 text-slate-200 hover:border-white/30 hover:text-white'
            }`}
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${isCheckingUpdate ? 'animate-spin text-indigo-400' : updateAvailableVersion ? 'text-amber-300' : 'text-emerald-400'}`}
            />
            <span>
              {updateAvailableVersion
                ? `Có bản mới v${updateAvailableVersion}`
                : 'Kiểm Tra Cập Nhật'}
            </span>
          </button>
        </div>
      </div>
    </section>
  );
};
