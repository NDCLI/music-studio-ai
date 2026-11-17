import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cpu,
  Download,
  ExternalLink,
  Loader2,
  RefreshCw,
  X,
  AlertCircle,
} from 'lucide-react';
import { APP_VERSION, AppUpdateCheckResult } from '../services/updateChecker';
import {
  downloadAndInstallUpdateViaRust,
  isTauriDesktopRuntime,
  openExternalUrl,
} from '../services/tauriBridge';

interface UpdateModalProps {
  isOpen: boolean;
  isChecking: boolean;
  result: AppUpdateCheckResult | null;
  errorMsg: string | null;
  onCheckAgain: () => void;
  onClose: () => void;
}

export const UpdateModal: React.FC<UpdateModalProps> = ({
  isOpen,
  isChecking,
  result,
  errorMsg,
  onCheckAgain,
  onClose,
}) => {
  const [isUpdating, setIsUpdating] = useState(false);
  const [updatePercent, setUpdatePercent] = useState(0);
  const [updateError, setUpdateError] = useState<string | null>(null);

  useEffect(() => {
    if (!isUpdating) return;
    let unlisten: (() => void) | undefined;
    const tauriEvent = (
      window as unknown as {
        __TAURI__?: {
          event?: {
            listen: (
              event: string,
              handler: (ev: { payload: { percent: number } }) => void
            ) => Promise<() => void>;
          };
        };
      }
    ).__TAURI__?.event;

    if (tauriEvent?.listen) {
      void tauriEvent
        .listen('update-progress', (ev) => {
          if (typeof ev.payload?.percent === 'number') {
            setUpdatePercent(ev.payload.percent);
          }
        })
        .then((fn) => {
          unlisten = fn;
        });
    }

    return () => {
      if (unlisten) unlisten();
    };
  }, [isUpdating]);

  if (!isOpen) return null;

  const handleDirectUpdate = async () => {
    if (!result?.installerUrl) return;
    if (!isTauriDesktopRuntime()) {
      void openExternalUrl(result.installerUrl);
      return;
    }
    setUpdateError(null);
    setUpdatePercent(0);
    setIsUpdating(true);
    try {
      await downloadAndInstallUpdateViaRust(
        result.installerUrl,
        result.installerName ?? `MusicStudioAI-v${result.latestVersion}-setup.exe`
      );
    } catch (err) {
      setUpdateError(err instanceof Error ? err.message : String(err));
      setIsUpdating(false);
    }
  };

  const formattedDate = result?.publishedAt
    ? new Date(result.publishedAt).toLocaleString('vi-VN')
    : null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="update-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
    >
      <div className="w-full max-w-lg overflow-hidden rounded-2xl border border-white/15 bg-[#0F1424] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <h2 id="update-modal-title" className="text-base font-bold text-white">
                KIỂM TRA CẬP NHẬT ỨNG DỤNG
              </h2>
              <p className="font-mono text-xs text-slate-400">
                Phiên bản hiện tại: v{APP_VERSION}
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
        <div className="p-6 space-y-4">
          {isChecking && (
            <div className="flex items-center justify-center gap-2.5 rounded-xl border border-indigo-500/30 bg-indigo-500/10 py-6 text-xs font-semibold text-indigo-200">
              <RefreshCw className="h-4 w-4 animate-spin text-indigo-400" />
              <span>Đang kiểm tra phiên bản mới từ GitHub Releases...</span>
            </div>
          )}

          {!isChecking && errorMsg && (
            <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/35 bg-rose-500/10 p-4 text-xs text-rose-200">
              <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-bold text-rose-300">Không thể kiểm tra cập nhật</div>
                <p className="mt-1 text-[11px] text-rose-200/90">{errorMsg}</p>
              </div>
            </div>
          )}

          {!isChecking && !errorMsg && result && (
            <div className="space-y-3">
              {result.hasUpdate ? (
                <div className="rounded-xl border border-amber-500/40 bg-amber-500/15 p-4 text-xs text-amber-100 space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-amber-300">
                      Đã có phiên bản mới: v{result.latestVersion}
                    </span>
                    {formattedDate && (
                      <span className="font-mono text-[10px] text-amber-200/80">
                        {formattedDate}
                      </span>
                    )}
                  </div>
                  {result.releaseNotes && (
                    <p className="whitespace-pre-line text-[11px] text-amber-100/85 max-h-36 overflow-y-auto rounded-lg bg-black/25 p-2.5">
                      {result.releaseNotes}
                    </p>
                  )}
                  {isUpdating && (
                    <div className="rounded-lg bg-black/30 p-2.5 space-y-1.5">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-amber-200 font-semibold">
                          Đang tải bản cập nhật &amp; chuẩn bị cài đặt...
                        </span>
                        <span className="font-mono font-bold text-amber-300">
                          {updatePercent}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full bg-amber-400 transition-all duration-200"
                          style={{ width: `${updatePercent}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {updateError && (
                    <p className="text-[11px] text-rose-300">
                      Lỗi cập nhật: {updateError}
                    </p>
                  )}
                  <div className="flex flex-wrap items-center gap-2 pt-1">
                    {result.installerUrl && (
                      <button
                        type="button"
                        disabled={isUpdating}
                        onClick={() => void handleDirectUpdate()}
                        className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg bg-amber-400 px-3.5 py-1.5 text-xs font-bold text-slate-950 hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {isUpdating ? (
                          <>
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            <span>Đang cập nhật ({updatePercent}%)...</span>
                          </>
                        ) : (
                          <>
                            <Download className="h-3.5 w-3.5" />
                            <span>
                              {isTauriDesktopRuntime()
                                ? `Cập Nhật Ngay (v${result.latestVersion})`
                                : `Tải bộ cài (${result.installerName ?? '.exe'})`}
                            </span>
                          </>
                        )}
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => void openExternalUrl(result.releaseUrl)}
                      className="cursor-pointer inline-flex items-center gap-1 rounded-lg border border-amber-400/40 bg-amber-500/20 px-3 py-1.5 text-xs font-semibold text-amber-100 hover:bg-amber-500/30"
                    >
                      <span>Xem trang Release</span>
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-3 rounded-xl border border-emerald-500/35 bg-emerald-500/15 p-4 text-xs text-emerald-100">
                  <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <div className="font-bold text-emerald-300">
                      Bạn đang dùng phiên bản mới nhất (v{result.currentVersion})
                    </div>
                    <p className="text-[11px] text-emerald-200/85">
                      Bản phát hành mới nhất trên GitHub: <strong className="font-mono">v{result.latestVersion}</strong>
                      {formattedDate ? ` (${formattedDate})` : ''}.
                    </p>
                    <button
                      type="button"
                      onClick={() => void openExternalUrl(result.releaseUrl)}
                      className="cursor-pointer mt-1 inline-flex items-center gap-1 text-[11px] text-indigo-300 hover:underline"
                    >
                      <span>Xem ghi chú phát hành trên GitHub</span>
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between gap-2.5 border-t border-white/10 px-6 py-4">
          <button
            type="button"
            disabled={isChecking}
            onClick={onCheckAgain}
            className="cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/40 bg-indigo-500/15 px-3.5 py-2 text-xs font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/25 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isChecking ? 'animate-spin' : ''}`} />
            <span>Kiểm Tra Lại</span>
          </button>
          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-xl border border-white/10 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};
