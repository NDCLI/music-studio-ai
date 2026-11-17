import React, { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  Filter,
  Info,
  Link2,
  ListMusic,
  Loader2,
  Music2,
  Play,
  X,
} from 'lucide-react';
import {
  fetchYouTubeTrackMetadata,
  parseYouTubeSingleVideoUrl,
  YouTubeTrackResolvedInfo,
} from '../services/youtubeParser';
import { openExternalUrl } from '../services/tauriBridge';

interface YouTubeImportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const YouTubeImportModal: React.FC<YouTubeImportModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [rawUrl, setRawUrl] = useState('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resolvedInfo, setResolvedInfo] =
    useState<YouTubeTrackResolvedInfo | null>(null);

  if (!isOpen) return null;

  const handlePreviewUrl = async (inputUrl: string) => {
    setRawUrl(inputUrl);
    if (!inputUrl.trim()) {
      setErrorMsg(null);
      setResolvedInfo(null);
      return;
    }

    const parsedResult = parseYouTubeSingleVideoUrl(inputUrl);
    if (!parsedResult.ok) {
      setErrorMsg(parsedResult.error);
      setResolvedInfo(null);
      return;
    }

    setErrorMsg(null);
    setIsLoading(true);
    try {
      const metadata = await fetchYouTubeTrackMetadata(parsedResult.data);
      setResolvedInfo(metadata);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="youtube-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-md"
    >
      <div className="w-full max-w-2xl overflow-hidden rounded-2xl border border-white/15 bg-[#0F1424] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/20 text-rose-400">
              <Play className="h-5 w-5 fill-current" />
            </div>
            <div>
              <h2 id="youtube-modal-title" className="text-base font-bold text-white">
                KIỂM TRA &amp; TÁCH LINK YOUTUBE
              </h2>
              <p className="text-xs text-slate-400">
                Tự động loại bỏ tham số Playlist, giữ đúng video đơn đang phát
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
          {/* Input Field */}
          <div>
            <label
              htmlFor="youtube-url-input"
              className="block text-xs font-bold uppercase tracking-wider text-slate-300 mb-2"
            >
              Dán đường dẫn YouTube / YouTube Music / Link Playlist của bạn
            </label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Link2 className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <input
                  id="youtube-url-input"
                  type="text"
                  value={rawUrl}
                  onFocus={(e) => e.currentTarget.select()}
                  onChange={(e) => void handlePreviewUrl(e.target.value)}
                  placeholder="https://youtu.be/... hoặc https://www.youtube.com/watch?v=..."
                  className="w-full rounded-xl border border-white/15 bg-[#080B14] pl-10 pr-9 py-2.5 font-mono text-xs text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
                />
                {rawUrl && (
                  <button
                    type="button"
                    onClick={() => void handlePreviewUrl('')}
                    aria-label="Xóa link"
                    title="Xóa link đang nhập"
                    className="cursor-pointer absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-white"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
              <button
                type="button"
                onClick={() => void handlePreviewUrl(rawUrl)}
                className="cursor-pointer rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-bold text-white transition-colors hover:bg-indigo-500"
              >
                Phân Tích Link
              </button>
            </div>
          </div>

          <div className="flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3.5 text-xs text-amber-200">
            <Info className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
            <span>
              YouTube yêu cầu xác thực bot khi tải luồng âm thanh trực tiếp (<code>LOGIN_REQUIRED</code>). Để tách nhạc, nâng/hạ tone và xuất file thật, hãy tải audio/video về máy rồi mở bằng nút <strong>Mở Nhạc / Video</strong>.
            </span>
          </div>

          {/* Error Banner */}
          {errorMsg && (
            <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3.5 text-xs text-rose-200">
              <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-[#080B14] py-8 text-xs text-slate-300">
              <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
              <span>Đang đọc thông tin video từ YouTube...</span>
            </div>
          )}

          {/* Resolved Single Track Preview */}
          {!isLoading && resolvedInfo && (
            <div className="space-y-3">
              {resolvedInfo.parsed.wasInPlaylist ? (
                <div className="flex items-start gap-2.5 rounded-xl border border-emerald-500/35 bg-emerald-500/10 p-3.5 text-xs text-emerald-200">
                  <Filter className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-bold text-emerald-300">
                      Đã tách riêng đúng bài hát đơn (bỏ qua Playlist)
                    </div>
                    <p className="mt-0.5 text-[11px] text-emerald-200/85">
                      Đã bỏ qua Playlist{' '}
                      <code className="rounded bg-black/40 px-1 py-0.5 font-mono text-amber-300">
                        list={resolvedInfo.parsed.ignoredPlaylistId}
                      </code>{' '}
                      và giữ lại mã video{' '}
                      <code className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-emerald-300">
                        v={resolvedInfo.parsed.videoId}
                      </code>
                      .
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3.5 py-2.5 text-xs text-emerald-200">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                  <span>
                    Link video hợp lệ (Mã video:{' '}
                    <strong className="font-mono">{resolvedInfo.parsed.videoId}</strong>)
                  </span>
                </div>
              )}

              {/* Video Card */}
              <div className="flex flex-col sm:flex-row items-center gap-4 rounded-xl border border-white/10 bg-[#080B14] p-4">
                <img
                  src={resolvedInfo.thumbnailUrl}
                  alt={resolvedInfo.title}
                  className="h-24 w-40 rounded-lg object-cover border border-white/10 shrink-0 bg-slate-900"
                />
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1 rounded bg-rose-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-rose-300">
                      <ListMusic className="h-3 w-3" />
                      SINGLE VIDEO
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      ID: {resolvedInfo.parsed.videoId}
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-white truncate">
                    {resolvedInfo.title}
                  </h3>
                  <p className="text-xs text-slate-400 flex items-center gap-2">
                    <Music2 className="h-3.5 w-3.5 text-indigo-400" />
                    <span>Kênh: {resolvedInfo.authorName}</span>
                  </p>
                  <div className="pt-1 font-mono text-[11px]">
                    <button
                      type="button"
                      onClick={() => void openExternalUrl(resolvedInfo.parsed.canonicalWatchUrl)}
                      className="cursor-pointer inline-flex items-center gap-1 text-indigo-400 hover:underline"
                    >
                      <span>Mở link đơn trên trình duyệt</span>
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  </div>
                </div>
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
        </div>
      </div>
    </div>
  );
};
