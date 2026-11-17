import React, { useRef } from 'react';
import {
  AudioWaveform,
  Cpu,
  Download,
  Film,
  FolderOpen,
  Layers,
  Music2,
  Play,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { SongMetadata } from '../types/studio';
import { computeTransposedKey } from '../data/modelsCatalog';

interface StudioHeaderProps {
  song: SongMetadata;
  semitones: number;
  speed: number;
  installedModelCount: number;
  totalModelCount: number;
  onSelectAudioFile: (file: File) => void;
  onOpenYouTubeModal: () => void;
  onOpenModelManager: () => void;
  onOpenExportModal: () => void;
}

export const StudioHeader: React.FC<StudioHeaderProps> = ({
  song,
  semitones,
  speed,
  installedModelCount,
  totalModelCount,
  onSelectAudioFile,
  onOpenYouTubeModal,
  onOpenModelManager,
  onOpenExportModal,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const targetKey = computeTransposedKey(song.originalKey, semitones);
  const effectiveBpm = (song.originalBpm * speed).toFixed(1);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onSelectAudioFile(file);
    }
  };

  const renderSourceBadge = () => {
    switch (song.sourceType) {
      case 'youtube_single':
        return (
          <span className="rounded bg-rose-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-rose-300">
            {song.youtubeInfo?.wasInPlaylist
              ? `YT SINGLE (#${song.youtubeInfo.playlistIndex ?? 1} IN LIST)`
              : 'YOUTUBE SINGLE'}
          </span>
        );
      case 'video_file':
        return (
          <span className="inline-flex items-center gap-1 rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-amber-300">
            <Film className="h-3 w-3" />
            VIDEO AUDIO (MP4/MKV)
          </span>
        );
      case 'audio_file':
        return (
          <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-emerald-300">
            LOCAL AUDIO
          </span>
        );
      default:
        return (
          <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-indigo-300">
            STUDIO DEMO
          </span>
        );
    }
  };

  return (
    <header className="sticky top-3 z-30 mx-4 mt-3 rounded-2xl border border-white/10 bg-[#0F1424]/85 px-5 py-3.5 shadow-2xl backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Brand Identity */}
        <div className="flex items-center gap-3.5">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 via-indigo-500 to-emerald-500 shadow-lg shadow-indigo-500/25">
            <AudioWaveform className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="font-display text-lg tracking-wide text-white">
                MUSIC STUDIO AI
              </h1>
              <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-emerald-400">
                <ShieldCheck className="h-3.5 w-3.5" />
                Rust Single-Binary
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Nâng Hạ Tone Bảo Toàn Formant &bull; Tách Nhạc Nền, Vocal Chính &amp; Bè (Audio / Video MP4, MKV / YouTube)
            </p>
          </div>
        </div>

        {/* Active Song Info & Key/BPM Telemetry */}
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/10 bg-[#080B14]/80 px-4 py-2">
          <Music2 className="h-4 w-4 text-indigo-400" />
          <div className="min-w-[175px]">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-100 truncate max-w-[200px]">
                {song.title}
              </span>
              {renderSourceBadge()}
            </div>
            <div className="font-mono text-[11px] text-slate-400 truncate max-w-[260px]">
              {song.fileName} &bull; {(song.sampleRate / 1000).toFixed(1)}kHz Stereo
            </div>
          </div>

          <div className="h-7 w-px bg-white/10" />

          {/* Live Musical Key Transition */}
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wider text-slate-400">
              Tone Nhạc (Key)
            </span>
            <div className="flex items-center gap-1.5 font-mono text-xs font-semibold">
              <span className="text-slate-300">{song.originalKey}</span>
              <span className="text-slate-500">&rarr;</span>
              <span
                className={
                  semitones === 0
                    ? 'text-emerald-400'
                    : 'rounded bg-indigo-500/20 px-1.5 py-0.2 text-indigo-300'
                }
              >
                {targetKey} ({semitones >= 0 ? `+${semitones}` : semitones} st)
              </span>
            </div>
          </div>

          <div className="h-7 w-px bg-white/10" />

          {/* Live BPM Readout */}
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wider text-slate-400">
              Nhịp Độ (BPM)
            </span>
            <div className="font-mono text-xs font-semibold text-amber-400">
              {effectiveBpm} BPM
            </div>
          </div>
        </div>

        {/* Right Studio Actions */}
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.opus,.mp4,.mkv,.webm,.mov,.avi,.m4v"
            onChange={handleFileChange}
            className="hidden"
          />

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            title="Hỗ trợ cả file Nhạc (MP3, WAV, FLAC, M4A) và file Video (MP4, MKV, WEBM, MOV)"
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl border border-white/15 bg-slate-800/80 px-3.5 py-2 text-xs font-semibold text-slate-200 transition-colors duration-200 hover:border-indigo-400/50 hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <FolderOpen className="h-4 w-4 text-indigo-400" />
            Mở Nhạc / Video (MP4, MKV)
          </button>

          <button
            type="button"
            onClick={onOpenYouTubeModal}
            title="Dán link YouTube hoặc Playlist YouTube (Tự động chỉ lấy bài đang phát)"
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl border border-rose-500/35 bg-rose-500/15 px-3.5 py-2 text-xs font-semibold text-rose-200 transition-colors duration-200 hover:bg-rose-500/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
          >
            <Play className="h-3.5 w-3.5 fill-current text-rose-400" />
            Nạp Link YouTube
          </button>

          <button
            type="button"
            onClick={onOpenModelManager}
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl border border-indigo-500/30 bg-indigo-500/10 px-3.5 py-2 text-xs font-semibold text-indigo-300 transition-colors duration-200 hover:bg-indigo-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <Cpu className="h-4 w-4 text-indigo-400" />
            <span>Model AI Local</span>
            <span className="rounded-md bg-indigo-500/30 px-1.5 py-0.5 font-mono text-[11px] text-white">
              {installedModelCount}/{totalModelCount}
            </span>
          </button>

          <button
            type="button"
            onClick={onOpenExportModal}
            className="cursor-pointer inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2 text-xs font-bold text-slate-950 shadow-lg shadow-emerald-500/20 transition-colors duration-200 hover:bg-emerald-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
          >
            <Download className="h-4 w-4" />
            Xuất File (Export)
          </button>
        </div>
      </div>

      {/* Sub-bar: Hardware & Anti-AhnLab Status */}
      <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t border-white/5 pt-2 text-[11px] text-slate-400">
        <div className="flex items-center gap-4">
          <span className="inline-flex items-center gap-1.5 text-slate-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Nguồn đầu vào: <strong className="font-mono text-white">MP3/WAV/FLAC + Video MP4/MKV/WEBM + YouTube (Single Track)</strong>
          </span>
          <span className="hidden sm:inline-flex items-center gap-1.5">
            <Layers className="h-3.5 w-3.5 text-indigo-400" />
            Chế độ máy công ty: <strong className="text-emerald-400">Anti-AhnLab Lag (Single-Binary / 0 Sidecar .exe)</strong>
          </span>
        </div>
        <div className="inline-flex items-center gap-1.5 font-mono text-indigo-300">
          <Sparkles className="h-3.5 w-3.5" />
          UI Design System: ui-ux-pro-max (Dark OLED + Glass)
        </div>
      </div>
    </header>
  );
};
