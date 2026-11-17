import React, { useEffect, useRef, useState } from 'react';
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
import { isTauriDesktopRuntime } from '../services/tauriBridge';

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
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setIsScrolled(window.scrollY > 36);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const isDesktopIpc = isTauriDesktopRuntime();
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
            VIDEO AUDIO
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
    <header
      className={`sticky top-2.5 z-30 mx-4 mt-3 rounded-2xl border border-white/10 bg-[#0F1424]/92 shadow-2xl backdrop-blur-xl transition-all duration-200 ${
        isScrolled ? 'px-4 py-2' : 'px-5 py-3'
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Brand Identity (compacts cleanly on scroll) */}
        <div className="flex items-center gap-3">
          <div
            className={`flex items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 via-indigo-500 to-emerald-500 shadow-lg shadow-indigo-500/25 transition-all duration-200 ${
              isScrolled ? 'h-8 w-8' : 'h-10 w-10'
            }`}
          >
            <AudioWaveform className={isScrolled ? 'h-4 w-4 text-white' : 'h-5 w-5 text-white'} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1
                className={`font-display tracking-wide text-white transition-all ${
                  isScrolled ? 'text-sm' : 'text-base'
                }`}
              >
                MUSIC STUDIO AI
              </h1>
              {!isScrolled && (
                <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-emerald-400">
                  <ShieldCheck className="h-3 w-3" />
                  {isDesktopIpc ? 'Tauri Rust IPC' : 'Rust Single-Binary'}
                </span>
              )}
            </div>
            {!isScrolled && (
              <p className="text-[11px] text-slate-400">
                Nâng Hạ Tone Bảo Toàn Formant &bull; Tách Nhạc Nền, Vocal &amp; Bè (Audio / MP4, MKV / YouTube)
              </p>
            )}
          </div>
        </div>

        {/* Active Song Info & Key/BPM Telemetry */}
        <div
          className={`flex flex-wrap items-center gap-2.5 rounded-xl border border-white/10 bg-[#080B14]/85 transition-all ${
            isScrolled ? 'px-3 py-1' : 'px-3.5 py-1.5'
          }`}
        >
          <Music2 className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
          <div className="min-w-[130px]">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-100 truncate max-w-[165px]">
                {song.title}
              </span>
              {!isScrolled && renderSourceBadge()}
            </div>
            {!isScrolled && (
              <div className="font-mono text-[10px] text-slate-400 truncate max-w-[230px]">
                {song.fileName} &bull; {(song.sampleRate / 1000).toFixed(1)}kHz Stereo
              </div>
            )}
          </div>

          <div className="h-6 w-px bg-white/10" />

          {/* Live Musical Key Transition */}
          <div className="flex items-center gap-1.5 font-mono text-xs font-semibold">
            {!isScrolled && (
              <span className="text-[10px] font-sans uppercase tracking-wider text-slate-400 mr-0.5">
                Key:
              </span>
            )}
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

          <div className="h-6 w-px bg-white/10" />

          {/* Live BPM Readout */}
          <div className="font-mono text-xs font-semibold text-amber-400">
            {effectiveBpm} BPM
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
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-white/15 bg-slate-800/80 text-xs font-semibold text-slate-200 transition-colors duration-200 hover:border-indigo-400/50 hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              isScrolled ? 'px-2.5 py-1.5' : 'px-3 py-2'
            }`}
          >
            <FolderOpen className="h-3.5 w-3.5 text-indigo-400" />
            <span>{isScrolled ? 'Mở File' : 'Mở Nhạc / Video'}</span>
          </button>

          <button
            type="button"
            onClick={onOpenYouTubeModal}
            title="Dán link YouTube hoặc Playlist YouTube (Tự động chỉ lấy bài đang phát)"
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-rose-500/35 bg-rose-500/15 text-xs font-semibold text-rose-200 transition-colors duration-200 hover:bg-rose-500/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 ${
              isScrolled ? 'px-2.5 py-1.5' : 'px-3 py-2'
            }`}
          >
            <Play className="h-3 w-3 fill-current text-rose-400" />
            <span>{isScrolled ? 'YouTube' : 'Nạp Link YouTube'}</span>
          </button>

          <button
            type="button"
            onClick={onOpenModelManager}
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/30 bg-indigo-500/10 text-xs font-semibold text-indigo-300 transition-colors duration-200 hover:bg-indigo-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              isScrolled ? 'px-2.5 py-1.5' : 'px-3 py-2'
            }`}
          >
            <Cpu className="h-3.5 w-3.5 text-indigo-400" />
            <span>{isScrolled ? 'AI' : 'Model AI Local'}</span>
            <span className="rounded-md bg-indigo-500/30 px-1.5 py-0.5 font-mono text-[10px] text-white">
              {installedModelCount}/{totalModelCount}
            </span>
          </button>

          <button
            type="button"
            onClick={onOpenExportModal}
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 text-xs font-bold text-slate-950 shadow-lg shadow-emerald-500/20 transition-colors duration-200 hover:bg-emerald-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
              isScrolled ? 'px-3 py-1.5' : 'px-3.5 py-2'
            }`}
          >
            <Download className="h-3.5 w-3.5" />
            <span>{isScrolled ? 'Xuất' : 'Xuất File'}</span>
          </button>
        </div>
      </div>

      {/* Sub-bar hidden automatically on scroll so content beneath stays unobstructed */}
      {!isScrolled && (
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-white/5 pt-1.5 text-[11px] text-slate-400">
          <div className="flex flex-wrap items-center gap-4">
            <span className="inline-flex items-center gap-1.5 text-slate-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              IPC Bridge:{' '}
              <strong className="font-mono text-white">
                {isDesktopIpc
                  ? 'Tauri v2 Native Rust IPC Active'
                  : 'Browser Studio Mode (0 .exe) • Sẵn sàng ghép Tauri v2'}
              </strong>
            </span>
            <span className="hidden md:inline-flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5 text-indigo-400" />
              Máy công ty:{' '}
              <strong className="text-emerald-400">Anti-AhnLab (Single-Binary / 0 Sidecar .exe)</strong>
            </span>
          </div>
          <div className="inline-flex items-center gap-1.5 font-mono text-indigo-300">
            <Sparkles className="h-3.5 w-3.5" />
            ui-ux-pro-max (Dark OLED)
          </div>
        </div>
      )}
    </header>
  );
};
