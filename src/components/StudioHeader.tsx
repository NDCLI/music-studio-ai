import React, { useEffect, useRef, useState } from 'react';
import {
  AudioWaveform,
  Cpu,
  Download,
  Film,
  FolderOpen,
  Music2,
  Play,
} from 'lucide-react';
import { SongMetadata } from '../types/studio';
import { computeTransposedKey } from '../data/modelsCatalog';

interface StudioHeaderProps {
  song: SongMetadata | null;
  semitones: number;
  speed: number;
  installedModelCount: number;
  totalModelCount: number;
  onSelectAudioFile: (file: File) => void;
  onRequestOpenTrack?: () => void;
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
  onRequestOpenTrack,
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

  const targetKey = song ? computeTransposedKey(song.originalKey, semitones) : '--';
  const effectiveBpm = song ? (song.originalBpm * speed).toFixed(1) : '--';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onSelectAudioFile(file);
    }
    e.target.value = '';
  };

  const renderSourceBadge = () => {
    if (!song) return null;
    if (song.sourceType === 'video_file') {
      return (
        <span className="inline-flex items-center gap-1 rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-amber-300">
          <Film className="h-3 w-3" />
          VIDEO AUDIO
        </span>
      );
    }
    return (
      <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-emerald-300">
        AUDIO THỰC TẾ
      </span>
    );
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
              <span className="text-xs font-semibold text-slate-100 truncate max-w-[190px]">
                {song ? song.title : 'Chưa mở file âm thanh'}
              </span>
              {!isScrolled && renderSourceBadge()}
            </div>
            {!isScrolled && (
              <div className="font-mono text-[10px] text-slate-400 truncate max-w-[230px]">
                {song
                  ? `${song.fileName} • ${(song.sampleRate / 1000).toFixed(1)}kHz Stereo`
                  : 'Chọn file Nhạc hoặc Video từ máy để bắt đầu'}
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
            <span className="text-slate-300">{song ? song.originalKey : '--'}</span>
            <span className="text-slate-500">&rarr;</span>
            <span
              className={
                !song
                  ? 'text-slate-400'
                  : semitones === 0
                    ? 'text-emerald-400'
                    : 'rounded bg-indigo-500/20 px-1.5 py-0.2 text-indigo-300'
              }
            >
              {song
                ? `${targetKey} (${semitones >= 0 ? `+${semitones}` : semitones} st)`
                : '--'}
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
            onClick={() => {
              if (onRequestOpenTrack) {
                onRequestOpenTrack();
              } else {
                fileInputRef.current?.click();
              }
            }}
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
            title="Lọc link YouTube đơn từ Playlist"
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-rose-500/35 bg-rose-500/15 text-xs font-semibold text-rose-200 transition-colors duration-200 hover:bg-rose-500/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 ${
              isScrolled ? 'px-2.5 py-1.5' : 'px-3 py-2'
            }`}
          >
            <Play className="h-3 w-3 fill-current text-rose-400" />
            <span>{isScrolled ? 'YouTube' : 'Lọc Link YouTube'}</span>
          </button>

          <button
            type="button"
            onClick={onOpenModelManager}
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/30 bg-indigo-500/10 text-xs font-semibold text-indigo-300 transition-colors duration-200 hover:bg-indigo-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
              isScrolled ? 'px-2.5 py-1.5' : 'px-3 py-2'
            }`}
          >
            <Cpu className="h-3.5 w-3.5 text-indigo-400" />
            <span>{isScrolled ? 'AI' : 'Model AI'}</span>
            <span className="rounded-md bg-indigo-500/30 px-1.5 py-0.5 font-mono text-[10px] text-white">
              {installedModelCount}/{totalModelCount}
            </span>
          </button>

          <button
            type="button"
            disabled={!song}
            onClick={onOpenExportModal}
            className={`cursor-pointer inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 text-xs font-bold text-slate-950 shadow-lg shadow-emerald-500/20 transition-colors duration-200 hover:bg-emerald-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 disabled:cursor-not-allowed disabled:opacity-45 ${
              isScrolled ? 'px-3 py-1.5' : 'px-3.5 py-2'
            }`}
          >
            <Download className="h-3.5 w-3.5" />
            <span>{isScrolled ? 'Xuất' : 'Xuất File'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
