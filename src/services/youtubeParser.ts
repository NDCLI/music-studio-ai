import { MUSICAL_KEYS } from '../data/modelsCatalog';
import { generateStemPeaks } from './audioSynthesizer';
import { StemId } from '../types/studio';

export interface ParsedYouTubeTarget {
  videoId: string;
  canonicalWatchUrl: string;
  wasInPlaylist: boolean;
  ignoredPlaylistId?: string;
  playlistIndex?: number;
  startTimeSec?: number;
}

export interface YouTubeTrackResolvedInfo {
  parsed: ParsedYouTubeTarget;
  title: string;
  authorName: string;
  thumbnailUrl: string;
  durationSec: number;
  detectedKey: string;
  detectedBpm: number;
  peaks: Record<StemId, number[]>;
}

/**
 * Parses any YouTube / YouTube Music / Shorts / youtu.be URL, including URLs with playlist parameters
 * (`list=...&index=...`), and strictly isolates ONLY the currently playing video (`v=<video_id>`).
 */
export function parseYouTubeSingleVideoUrl(rawUrl: string): {
  ok: true;
  data: ParsedYouTubeTarget;
} | {
  ok: false;
  error: string;
} {
  const trimmed = rawUrl.trim();
  if (!trimmed) {
    return { ok: false, error: 'Vui lòng nhập đường dẫn YouTube.' };
  }

  let urlObj: URL;
  try {
    urlObj = new URL(
      trimmed.startsWith('http://') || trimmed.startsWith('https://')
        ? trimmed
        : `https://${trimmed}`
    );
  } catch {
    return { ok: false, error: 'Định dạng đường dẫn URL không hợp lệ.' };
  }

  const host = urlObj.hostname.toLowerCase();
  const isYouTubeDomain =
    host.includes('youtube.com') ||
    host.includes('youtu.be') ||
    host.includes('youtube-nocookie.com');

  if (!isYouTubeDomain) {
    return {
      ok: false,
      error: 'Đường dẫn phải thuộc tên miền youtube.com, music.youtube.com hoặc youtu.be.',
    };
  }

  let videoId = urlObj.searchParams.get('v') ?? '';
  const listParam = urlObj.searchParams.get('list') ?? undefined;
  const indexParam = urlObj.searchParams.get('index');
  const tParam = urlObj.searchParams.get('t') ?? urlObj.searchParams.get('start');

  // Support short URLs (https://youtu.be/<id>?list=...) and /shorts/<id> or /embed/<id>
  if (!videoId) {
    if (host.includes('youtu.be')) {
      const seg = urlObj.pathname.split('/').filter(Boolean)[0];
      if (seg) videoId = seg;
    } else if (urlObj.pathname.startsWith('/shorts/')) {
      const seg = urlObj.pathname.replace('/shorts/', '').split('/')[0];
      if (seg) videoId = seg;
    } else if (urlObj.pathname.startsWith('/embed/')) {
      const seg = urlObj.pathname.replace('/embed/', '').split('/')[0];
      if (seg) videoId = seg;
    } else if (urlObj.pathname.startsWith('/live/')) {
      const seg = urlObj.pathname.replace('/live/', '').split('/')[0];
      if (seg) videoId = seg;
    }
  }

  if (!videoId) {
    if (listParam) {
      return {
        ok: false,
        error:
          'Link này chỉ chứa mã Playlist chung (`list=' +
          listParam +
          '`) mà chưa có mã bài đang phát (`v=`). Hãy mở bài hát bạn muốn trong Playlist rồi copy link trên thanh địa chỉ.',
      };
    }
    return {
      ok: false,
      error: 'Không tìm thấy mã bài hát (`v=...`) trong đường dẫn YouTube.',
    };
  }

  // Clean videoId to valid characters
  const cleanId = videoId.replace(/[^a-zA-Z0-9_-]/g, '');
  if (cleanId.length < 6) {
    return { ok: false, error: 'Mã video YouTube (`v=`) không hợp lệ.' };
  }

  const parsedIndex = indexParam ? parseInt(indexParam, 10) : undefined;
  const parsedStart = tParam ? parseInt(tParam.replace(/[^0-9]/g, ''), 10) : undefined;

  return {
    ok: true,
    data: {
      videoId: cleanId,
      canonicalWatchUrl: `https://www.youtube.com/watch?v=${cleanId}`,
      wasInPlaylist: Boolean(listParam),
      ignoredPlaylistId: listParam,
      playlistIndex: Number.isFinite(parsedIndex) ? parsedIndex : undefined,
      startTimeSec: Number.isFinite(parsedStart) ? parsedStart : undefined,
    },
  };
}

/**
 * Computes a deterministic numeric hash from a string (e.g. videoId)
 * to derive consistent musical characteristics when previewing.
 */
function hashString(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h);
}

/**
 * Fetches real YouTube video title, author, and thumbnail via YouTube oEmbed API
 * for the isolated single track (`canonicalWatchUrl`).
 */
export async function fetchYouTubeTrackMetadata(
  parsed: ParsedYouTubeTarget
): Promise<YouTubeTrackResolvedInfo> {
  const hash = hashString(parsed.videoId);
  const keyRoot = MUSICAL_KEYS[hash % MUSICAL_KEYS.length];
  const isMinor = hash % 2 === 0;
  const detectedKey = `${keyRoot}${isMinor ? 'm' : ''}`;
  const detectedBpm = 92 + (hash % 56);
  const durationSec = 185 + (hash % 95);
  const seedBoost = 0.8 + ((hash % 40) / 100);

  let title = `YouTube Track (${parsed.videoId})`;
  let authorName = 'YouTube Music Stream';
  let thumbnailUrl = `https://i.ytimg.com/vi/${parsed.videoId}/hqdefault.jpg`;

  try {
    const oembedUrl = `https://www.youtube.com/oembed?url=${encodeURIComponent(
      parsed.canonicalWatchUrl
    )}&format=json`;
    const res = await fetch(oembedUrl);
    if (res.ok) {
      const json = (await res.json()) as {
        title?: string;
        author_name?: string;
        thumbnail_url?: string;
      };
      if (json.title) title = json.title;
      if (json.author_name) authorName = json.author_name;
      if (json.thumbnail_url) thumbnailUrl = json.thumbnail_url;
    }
  } catch {
    // Fallback to deterministic metadata if offline
  }

  return {
    parsed,
    title,
    authorName,
    thumbnailUrl,
    durationSec,
    detectedKey,
    detectedBpm,
    peaks: {
      lead_vocal: generateStemPeaks('lead_vocal', 140, seedBoost),
      backing_vocal: generateStemPeaks('backing_vocal', 140, seedBoost * 1.1),
      instrumental: generateStemPeaks('instrumental', 140, seedBoost * 0.95),
      reverb_tail: generateStemPeaks('reverb_tail', 140, seedBoost * 1.05),
    },
  };
}
