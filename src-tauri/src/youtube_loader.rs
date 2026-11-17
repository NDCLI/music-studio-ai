//! Direct YouTube Single-Track URL Parser & Stream Loader.
//! Supports standard YouTube links, YouTube Music links, Shorts, and Playlist links
//! while strictly extracting ONLY the currently playing video (`v=<video_id>`),
//! ignoring the rest of the playlist (`--no-playlist` behavior).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedYouTubeTarget {
    /// Extracted 11-character YouTube video ID of the currently playing track.
    pub video_id: String,
    /// Canonical single-video URL (`https://www.youtube.com/watch?v=<video_id>`).
    pub canonical_watch_url: String,
    /// True if the original URL contained a `list=...` playlist parameter that was stripped.
    pub was_in_playlist: bool,
    /// Optional playlist ID (`list=...`) detected and ignored.
    pub ignored_playlist_id: Option<String>,
    /// Optional 1-based index (`index=...`) of the active track inside the playlist.
    pub playlist_index: Option<u32>,
    /// Optional start timestamp in seconds (`t=...`).
    pub start_time_sec: Option<u32>,
}

/// Parses any YouTube or YouTube Music URL, including playlist URLs, and isolates
/// only the currently playing video track.
pub fn parse_youtube_single_video_url(raw_url: &str) -> Result<ParsedYouTubeTarget, String> {
    let trimmed = raw_url.trim();
    if trimmed.is_empty() {
        return Err("YouTube URL cannot be empty".to_string());
    }

    let mut video_id: Option<String> = None;
    let mut ignored_playlist_id: Option<String> = None;
    let mut playlist_index: Option<u32> = None;
    let mut start_time_sec: Option<u32> = None;

    // Extract query string if present
    if let Some(query_idx) = trimmed.find('?') {
        let query = &trimmed[query_idx + 1..];
        let query_clean = query.split('#').next().unwrap_or(query);

        for pair in query_clean.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or("");
            let val = kv.next().unwrap_or("");

            match key {
                "v" if !val.is_empty() => {
                    video_id = Some(val.to_string());
                }
                "list" if !val.is_empty() => {
                    ignored_playlist_id = Some(val.to_string());
                }
                "index" if !val.is_empty() => {
                    playlist_index = val.parse::<u32>().ok();
                }
                "t" | "start" if !val.is_empty() => {
                    let digits: String = val.chars().take_while(|c| c.is_ascii_digit()).collect();
                    start_time_sec = digits.parse::<u32>().ok();
                }
                _ => {}
            }
        }
    }

    // Handle short links (https://youtu.be/<id>?list=...) or /shorts/<id> or /embed/<id>
    if video_id.is_none() {
        let without_query = trimmed.split('?').next().unwrap_or(trimmed);
        let without_hash = without_query.split('#').next().unwrap_or(without_query);

        if let Some(pos) = without_hash.find("youtu.be/") {
            let after = &without_hash[pos + "youtu.be/".len()..];
            let candidate = after.split('/').next().unwrap_or("");
            if !candidate.is_empty() {
                video_id = Some(candidate.to_string());
            }
        } else if let Some(pos) = without_hash.find("/shorts/") {
            let after = &without_hash[pos + "/shorts/".len()..];
            let candidate = after.split('/').next().unwrap_or("");
            if !candidate.is_empty() {
                video_id = Some(candidate.to_string());
            }
        } else if let Some(pos) = without_hash.find("/embed/") {
            let after = &without_hash[pos + "/embed/".len()..];
            let candidate = after.split('/').next().unwrap_or("");
            if !candidate.is_empty() {
                video_id = Some(candidate.to_string());
            }
        }
    }

    let vid = video_id.ok_or_else(|| {
        if ignored_playlist_id.is_some() {
            "Link chỉ chứa danh sách phát chung (không có mã bài đang phát `v=`). Vui lòng bấm vào bài hát cụ thể trong Playlist rồi copy link.".to_string()
        } else {
            "Không tìm thấy mã bài hát YouTube hợp lệ trong đường dẫn.".to_string()
        }
    })?;

    if vid.len() < 6 {
        return Err("Mã video YouTube không hợp lệ.".to_string());
    }

    let canonical_watch_url = format!("https://www.youtube.com/watch?v={}", vid);
    let was_in_playlist = ignored_playlist_id.is_some();

    Ok(ParsedYouTubeTarget {
        video_id: vid,
        canonical_watch_url,
        was_in_playlist,
        ignored_playlist_id,
        playlist_index,
        start_time_sec,
    })
}
