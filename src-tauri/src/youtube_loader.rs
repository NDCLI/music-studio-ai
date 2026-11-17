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

    let after_scheme = trimmed.split_once("://").map(|(_, rest)| rest).unwrap_or(trimmed);
    if after_scheme.contains("http://") || after_scheme.contains("https://") {
        return Err("Đường dẫn bị dính nhiều link liên tiếp. Vui lòng dán lại đúng 1 link YouTube.".to_string());
    }

    if vid.len() != 11
        || !vid
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("Mã video YouTube không đúng chuẩn 11 ký tự.".to_string());
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

fn find_or_install_yt_dlp(models_dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let exe_name = if cfg!(target_os = "windows") {
        "yt-dlp.exe"
    } else {
        "yt-dlp"
    };

    if let Ok(cur) = std::env::current_exe() {
        if let Some(dir) = cur.parent() {
            let p = dir.join(exe_name);
            if p.is_file() {
                return Ok(p);
            }
        }
    }

    let in_models = models_dir.join(exe_name);
    if in_models.is_file() {
        return Ok(in_models);
    }

    if let Ok(output) = std::process::Command::new(exe_name).arg("--version").output() {
        if output.status.success() {
            return Ok(std::path::PathBuf::from(exe_name));
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::io::{Read, Write};
        let url = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe";
        let agent = ureq::Agent::new_with_defaults();
        let mut resp = agent
            .get(url)
            .call()
            .map_err(|e| format!("Không tải được yt-dlp.exe: {}", e))?;
        let part = models_dir.join("yt-dlp.exe.part");
        let mut file = std::fs::File::create(&part)
            .map_err(|e| format!("Không tạo được file yt-dlp.exe: {}", e))?;
        let mut reader = resp.body_mut().as_reader();
        let mut buf = [0_u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        }
        file.flush().map_err(|e| e.to_string())?;
        drop(file);
        std::fs::rename(&part, &in_models).map_err(|e| e.to_string())?;
        return Ok(in_models);
    }

    #[cfg(not(target_os = "windows"))]
    Err("yt-dlp not found".to_string())
}

/// Downloads the best audio stream of a single YouTube video via `yt-dlp` into a local temp file.
pub fn download_youtube_audio_file(
    canonical_watch_url: &str,
    video_id: &str,
    models_dir: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    let yt_dlp = find_or_install_yt_dlp(models_dir)?;
    let out_dir = std::env::temp_dir().join("music_studio_yt");
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("Không tạo được thư mục tạm YouTube: {}", e))?;

    // Clean any stale file for this video_id before downloading
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(video_id) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    let out_template = out_dir.join(format!("{}.%(ext)s", video_id));
    let mut cmd = std::process::Command::new(&yt_dlp);
    cmd.arg("--no-playlist")
        .arg("-f")
        .arg("bestaudio[ext=m4a]/bestaudio")
        .arg("-o")
        .arg(&out_template)
        .arg(canonical_watch_url);

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Lỗi khi chạy yt-dlp: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "yt-dlp tải thất bại: {}",
            stderr.lines().last().unwrap_or("Unknown error")
        ));
    }

    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(video_id) && !name.ends_with(".part") {
                    return Ok(p);
                }
            }
        }
    }

    Err("Không tìm thấy file âm thanh sau khi tải từ YouTube.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_youtube_url_rejects_concatenated_urls_and_invalid_ids() {
        let dup = "https://www.youtube.com/watch?v=8nsV1giVhttps://www.youtube.com/watch?v=8nsV1giVLOQ";
        assert!(parse_youtube_single_video_url(dup).is_err());
        assert!(parse_youtube_single_video_url("https://youtu.be/abc").is_err());

        let valid = parse_youtube_single_video_url("https://youtu.be/8nsV1giVLOQ?si=TKLZ1x2UyRnoeIrD")
            .expect("valid short youtube link");
        assert_eq!(valid.video_id, "8nsV1giVLOQ");
    }
}
