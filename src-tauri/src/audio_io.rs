//! Pure-Rust Audio & Video Container I/O Module (`symphonia` + `rubato` + `hound` + Pure-Rust FLAC).
//! Decodes MP3, WAV, FLAC, OGG, M4A, MP4, MKV, WEBM, and MOV into stereo 44.1 kHz f32 buffers
//! and exports WAV (16/24-bit PCM, 32-bit Float) and Lossless FLAC without external `ffmpeg.exe`.

use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, Write};
use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::{Deserialize, Serialize};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const STUDIO_SAMPLE_RATE: u32 = 44_100;

/// Supported audio and video container extensions for demuxing audio tracks.
pub const SUPPORTED_AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus"];
pub const SUPPORTED_VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "webm", "mov", "m4v", "avi"];

/// Returns true if the file path has a video container extension (`mp4`, `mkv`, `webm`, `mov`, `avi`).
pub fn is_video_container_path<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| SUPPORTED_VIDEO_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Represents a decoded stereo audio buffer normalized to [-1.0, 1.0] at 44.1 kHz.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StereoAudioBuffer {
    pub sample_rate: u32,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl StereoAudioBuffer {
    pub fn new(sample_rate: u32, left: Vec<f32>, right: Vec<f32>) -> Self {
        Self {
            sample_rate,
            left,
            right,
        }
    }

    pub fn num_frames(&self) -> usize {
        self.left.len().min(self.right.len())
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Computes normalized waveform peak bars (0.05..=0.98) for 60fps UI rendering.
    pub fn compute_waveform_peaks(&self, num_bars: usize) -> Vec<f32> {
        let total_frames = self.num_frames();
        if total_frames == 0 || num_bars == 0 {
            return vec![0.1; num_bars];
        }

        let block_size = (total_frames / num_bars).max(1);
        let mut peaks = Vec::with_capacity(num_bars);

        for bar_idx in 0..num_bars {
            let start = bar_idx * block_size;
            let end = ((bar_idx + 1) * block_size).min(total_frames);
            let mut max_peak = 0.0_f32;

            for i in start..end {
                let mono_abs = 0.5 * (self.left[i].abs() + self.right[i].abs());
                if mono_abs > max_peak {
                    max_peak = mono_abs;
                }
            }
            peaks.push(max_peak.clamp(0.05, 0.98));
        }

        peaks
    }
}

/// Metadata and waveform summary returned to the UI upon loading a track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedTrackSummary {
    pub file_name: String,
    pub is_video_source: bool,
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_sec: f64,
    pub detected_key: String,
    pub detected_bpm: f32,
    pub waveform_peaks: Vec<f32>,
}

/// Decodes an audio or video file from disk using Pure-Rust `symphonia` and resamples to 44.1 kHz stereo.
pub fn decode_audio_file<P: AsRef<Path>>(path: P) -> Result<StereoAudioBuffer, String> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)
        .map_err(|e| format!("Failed to open audio/video file '{}': {}", path_ref.display(), e))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path_ref.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Unsupported or corrupted container format: {}", e))?;

    let mut format = probed.format;
    let codecs = symphonia::default::get_codecs();
    let mut selected = None;
    for t in format.tracks() {
        if t.codec_params.codec == CODEC_TYPE_NULL {
            continue;
        }
        if let Ok(dec) = codecs.make(&t.codec_params, &DecoderOptions::default()) {
            selected = Some((
                t.id,
                t.codec_params.sample_rate.unwrap_or(STUDIO_SAMPLE_RATE),
                dec,
            ));
            break;
        }
    }

    let (track_id, src_sample_rate, mut decoder) = selected.ok_or_else(|| {
        "Failed to initialize codec decoder: unsupported feature: core (codec):unsupported codec"
            .to_string()
    })?;

    let mut left_channel = Vec::new();
    let mut right_channel = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(err) => return Err(format!("Decode error: {}", err)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| format!("Packet decode failure: {}", e))?;

        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let num_channels = spec.channels.count().max(1);

        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
        }

        if let Some(buf) = &mut sample_buf {
            buf.copy_interleaved_ref(decoded);
            let samples = buf.samples();

            for frame in samples.chunks(num_channels) {
                let l = frame[0];
                let r = if num_channels > 1 { frame[1] } else { l };
                left_channel.push(l);
                right_channel.push(r);
            }
        }
    }

    let raw_buffer = StereoAudioBuffer::new(src_sample_rate, left_channel, right_channel);
    if src_sample_rate != STUDIO_SAMPLE_RATE {
        resample_stereo_buffer(&raw_buffer, STUDIO_SAMPLE_RATE)
    } else {
        Ok(raw_buffer)
    }
}

/// High-quality Sinc resampling to convert arbitrary input sample rates (e.g. 48 kHz) to 44.1 kHz.
pub fn resample_stereo_buffer(
    input: &StereoAudioBuffer,
    target_rate: u32,
) -> Result<StereoAudioBuffer, String> {
    if input.sample_rate == target_rate || input.num_frames() == 0 {
        return Ok(input.clone());
    }

    let ratio = target_rate as f64 / input.sample_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.92,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024usize;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 2)
        .map_err(|e| format!("Resampler init error: {}", e))?;

    let total_frames = input.num_frames();
    let estimated_out = ((total_frames as f64) * ratio) as usize + chunk_size;
    let mut out_left = Vec::with_capacity(estimated_out);
    let mut out_right = Vec::with_capacity(estimated_out);

    let mut pos = 0;
    while pos + chunk_size <= total_frames {
        let waves_in = vec![
            input.left[pos..pos + chunk_size].to_vec(),
            input.right[pos..pos + chunk_size].to_vec(),
        ];
        let waves_out = resampler
            .process(&waves_in, None)
            .map_err(|e| format!("Resample chunk error: {}", e))?;
        out_left.extend_from_slice(&waves_out[0]);
        out_right.extend_from_slice(&waves_out[1]);
        pos += chunk_size;
    }

    if pos < total_frames {
        let remain = total_frames - pos;
        let mut tail_l = vec![0.0_f32; chunk_size];
        let mut tail_r = vec![0.0_f32; chunk_size];
        tail_l[..remain].copy_from_slice(&input.left[pos..]);
        tail_r[..remain].copy_from_slice(&input.right[pos..]);

        let waves_out = resampler
            .process(&[tail_l, tail_r], None)
            .map_err(|e| format!("Resample tail error: {}", e))?;
        let expected_tail_out = ((remain as f64) * ratio).round() as usize;
        let take_len = expected_tail_out.min(waves_out[0].len());
        out_left.extend_from_slice(&waves_out[0][..take_len]);
        out_right.extend_from_slice(&waves_out[1][..take_len]);
    }

    Ok(StereoAudioBuffer::new(target_rate, out_left, out_right))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioExportFormat {
    Wav16Bit,
    Wav24Bit,
    Wav32Float,
    Flac24BitLossless,
}

/// Exports a processed stereo buffer in WAV (16/24-bit PCM, 32-bit Float) or Lossless 24-bit FLAC.
pub fn export_audio_file<P: AsRef<Path>>(
    path: P,
    buffer: &StereoAudioBuffer,
    format: AudioExportFormat,
) -> Result<(), String> {
    match format {
        AudioExportFormat::Wav16Bit => export_wav_internal(path, buffer, 16, SampleFormat::Int),
        AudioExportFormat::Wav24Bit => export_wav_internal(path, buffer, 24, SampleFormat::Int),
        AudioExportFormat::Wav32Float => export_wav_internal(path, buffer, 32, SampleFormat::Float),
        AudioExportFormat::Flac24BitLossless => export_flac_24bit_file(path, buffer),
    }
}

/// Encodes a stereo buffer into in-memory 16-bit PCM WAV bytes for low-overhead IPC preview transfer.
pub fn encode_wav_16bit_bytes(buffer: &StereoAudioBuffer) -> Result<Vec<u8>, String> {
    let frames = buffer.num_frames();
    let mut bytes = Vec::with_capacity(44 + frames * 4);
    let spec = WavSpec {
        channels: 2,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    {
        let mut writer = WavWriter::new(Cursor::new(&mut bytes), spec)
            .map_err(|e| format!("Failed to init WAV buffer: {}", e))?;
        write_wav_samples(&mut writer, buffer, 16, SampleFormat::Int)?;
        writer
            .finalize()
            .map_err(|e| format!("Failed to finalize WAV buffer: {}", e))?;
    }
    Ok(bytes)
}

fn export_wav_internal<P: AsRef<Path>>(
    path: P,
    buffer: &StereoAudioBuffer,
    bits_per_sample: u16,
    sample_format: SampleFormat,
) -> Result<(), String> {
    let spec = WavSpec {
        channels: 2,
        sample_rate: buffer.sample_rate,
        bits_per_sample,
        sample_format,
    };

    let mut writer = WavWriter::create(path.as_ref(), spec)
        .map_err(|e| format!("Failed to create WAV file: {}", e))?;

    write_wav_samples(&mut writer, buffer, bits_per_sample, sample_format)?;

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV file: {}", e))
}

fn write_wav_samples<W: Write + Seek>(
    writer: &mut WavWriter<W>,
    buffer: &StereoAudioBuffer,
    bits_per_sample: u16,
    sample_format: SampleFormat,
) -> Result<(), String> {
    let frames = buffer.num_frames();
    match (bits_per_sample, sample_format) {
        (16, SampleFormat::Int) => {
            for i in 0..frames {
                let l = (buffer.left[i].clamp(-1.0, 1.0) * 32_767.0) as i16;
                let r = (buffer.right[i].clamp(-1.0, 1.0) * 32_767.0) as i16;
                writer.write_sample(l).map_err(|e| e.to_string())?;
                writer.write_sample(r).map_err(|e| e.to_string())?;
            }
        }
        (24, SampleFormat::Int) => {
            let max_i24 = 8_388_607.0_f32;
            for i in 0..frames {
                let l = (buffer.left[i].clamp(-1.0, 1.0) * max_i24) as i32;
                let r = (buffer.right[i].clamp(-1.0, 1.0) * max_i24) as i32;
                writer.write_sample(l).map_err(|e| e.to_string())?;
                writer.write_sample(r).map_err(|e| e.to_string())?;
            }
        }
        _ => {
            for i in 0..frames {
                writer
                    .write_sample(buffer.left[i].clamp(-1.0, 1.0))
                    .map_err(|e| e.to_string())?;
                writer
                    .write_sample(buffer.right[i].clamp(-1.0, 1.0))
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

/// Encodes and writes a valid 24-bit Stereo Lossless FLAC file (`fLaC` + STREAMINFO + VERBATIM frames)
/// in Pure Rust without external binaries.
pub fn export_flac_24bit_file<P: AsRef<Path>>(
    path: P,
    buffer: &StereoAudioBuffer,
) -> Result<(), String> {
    let bytes = encode_flac_24bit_bytes(buffer);
    let file = File::create(path.as_ref())
        .map_err(|e| format!("Failed to create FLAC file '{}': {}", path.as_ref().display(), e))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(&bytes)
        .map_err(|e| format!("Failed to write FLAC data: {}", e))?;
    writer
        .flush()
        .map_err(|e| format!("Failed to flush FLAC file: {}", e))
}

/// Produces a complete, spec-compliant 24-bit Stereo FLAC byte stream.
pub fn encode_flac_24bit_bytes(buffer: &StereoAudioBuffer) -> Vec<u8> {
    let total_frames = buffer.num_frames();
    let max_block_size: usize = 4096;
    let min_block_size: u16 = total_frames.clamp(16, max_block_size) as u16;
    let sample_rate = buffer.sample_rate.clamp(1, 655_350);

    let mut out = Vec::with_capacity(42 + total_frames * 6 + 1024);

    // 1. Magic marker "fLaC"
    out.extend_from_slice(b"fLaC");

    // 2. Last metadata block (STREAMINFO, type 0, length 34 bytes)
    out.push(0x80);
    out.extend_from_slice(&[0x00, 0x00, 0x22]);

    out.extend_from_slice(&min_block_size.to_be_bytes());
    out.extend_from_slice(&(max_block_size as u16).to_be_bytes());
    out.extend_from_slice(&[0x00, 0x00, 0x00]); // min_frame_size unknown
    out.extend_from_slice(&[0x00, 0x00, 0x00]); // max_frame_size unknown

    // 20 bits sample rate | 3 bits (channels - 1) | 5 bits (bps - 1) | 36 bits total_samples
    let packed: u64 = ((sample_rate as u64) << 44)
        | (1_u64 << 41) // 2 channels -> 2 - 1 = 1
        | (23_u64 << 36) // 24-bit -> 24 - 1 = 23
        | ((total_frames as u64) & 0x0F_FFFF_FFFF);
    out.extend_from_slice(&packed.to_be_bytes());
    out.extend_from_slice(&[0_u8; 16]); // Uncomputed MD5 allowed by FLAC spec

    // 3. Encode Verbatim Stereo Audio Frames (variable-blocksize strategy = 1)
    let mut start = 0usize;
    let max_i24 = 8_388_607.0_f32;

    while start < total_frames {
        let block_len = (total_frames - start).min(max_block_size);
        let frame_start_offset = out.len();

        // Sync code 0x3FFE + reserved 0 + variable-blocksize bit 1 => 0xFF, 0xF9
        out.push(0xFF);
        out.push(0xF9);
        // Block size 0b0111 (16-bit (n-1) at end of header) | Sample rate 0b0000 (from STREAMINFO)
        out.push(0x70);
        // Channel assignment 0b0001 (independent L/R) | Sample size 0b110 (24-bit) | reserved 0
        out.push(0x1C);

        // UTF-8 coded starting sample index
        encode_utf8_uint64(start as u64, &mut out);

        // 16-bit (block_size - 1) big-endian
        out.extend_from_slice(&((block_len - 1) as u16).to_be_bytes());

        // Header CRC-8
        let crc8 = flac_crc8(&out[frame_start_offset..]);
        out.push(crc8);

        // Left Subframe: VERBATIM (0x02) + raw 24-bit big-endian signed samples
        out.push(0x02);
        for i in start..(start + block_len) {
            let sample = (buffer.left[i].clamp(-1.0, 1.0) * max_i24).round() as i32;
            let bytes = sample.to_be_bytes();
            out.extend_from_slice(&bytes[1..4]);
        }

        // Right Subframe: VERBATIM (0x02) + raw 24-bit big-endian signed samples
        out.push(0x02);
        for i in start..(start + block_len) {
            let sample = (buffer.right[i].clamp(-1.0, 1.0) * max_i24).round() as i32;
            let bytes = sample.to_be_bytes();
            out.extend_from_slice(&bytes[1..4]);
        }

        // Frame CRC-16
        let crc16 = flac_crc16(&out[frame_start_offset..]);
        out.extend_from_slice(&crc16.to_be_bytes());

        start += block_len;
    }

    out
}

fn encode_utf8_uint64(val: u64, out: &mut Vec<u8>) {
    if val < 0x80 {
        out.push(val as u8);
    } else if val < 0x800 {
        out.push(0xC0 | ((val >> 6) as u8));
        out.push(0x80 | ((val & 0x3F) as u8));
    } else if val < 0x1_0000 {
        out.push(0xE0 | ((val >> 12) as u8));
        out.push(0x80 | (((val >> 6) & 0x3F) as u8));
        out.push(0x80 | ((val & 0x3F) as u8));
    } else if val < 0x20_0000 {
        out.push(0xF0 | ((val >> 18) as u8));
        out.push(0x80 | (((val >> 12) & 0x3F) as u8));
        out.push(0x80 | (((val >> 6) & 0x3F) as u8));
        out.push(0x80 | ((val & 0x3F) as u8));
    } else {
        out.push(0xF8 | ((val >> 24) as u8));
        out.push(0x80 | (((val >> 18) & 0x3F) as u8));
        out.push(0x80 | (((val >> 12) & 0x3F) as u8));
        out.push(0x80 | (((val >> 6) & 0x3F) as u8));
        out.push(0x80 | ((val & 0x3F) as u8));
    }
}

fn flac_crc8(data: &[u8]) -> u8 {
    let mut crc = 0_u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

fn flac_crc16(data: &[u8]) -> u16 {
    let mut crc = 0_u16;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_24bit_stream_encoding_structure() {
        let frames = 512usize;
        let left: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let right: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.05).cos() * 0.5).collect();
        let buf = StereoAudioBuffer::new(44_100, left, right);

        let flac_bytes = encode_flac_24bit_bytes(&buf);
        assert!(flac_bytes.starts_with(b"fLaC"));
        assert_eq!(flac_bytes[4], 0x80); // Last metadata block + STREAMINFO
        assert_eq!(&flac_bytes[5..8], &[0x00, 0x00, 0x22]); // 34 bytes STREAMINFO
        assert_eq!(&flac_bytes[42..44], &[0xFF, 0xF9]); // FLAC audio frame sync + variable block
        assert!(flac_bytes.len() > 42 + frames * 6);
    }
}
