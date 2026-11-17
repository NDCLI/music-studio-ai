//! Pure-Rust Audio I/O Module (`symphonia` + `rubato` + `hound`).
//! Decodes MP3, WAV, FLAC, OGG, and M4A/AAC into stereo 44.1 kHz f32 buffers
//! without spawning external `ffmpeg.exe` processes (preventing AhnLab scan lag).

use std::fs::File;
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
    pub sample_rate: u32,
    pub channels: usize,
    pub duration_sec: f64,
    pub detected_key: String,
    pub detected_bpm: f32,
    pub waveform_peaks: Vec<f32>,
}

/// Decodes an audio file from disk using Pure-Rust `symphonia` and resamples to 44.1 kHz stereo.
pub fn decode_audio_file<P: AsRef<Path>>(path: P) -> Result<StereoAudioBuffer, String> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)
        .map_err(|e| format!("Failed to open audio file '{}': {}", path_ref.display(), e))?;

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
        .map_err(|e| format!("Unsupported or corrupted audio format: {}", e))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio track found in container".to_string())?;

    let track_id = track.id;
    let src_sample_rate = track.codec_params.sample_rate.unwrap_or(STUDIO_SAMPLE_RATE);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to initialize codec decoder: {}", e))?;

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

    // Handle remaining tail frames with zero padding
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
pub enum WavExportBitDepth {
    Pcm24Bit,
    Float32Bit,
}

/// Exports a processed stereo buffer to disk as 24-bit PCM or 32-bit Float WAV.
pub fn export_wav_file<P: AsRef<Path>>(
    path: P,
    buffer: &StereoAudioBuffer,
    bit_depth: WavExportBitDepth,
) -> Result<(), String> {
    let spec = match bit_depth {
        WavExportBitDepth::Pcm24Bit => WavSpec {
            channels: 2,
            sample_rate: buffer.sample_rate,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        },
        WavExportBitDepth::Float32Bit => WavSpec {
            channels: 2,
            sample_rate: buffer.sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    };

    let mut writer = WavWriter::create(path.as_ref(), spec)
        .map_err(|e| format!("Failed to create WAV file: {}", e))?;

    let frames = buffer.num_frames();
    match bit_depth {
        WavExportBitDepth::Pcm24Bit => {
            let max_i24 = 8_388_607.0_f32;
            for i in 0..frames {
                let l = (buffer.left[i].clamp(-1.0, 1.0) * max_i24) as i32;
                let r = (buffer.right[i].clamp(-1.0, 1.0) * max_i24) as i32;
                writer.write_sample(l).map_err(|e| e.to_string())?;
                writer.write_sample(r).map_err(|e| e.to_string())?;
            }
        }
        WavExportBitDepth::Float32Bit => {
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

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV file: {}", e))
}
