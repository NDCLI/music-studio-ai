/**
 * Stage 2, 3 & 6 Real-Time DSP & Multi-Format Audio Encoder Utilities:
 * 1. Krumhansl-Schmuckler Chromagram Musical Key Detector
 * 2. Energy Flux Autocorrelation BPM Estimator
 * 3. Multi-Format WAV (16-bit, 24-bit PCM, 32-bit Float) & 24-bit Lossless FLAC Encoders
 */

const NOTE_NAMES = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
] as const;

const MAJOR_PROFILE = [
  6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const MINOR_PROFILE = [
  6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

export type BrowserAudioExportFormat = 'wav16' | 'wav24' | 'wav32f' | 'flac';

/**
 * Detects the musical key (e.g. "Am", "C", "Em", "G") of an AudioBuffer
 * using Goertzel harmonic pitch-class profiling and Krumhansl-Schmuckler correlation.
 */
export function detectMusicalKeyFromBuffer(buffer: AudioBuffer): string {
  const data = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  if (data.length < 4096) return 'Am';

  const chroma = new Float32Array(12);
  const windowSize = 2048;
  const numSlices = 18;
  const sliceStep = Math.max(windowSize, Math.floor(data.length / numSlices));

  for (let s = 0; s < numSlices; s++) {
    const start = s * sliceStep;
    if (start + windowSize > data.length) break;

    for (let octave = 2; octave <= 5; octave++) {
      for (let pc = 0; pc < 12; pc++) {
        const midi = (octave + 1) * 12 + pc;
        const freq = 440 * Math.pow(2, (midi - 69) / 12);
        const k = Math.round((windowSize * freq) / sampleRate);
        const omega = (2 * Math.PI * k) / windowSize;
        const coeff = 2 * Math.cos(omega);

        let q1 = 0;
        let q2 = 0;
        for (let i = 0; i < windowSize; i++) {
          const q0 = data[start + i] + coeff * q1 - q2;
          q2 = q1;
          q1 = q0;
        }
        const magSq = q1 * q1 + q2 * q2 - q1 * q2 * coeff;
        chroma[pc] += Math.sqrt(Math.max(0, magSq));
      }
    }
  }

  let bestScore = -Infinity;
  let bestKey = 'Am';

  for (let root = 0; root < 12; root++) {
    let majScore = 0;
    let minScore = 0;
    for (let i = 0; i < 12; i++) {
      const val = chroma[(root + i) % 12];
      majScore += val * MAJOR_PROFILE[i];
      minScore += val * MINOR_PROFILE[i];
    }
    if (majScore > bestScore) {
      bestScore = majScore;
      bestKey = NOTE_NAMES[root];
    }
    if (minScore > bestScore) {
      bestScore = minScore;
      bestKey = `${NOTE_NAMES[root]}m`;
    }
  }

  return bestKey;
}

/**
 * Estimates the tempo (BPM) of an AudioBuffer via onset envelope autocorrelation.
 */
export function detectBpmFromBuffer(buffer: AudioBuffer): number {
  const data = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const hop = 512;
  const maxSamples = Math.min(data.length, sampleRate * 25);

  if (maxSamples < hop * 40) return 120;

  const numFrames = Math.floor(maxSamples / hop);
  const onsetEnv = new Float32Array(numFrames);
  let prevEnergy = 0;

  for (let f = 0; f < numFrames; f++) {
    let energy = 0;
    const base = f * hop;
    for (let i = 0; i < hop; i++) {
      const v = data[base + i];
      energy += v * v;
    }
    const diff = Math.max(0, energy - prevEnergy);
    onsetEnv[f] = Math.sqrt(diff);
    prevEnergy = energy;
  }

  const envRate = sampleRate / hop;
  const minLag = Math.round((envRate * 60) / 172);
  const maxLag = Math.round((envRate * 60) / 70);

  let bestLag = minLag;
  let bestCorr = -1;

  for (let lag = minLag; lag <= Math.min(maxLag, Math.floor(numFrames / 2)); lag++) {
    let corr = 0;
    for (let i = 0; i < numFrames - lag; i++) {
      corr += onsetEnv[i] * onsetEnv[i + lag];
    }
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }

  const rawBpm = (60 * envRate) / bestLag;
  return Math.round(rawBpm * 2) / 2;
}

function softLimitSample(s: number): number {
  const ceiling = 0.966; // -0.3 dBFS
  const abs = Math.abs(s);
  const limited =
    abs > ceiling
      ? Math.sign(s) *
        Math.min(0.995, ceiling + (abs - ceiling) / (1 + (abs - ceiling) * 4))
      : s;
  return Math.max(-1, Math.min(1, limited));
}

/**
 * Encodes stereo Float32 arrays into WAV (16-bit PCM, 24-bit PCM, or 32-bit IEEE Float) or 24-bit FLAC.
 */
export function encodeStereoAudioBlob(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
  format: BrowserAudioExportFormat = 'wav24'
): Blob {
  if (format === 'flac') {
    return encodeStereoFlac24Blob(left, right, sampleRate);
  }

  const numFrames = Math.min(left.length, right.length);
  const numChannels = 2;
  const isFloat = format === 'wav32f';
  const bitsPerSample = format === 'wav16' ? 16 : format === 'wav24' ? 24 : 32;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataByteLength = numFrames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataByteLength);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataByteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, isFloat ? 3 : 1, true); // 3 = IEEE Float, 1 = PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, 'data');
  view.setUint32(40, dataByteLength, true);

  let offset = 44;
  if (format === 'wav16') {
    for (let i = 0; i < numFrames; i++) {
      view.setInt16(offset, Math.round(softLimitSample(left[i]) * 32767), true);
      view.setInt16(offset + 2, Math.round(softLimitSample(right[i]) * 32767), true);
      offset += 4;
    }
  } else if (format === 'wav24') {
    const maxI24 = 8388607;
    for (let i = 0; i < numFrames; i++) {
      const l = Math.round(softLimitSample(left[i]) * maxI24);
      const r = Math.round(softLimitSample(right[i]) * maxI24);
      view.setUint8(offset, l & 0xff);
      view.setUint8(offset + 1, (l >> 8) & 0xff);
      view.setUint8(offset + 2, (l >> 16) & 0xff);
      view.setUint8(offset + 3, r & 0xff);
      view.setUint8(offset + 4, (r >> 8) & 0xff);
      view.setUint8(offset + 5, (r >> 16) & 0xff);
      offset += 6;
    }
  } else {
    for (let i = 0; i < numFrames; i++) {
      view.setFloat32(offset, softLimitSample(left[i]), true);
      view.setFloat32(offset + 4, softLimitSample(right[i]), true);
      offset += 8;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function encodeUtf8SampleIndex(val: number, bytes: number[]): void {
  if (val < 0x80) {
    bytes.push(val);
  } else if (val < 0x800) {
    bytes.push(0xc0 | (val >> 6));
    bytes.push(0x80 | (val & 0x3f));
  } else if (val < 0x10000) {
    bytes.push(0xe0 | (val >> 12));
    bytes.push(0x80 | ((val >> 6) & 0x3f));
    bytes.push(0x80 | (val & 0x3f));
  } else if (val < 0x200000) {
    bytes.push(0xf0 | (val >> 18));
    bytes.push(0x80 | ((val >> 12) & 0x3f));
    bytes.push(0x80 | ((val >> 6) & 0x3f));
    bytes.push(0x80 | (val & 0x3f));
  } else {
    bytes.push(0xf8 | Math.floor(val / 0x1000000));
    bytes.push(0x80 | ((val >> 18) & 0x3f));
    bytes.push(0x80 | ((val >> 12) & 0x3f));
    bytes.push(0x80 | ((val >> 6) & 0x3f));
    bytes.push(0x80 | (val & 0x3f));
  }
}

function flacCrc8(data: Uint8Array, start: number, end: number): number {
  let crc = 0;
  for (let i = start; i < end; i++) {
    crc ^= data[i];
    for (let bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) !== 0 ? ((crc << 1) ^ 0x07) & 0xff : (crc << 1) & 0xff;
    }
  }
  return crc;
}

function flacCrc16(data: Uint8Array, start: number, end: number): number {
  let crc = 0;
  for (let i = start; i < end; i++) {
    crc ^= data[i] << 8;
    for (let bit = 0; bit < 8; bit++) {
      crc = (crc & 0x8000) !== 0 ? ((crc << 1) ^ 0x8005) & 0xffff : (crc << 1) & 0xffff;
    }
  }
  return crc;
}

/**
 * Encodes stereo Float32 audio into a valid 24-bit Lossless FLAC stream (`fLaC`)
 * using pre-allocated chunk buffers so full-length multi-minute songs encode fast with low RAM.
 */
export function encodeStereoFlac24Blob(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number
): Blob {
  const totalFrames = Math.min(left.length, right.length);
  const maxBlockSize = 4096;
  const minBlockSize = Math.max(16, Math.min(maxBlockSize, totalFrames));
  const chunks: BlobPart[] = [];

  const header = new Uint8Array(42);
  // "fLaC" magic + STREAMINFO header (34 bytes)
  header.set([0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22], 0);
  header[8] = (minBlockSize >> 8) & 0xff;
  header[9] = minBlockSize & 0xff;
  header[10] = (maxBlockSize >> 8) & 0xff;
  header[11] = maxBlockSize & 0xff;

  const sr = Math.max(1, Math.min(655350, sampleRate));
  const hi32 = ((sr << 12) | (1 << 9) | (23 << 4) | Math.floor(totalFrames / 0x100000000)) >>> 0;
  const lo32 = totalFrames >>> 0;
  header[18] = (hi32 >>> 24) & 0xff;
  header[19] = (hi32 >>> 16) & 0xff;
  header[20] = (hi32 >>> 8) & 0xff;
  header[21] = hi32 & 0xff;
  header[22] = (lo32 >>> 24) & 0xff;
  header[23] = (lo32 >>> 16) & 0xff;
  header[24] = (lo32 >>> 8) & 0xff;
  header[25] = lo32 & 0xff;
  chunks.push(header);

  const maxI24 = 8388607;
  let start = 0;

  while (start < totalFrames) {
    const blockLen = Math.min(maxBlockSize, totalFrames - start);
    const utf8SampleBytes: number[] = [];
    encodeUtf8SampleIndex(start, utf8SampleBytes);

    const headerLen = 4 + utf8SampleBytes.length + 2;
    const frameBytes = new Uint8Array(headerLen + 1 + 1 + blockLen * 3 + 1 + blockLen * 3 + 2);
    let pos = 0;

    frameBytes[pos++] = 0xff;
    frameBytes[pos++] = 0xf9;
    frameBytes[pos++] = 0x70;
    frameBytes[pos++] = 0x1c;
    for (let b = 0; b < utf8SampleBytes.length; b++) {
      frameBytes[pos++] = utf8SampleBytes[b];
    }
    const blMinus1 = blockLen - 1;
    frameBytes[pos++] = (blMinus1 >> 8) & 0xff;
    frameBytes[pos++] = blMinus1 & 0xff;
    frameBytes[pos] = flacCrc8(frameBytes, 0, pos);
    pos++;

    // Left verbatim subframe
    frameBytes[pos++] = 0x02;
    for (let i = start; i < start + blockLen; i++) {
      const s = Math.round(softLimitSample(left[i]) * maxI24);
      frameBytes[pos++] = (s >> 16) & 0xff;
      frameBytes[pos++] = (s >> 8) & 0xff;
      frameBytes[pos++] = s & 0xff;
    }

    // Right verbatim subframe
    frameBytes[pos++] = 0x02;
    for (let i = start; i < start + blockLen; i++) {
      const s = Math.round(softLimitSample(right[i]) * maxI24);
      frameBytes[pos++] = (s >> 16) & 0xff;
      frameBytes[pos++] = (s >> 8) & 0xff;
      frameBytes[pos++] = s & 0xff;
    }

    const crc16 = flacCrc16(frameBytes, 0, pos);
    frameBytes[pos++] = (crc16 >> 8) & 0xff;
    frameBytes[pos++] = crc16 & 0xff;

    chunks.push(frameBytes);
    start += blockLen;
  }

  return new Blob(chunks, { type: 'audio/flac' });
}

export function encodeStereoWavBlob(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number
): Blob {
  return encodeStereoAudioBlob(left, right, sampleRate, 'wav16');
}
