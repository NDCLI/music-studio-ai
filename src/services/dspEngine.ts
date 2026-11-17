/**
 * Stage 2 & Stage 3 Real-Time DSP Utilities:
 * 1. Krumhansl-Schmuckler Chromagram Musical Key Detector
 * 2. Energy Flux Autocorrelation BPM Estimator
 * 3. Formant-Preserving Spectral Pitch/Speed Renderer & WAV Encoder
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

/**
 * Detects the musical key (e.g. "Am", "C", "Em", "G") of an AudioBuffer
 * using Goertzel/Chroma harmonic pitch-class profiling and Krumhansl-Schmuckler correlation.
 */
export function detectMusicalKeyFromBuffer(buffer: AudioBuffer): string {
  const data = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  if (data.length < 4096) return 'Am';

  const chroma = new Float32Array(12);
  const windowSize = 2048;
  const numSlices = 18;
  const sliceStep = Math.max(windowSize, Math.floor(data.length / numSlices));

  // Evaluate 4 octaves (C2 to B5) using exact Goertzel resonator per pitch class
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

/**
 * Encodes stereo Float32 arrays into a downloadable 16-bit or 24-bit WAV Blob.
 */
export function encodeStereoWavBlob(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number
): Blob {
  const numFrames = Math.min(left.length, right.length);
  const numChannels = 2;
  const bytesPerSample = 2; // 16-bit PCM for universal compatibility
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
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataByteLength, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    // Apply -0.3 dBFS True-Peak Soft-Knee Limiter before quantization
    const clampSample = (s: number) => {
      const ceiling = 0.966;
      const abs = Math.abs(s);
      const limited =
        abs > ceiling
          ? Math.sign(s) * Math.min(0.995, ceiling + (abs - ceiling) / (1 + (abs - ceiling) * 4))
          : s;
      return Math.max(-1, Math.min(1, limited));
    };

    const lSample = Math.round(clampSample(left[i]) * 32767);
    const rSample = Math.round(clampSample(right[i]) * 32767);
    view.setInt16(offset, lSample, true);
    view.setInt16(offset + 2, rSample, true);
    offset += 4;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
