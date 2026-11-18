import {
  SeparatedStemsMap,
  SeparationQualityMode,
  StemId,
  StereoStemBuffer,
} from '../types/studio';

export interface SeparationProgressInfo {
  stage: 1 | 2 | 3;
  percent: number;
  message: string;
}

/**
 * In-place iterative Radix-2 Cooley-Tukey FFT / IFFT.
 * `inverse = false` computes forward DFT; `inverse = true` computes unscaled IDFT divided by N.
 */
function radix2Fft(re: Float32Array, im: Float32Array, inverse: boolean): void {
  const n = re.length;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wLenRe = Math.cos(angle);
    const wLenIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let k = 0; k < halfLen; k++) {
        const uIdx = i + k;
        const vIdx = uIdx + halfLen;
        const tRe = re[vIdx] * wRe - im[vIdx] * wIm;
        const tIm = re[vIdx] * wIm + im[vIdx] * wRe;

        re[vIdx] = re[uIdx] - tRe;
        im[vIdx] = im[uIdx] - tIm;
        re[uIdx] += tRe;
        im[uIdx] += tIm;

        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }

  if (inverse) {
    const invN = 1 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= invN;
      im[i] *= invN;
    }
  }
}

function createHannWindow(size: number): Float32Array {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }
  return win;
}

function yieldToUiFrame(): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, 0);
  });
}

function extractWaveformPeaksFromStereo(
  left: Float32Array,
  right: Float32Array,
  numBars = 140
): number[] {
  const total = Math.min(left.length, right.length);
  if (total === 0) return new Array(numBars).fill(0.08);
  const blockSize = Math.max(1, Math.floor(total / numBars));
  const peaks: number[] = [];

  for (let i = 0; i < numBars; i++) {
    const start = i * blockSize;
    const end = Math.min(total, start + blockSize);
    let maxVal = 0;
    for (let j = start; j < end; j++) {
      const m = 0.5 * (Math.abs(left[j]) + Math.abs(right[j]));
      if (m > maxVal) maxVal = m;
    }
    peaks.push(Math.max(0.05, Math.min(0.98, Number(maxVal.toFixed(3)))));
  }
  return peaks;
}

function subtractStereo(
  minuend: StereoStemBuffer,
  subtrahend: StereoStemBuffer
): StereoStemBuffer {
  const len = Math.min(minuend.left.length, subtrahend.left.length);
  const left = new Float32Array(len);
  const right = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    left[i] = minuend.left[i] - subtrahend.left[i];
    right[i] = minuend.right[i] - subtrahend.right[i];
  }
  return { sampleRate: minuend.sampleRate, left, right };
}

/**
 * Applies a frame-by-frame STFT spectral mask with Hann synthesis + COLA Overlap-Add.
 */
async function runStftSpectralMaskPass(
  input: StereoStemBuffer,
  fftSize: number,
  hopSize: number,
  yieldEveryFrames: number,
  computeBinMasks: (
    reL: Float32Array,
    imL: Float32Array,
    reR: Float32Array,
    imR: Float32Array,
    frameIdx: number,
    numBins: number,
    sampleRate: number,
    outMaskL: Float32Array,
    outMaskR: Float32Array
  ) => void,
  onProgress: (ratio: number) => void
): Promise<StereoStemBuffer> {
  const len = input.left.length;
  const pad = fftSize >> 1;
  const paddedLen = len + 2 * pad;

  const win = createHannWindow(fftSize);
  const accL = new Float32Array(paddedLen);
  const accR = new Float32Array(paddedLen);
  const winNorm = new Float32Array(paddedLen);

  const reL = new Float32Array(fftSize);
  const imL = new Float32Array(fftSize);
  const reR = new Float32Array(fftSize);
  const imR = new Float32Array(fftSize);

  const halfBins = (fftSize >> 1) + 1;
  const maskL = new Float32Array(halfBins);
  const maskR = new Float32Array(halfBins);

  const totalFrames = Math.max(1, Math.floor((paddedLen - fftSize) / hopSize) + 1);

  for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
    const start = frameIdx * hopSize;

    for (let i = 0; i < fftSize; i++) {
      const srcIdx = start + i - pad;
      const sL = srcIdx >= 0 && srcIdx < len ? input.left[srcIdx] : 0;
      const sR = srcIdx >= 0 && srcIdx < len ? input.right[srcIdx] : 0;
      const w = win[i];
      reL[i] = sL * w;
      imL[i] = 0;
      reR[i] = sR * w;
      imR[i] = 0;
    }

    radix2Fft(reL, imL, false);
    radix2Fft(reR, imR, false);

    computeBinMasks(
      reL,
      imL,
      reR,
      imR,
      frameIdx,
      halfBins,
      input.sampleRate,
      maskL,
      maskR
    );

    // Apply Hermitian-symmetric mask across positive and negative frequency bins
    reL[0] *= maskL[0];
    imL[0] *= maskL[0];
    reR[0] *= maskR[0];
    imR[0] *= maskR[0];

    const nyquist = fftSize >> 1;
    reL[nyquist] *= maskL[nyquist];
    imL[nyquist] *= maskL[nyquist];
    reR[nyquist] *= maskR[nyquist];
    imR[nyquist] *= maskR[nyquist];

    for (let k = 1; k < nyquist; k++) {
      const mL = maskL[k];
      const mR = maskR[k];
      const conjIdx = fftSize - k;

      reL[k] *= mL;
      imL[k] *= mL;
      reL[conjIdx] *= mL;
      imL[conjIdx] *= mL;

      reR[k] *= mR;
      imR[k] *= mR;
      reR[conjIdx] *= mR;
      imR[conjIdx] *= mR;
    }

    radix2Fft(reL, imL, true);
    radix2Fft(reR, imR, true);

    for (let i = 0; i < fftSize; i++) {
      const w = win[i];
      accL[start + i] += reL[i] * w;
      accR[start + i] += reR[i] * w;
      winNorm[start + i] += w * w;
    }

    if (frameIdx % yieldEveryFrames === 0) {
      onProgress(frameIdx / totalFrames);
      await yieldToUiFrame();
    }
  }

  const outL = new Float32Array(len);
  const outR = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const norm = Math.max(1e-5, winNorm[i + pad]);
    outL[i] = accL[i + pad] / norm;
    outR[i] = accR[i + pad] / norm;
  }

  onProgress(1);
  return { sampleRate: input.sampleRate, left: outL, right: outR };
}

/**
 * Executes a real 3-Stage Spectral STFT Separation Pipeline on a decoded AudioBuffer:
 * - Stage 1: Instrumental vs. All Vocals (Center-Pan Harmonic Wiener Masking + Phase-Exact Subtraction)
 * - Stage 2: Lead Vocal vs. Backing Vocal / Harmony
 * - Stage 3: Dry Lead Vocal vs. Late Reverb Tail (Temporal Spectral Decay Floor)
 */
export async function separateAudioBuffer3Stage(
  audioBuffer: AudioBuffer,
  qualityMode: SeparationQualityMode,
  onProgress: (info: SeparationProgressInfo) => void
): Promise<{
  stems: SeparatedStemsMap;
  peaks: Record<StemId, number[]>;
}> {
  const sampleRate = audioBuffer.sampleRate;
  const rawLeft = new Float32Array(audioBuffer.getChannelData(0));
  const rawRight =
    audioBuffer.numberOfChannels > 1
      ? new Float32Array(audioBuffer.getChannelData(1))
      : new Float32Array(rawLeft);

  const original: StereoStemBuffer = {
    sampleRate,
    left: rawLeft,
    right: rawRight,
  };

  const fftSize = 2048;
  const hopSize = 512;
  const halfBins = (fftSize >> 1) + 1;
  const isPro = qualityMode === 'pro_roformer';
  const yieldStep = isPro ? 220 : 320;

  // Stage 1: Extract All Vocals (Lead + Backing) and subtract from Original to get Instrumental
  const stage1ModelName = isPro ? 'Chế độ Siêu Nét' : 'Chế độ Nhanh';
  const prevStage1Mask = new Float32Array(halfBins);

  const allVocals = await runStftSpectralMaskPass(
    original,
    fftSize,
    hopSize,
    yieldStep,
    (reL, imL, reR, imR, _frame, numBins, sr, outL, outR) => {
      const binHz = sr / fftSize;
      for (let k = 0; k < numBins; k++) {
        const freq = k * binHz;
        if (freq < 95 || freq > 10800) {
          outL[k] = 0;
          outR[k] = 0;
          continue;
        }

        // Vocal presence bandpass envelope (peaks in 220 Hz .. 6.5 kHz)
        let bandWeight = 1.0;
        if (freq < 220) {
          bandWeight = (freq - 95) / (220 - 95);
        } else if (freq > 6500) {
          bandWeight = Math.max(0, 1 - (freq - 6500) / (10800 - 6500));
        }

        const midRe = 0.5 * (reL[k] + reR[k]);
        const midIm = 0.5 * (imL[k] + imR[k]);
        const sideRe = 0.5 * (reL[k] - reR[k]);
        const sideIm = 0.5 * (imL[k] - imR[k]);

        const midPow = midRe * midRe + midIm * midIm;
        const sidePow = sideRe * sideRe + sideIm * sideIm;
        const totalPow = midPow + sidePow + 1e-12;

        // Stereo correlation factor: 1.0 for centered vocals, < 0.4 for wide panned instruments
        const centerRatio = midPow / totalPow;
        const sharpness = isPro ? 2.4 : 1.9;
        const rawMask =
          Math.pow(Math.max(0, (centerRatio - 0.48) / 0.52), sharpness) * bandWeight;

        // Smooth across consecutive frames to prevent musical noise
        const alpha = isPro ? 0.55 : 0.45;
        const smoothed = alpha * rawMask + (1 - alpha) * prevStage1Mask[k];
        prevStage1Mask[k] = smoothed;

        const clamped = Math.max(0, Math.min(1, smoothed));
        outL[k] = clamped;
        outR[k] = clamped;
      }
    },
    (ratio) => {
      onProgress({
        stage: 1,
        percent: Math.round(ratio * 38),
        message: `Tầng 1/3: Đang tách Nhạc Nền & Tổng Vocal (${stage1ModelName})...`,
      });
    }
  );

  const instrumental = subtractStereo(original, allVocals);

  // Stage 2: Extract Lead Vocal (focused centered formant ridges) vs Backing Vocal / Harmony
  const stage2ModelName = isPro ? 'Chế độ Siêu Nét' : 'Chế độ Nhanh';
  const prevStage2Mask = new Float32Array(halfBins);
  const tempMaskBuf = new Float32Array(halfBins);

  const leadWithReverb = await runStftSpectralMaskPass(
    allVocals,
    fftSize,
    hopSize,
    yieldStep,
    (reL, imL, reR, imR, _frame, numBins, sr, outL, outR) => {
      const binHz = sr / fftSize;
      for (let k = 0; k < numBins; k++) {
        const freq = k * binHz;
        const midRe = 0.5 * (reL[k] + reR[k]);
        const midIm = 0.5 * (imL[k] + imR[k]);
        const sideRe = 0.5 * (reL[k] - reR[k]);
        const sideIm = 0.5 * (imL[k] - imR[k]);

        const midPow = midRe * midRe + midIm * midIm;
        const sidePow = sideRe * sideRe + sideIm * sideIm;
        const centerFocus = midPow / (midPow + sidePow + 1e-12);

        // Smooth S-curve transition eliminates hard threshold musical-noise chatter
        const leadBand = freq >= 150 && freq <= 5800 ? 1.0 : 0.72;
        const lo = isPro ? 0.62 : 0.56;
        const hi = isPro ? 0.90 : 0.86;
        const u = Math.max(0, Math.min(1, (centerFocus - lo) / (hi - lo)));
        const sCurve = u * u * (3 - 2 * u);
        tempMaskBuf[k] = (0.06 + 0.94 * sCurve) * leadBand;
      }

      for (let k = 0; k < numBins; k++) {
        const k0 = k > 0 ? k - 1 : 0;
        const k2 = k + 1 < numBins ? k + 1 : numBins - 1;
        const freqSmooth = 0.25 * tempMaskBuf[k0] + 0.5 * tempMaskBuf[k] + 0.25 * tempMaskBuf[k2];
        const prev = prevStage2Mask[k];
        const alpha = freqSmooth > prev ? 0.65 : 0.32;
        const smoothed = alpha * freqSmooth + (1 - alpha) * prev;
        prevStage2Mask[k] = smoothed;
        outL[k] = smoothed;
        outR[k] = smoothed;
      }
    },
    (ratio) => {
      onProgress({
        stage: 2,
        percent: 38 + Math.round(ratio * 34),
        message: `Tầng 2/3: Đang tách Vocal Chính & Giọng Bè (${stage2ModelName})...`,
      });
    }
  );

  const backingVocal = subtractStereo(allVocals, leadWithReverb);

  // Stage 3: De-Reverb (separate Dry Lead Vocal from Late Reverb Tail)
  const decayTailEstimate = new Float32Array(halfBins);
  const dryLeadVocal = await runStftSpectralMaskPass(
    leadWithReverb,
    fftSize,
    hopSize,
    yieldStep,
    (reL, imL, reR, imR, _frame, numBins, _sr, outL, outR) => {
      for (let k = 0; k < numBins; k++) {
        const powL = reL[k] * reL[k] + imL[k] * imL[k];
        const powR = reR[k] * reR[k] + imR[k] * imR[k];
        const energy = 0.5 * (powL + powR);

        // Exponential decay model of late room reverberation (~180 ms decay time)
        const decayedPrev = decayTailEstimate[k] * 0.82;
        const tailEnergy = Math.min(energy * 0.85, decayedPrev);
        decayTailEstimate[k] = Math.max(decayedPrev, energy * 0.38);

        // Spectral subtraction / Wiener de-reverb gain
        const dryGain = Math.max(0.12, Math.min(1, (energy - tailEnergy * 0.9) / (energy + 1e-12)));
        outL[k] = dryGain;
        outR[k] = dryGain;
      }
    },
    (ratio) => {
      onProgress({
        stage: 3,
        percent: 72 + Math.round(ratio * 28),
        message: 'Tầng 3/3: Đang lọc tiếng vang cho Vocal chính...',
      });
    }
  );

  const reverbTail = subtractStereo(leadWithReverb, dryLeadVocal);

  const stems: SeparatedStemsMap = {
    lead_vocal: dryLeadVocal,
    backing_vocal: backingVocal,
    instrumental,
    reverb_tail: reverbTail,
  };

  const peaks: Record<StemId, number[]> = {
    lead_vocal: extractWaveformPeaksFromStereo(dryLeadVocal.left, dryLeadVocal.right),
    backing_vocal: extractWaveformPeaksFromStereo(backingVocal.left, backingVocal.right),
    instrumental: extractWaveformPeaksFromStereo(instrumental.left, instrumental.right),
    reverb_tail: extractWaveformPeaksFromStereo(reverbTail.left, reverbTail.right),
  };

  onProgress({
    stage: 3,
    percent: 100,
    message: 'Hoàn tất tách 4 kênh âm thanh độc lập!',
  });

  return { stems, peaks };
}
