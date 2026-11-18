import {
  PitchSpeedSettings,
  SeparatedStemsMap,
  StemId,
  StemTrackState,
} from '../types/studio';
import {
  BrowserAudioExportFormat,
  detectBpmFromBuffer,
  detectMusicalKeyFromBuffer,
  encodeStereoAudioBlob,
} from './dspEngine';

/**
 * Extracts real waveform peaks from a decoded WebAudio AudioBuffer.
 */
export function extractPeaksFromBuffer(buffer: AudioBuffer, count = 140): number[] {
  const channelData = buffer.getChannelData(0);
  const blockSize = Math.max(1, Math.floor(channelData.length / count));
  const peaks: number[] = [];

  for (let i = 0; i < count; i++) {
    const start = i * blockSize;
    let maxVal = 0;
    for (let j = 0; j < blockSize && start + j < channelData.length; j++) {
      const sample = Math.abs(channelData[start + j]);
      if (sample > maxVal) maxVal = sample;
    }
    peaks.push(Math.max(0.06, Math.min(0.98, maxVal)));
  }

  return peaks;
}

function cubicHermiteAt(arr: Float32Array, pos: number): number {
  const len = arr.length;
  if (len === 0) return 0;
  const clamped = Math.max(0, Math.min(len - 1, pos));
  const i1 = Math.floor(clamped);
  const frac = clamped - i1;
  if (frac < 1e-6) return arr[i1];

  const i0 = i1 > 0 ? i1 - 1 : 0;
  const i2 = i1 + 1 < len ? i1 + 1 : len - 1;
  const i3 = i1 + 2 < len ? i1 + 2 : len - 1;

  const y0 = arr[i0];
  const y1 = arr[i1];
  const y2 = arr[i2];
  const y3 = arr[i3];

  const c0 = y1;
  const c1 = 0.5 * (y2 - y0);
  const c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
  const c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);

  return ((c3 * frac + c2) * frac + c1) * frac + c0;
}

function softLimitStudioSample(sample: number, ceiling: number): number {
  if (!Number.isFinite(sample)) return 0;
  const abs = Math.abs(sample);
  const knee = ceiling * 0.86;
  if (abs <= knee) return sample;
  const span = Math.max(1e-4, ceiling - knee);
  const excess = abs - knee;
  const compressed = knee + span * (excess / (excess + span));
  return Math.sign(sample) * Math.min(ceiling, compressed);
}

/**
 * Finds a local phase-aligned offset around `candidateCenter` that best correlates with
 * the currently sounding reference grain at `refPos`, preventing WSOLA phase cancellation / double-attack splats.
 */
function findBestWsolaOffset(
  monoRef: Float32Array,
  refPos: number,
  candidateCenter: number,
  searchRadius = 320
): number {
  const len = monoRef.length;
  const corrLen = 96;
  const refStart = Math.floor(refPos);
  if (refStart < 0 || refStart + corrLen >= len) return 0;

  const minDelta = -searchRadius;
  const maxDelta = searchRadius;
  let bestDelta = 0;
  let bestScore = -Infinity;

  // Coarse 4-sample stride scan
  for (let delta = minDelta; delta <= maxDelta; delta += 4) {
    const candStart = Math.floor(candidateCenter) + delta;
    if (candStart < 0 || candStart + corrLen >= len) continue;
    let dot = 0;
    let normCand = 1e-8;
    for (let k = 0; k < corrLen; k += 2) {
      const a = monoRef[refStart + k];
      const b = monoRef[candStart + k];
      dot += a * b;
      normCand += b * b;
    }
    const score = dot / Math.sqrt(normCand);
    if (score > bestScore) {
      bestScore = score;
      bestDelta = delta;
    }
  }

  // Fine 1-sample refinement around coarse peak
  const fineLo = Math.max(minDelta, bestDelta - 3);
  const fineHi = Math.min(maxDelta, bestDelta + 3);
  for (let delta = fineLo; delta <= fineHi; delta += 1) {
    const candStart = Math.floor(candidateCenter) + delta;
    if (candStart < 0 || candStart + corrLen >= len) continue;
    let dot = 0;
    let normCand = 1e-8;
    for (let k = 0; k < corrLen; k += 2) {
      const a = monoRef[refStart + k];
      const b = monoRef[candStart + k];
      dot += a * b;
      normCand += b * b;
    }
    const score = dot / Math.sqrt(normCand);
    if (score > bestScore) {
      bestScore = score;
      bestDelta = delta;
    }
  }

  return bestDelta;
}

/**
 * Real-time Multi-Stem & Phase-Locked WSOLA Pitch/Time Engine.
 * Operates exclusively on real decoded audio/video tracks and separated stems
 * with full-duration export (WAV 16/24-bit, WAV 32-bit Float, FLAC 24-bit).
 */
export class StudioPreviewAudioEngine {
  private ctx: AudioContext | null = null;
  private isPlaying = false;
  private customBuffer: AudioBuffer | null = null;
  private separatedStems: SeparatedStemsMap | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private uiRafId: number | null = null;
  private lastUiFlushMs = 0;
  private pendingDeltaSec = 0;
  private latestPeakDb = -60;
  private playheadSample = 0;
  private vocalPhaseA = 0;
  private vocalSyncA = 0;
  private vocalSyncB = 0;
  private instPhaseA = 0;
  private instSyncA = 0;
  private instSyncB = 0;
  private smoothPitchRatio = 1.0;
  private bqZ1L = 0;
  private bqZ2L = 0;
  private bqZ1R = 0;
  private bqZ2R = 0;
  private smoothLeadGain = 1.0;
  private smoothBackingGain = 1.0;
  private smoothInstGain = 1.0;
  private smoothReverbGain = 0.55;
  private smoothMasterGain = 1.0;
  private masterGain = 1.0;
  private limiterEnabled = true;

  public setMasterVolume(volumePercent: number): void {
    this.masterGain = Math.max(0, Math.min(2.0, volumePercent / 100));
  }

  public setLimiterEnabled(enabled: boolean): void {
    this.limiterEnabled = enabled;
  }

  public getCustomAudioBuffer(): AudioBuffer | null {
    return this.customBuffer;
  }

  public setSeparatedStems(stems: SeparatedStemsMap | null): void {
    this.separatedStems = stems;
  }

  public hasSeparatedStems(): boolean {
    return this.separatedStems !== null;
  }

  public clearTrack(): void {
    this.stop();
    this.customBuffer = null;
    this.separatedStems = null;
    this.playheadSample = 0;
  }

  public async loadAudioFile(file: File): Promise<{
    durationSec: number;
    sampleRate: number;
    channels: number;
    detectedKey: string;
    detectedBpm: number;
    peaks: Record<StemId, number[]>;
  }> {
    const arrayBuffer = await file.arrayBuffer();
    return this.loadAudioFromArrayBuffer(arrayBuffer);
  }

  public async loadAudioFromArrayBuffer(arrayBuffer: ArrayBuffer): Promise<{
    durationSec: number;
    sampleRate: number;
    channels: number;
    detectedKey: string;
    detectedBpm: number;
    peaks: Record<StemId, number[]>;
  }> {
    if (!this.ctx) {
      this.ctx = new AudioContext();
    }
    const audioBuffer = await this.ctx.decodeAudioData(arrayBuffer.slice(0));
    this.customBuffer = audioBuffer;
    this.separatedStems = null;
    this.playheadSample = 0;

    const detectedKey = detectMusicalKeyFromBuffer(audioBuffer);
    const detectedBpm = detectBpmFromBuffer(audioBuffer);
    const basePeaks = extractPeaksFromBuffer(audioBuffer, 140);

    return {
      durationSec: audioBuffer.duration,
      sampleRate: audioBuffer.sampleRate,
      channels: audioBuffer.numberOfChannels,
      detectedKey,
      detectedBpm,
      peaks: {
        lead_vocal: basePeaks,
        backing_vocal: basePeaks,
        instrumental: basePeaks,
        reverb_tail: basePeaks,
      },
    };
  }

  public async decodeWavArrayBufferToStereo(wavBuffer: ArrayBuffer): Promise<{
    sampleRate: number;
    left: Float32Array;
    right: Float32Array;
  }> {
    if (!this.ctx) {
      this.ctx = new AudioContext();
    }
    const decoded = await this.ctx.decodeAudioData(wavBuffer.slice(0));
    const left = new Float32Array(decoded.getChannelData(0));
    const right =
      decoded.numberOfChannels > 1
        ? new Float32Array(decoded.getChannelData(1))
        : new Float32Array(left);
    return {
      sampleRate: decoded.sampleRate,
      left,
      right,
    };
  }

  public seekToSeconds(sec: number): void {
    if (this.customBuffer) {
      const maxLen = this.separatedStems
        ? this.separatedStems.instrumental.left.length
        : this.customBuffer.length;
      this.playheadSample = Math.max(
        0,
        Math.min(maxLen - 1, Math.floor(sec * this.customBuffer.sampleRate))
      );
      this.vocalPhaseA = 0;
      this.vocalSyncA = 0;
      this.vocalSyncB = 0;
      this.instPhaseA = 0;
      this.instSyncA = 0;
      this.instSyncB = 0;
    }
  }

  public start(
    getStems: () => StemTrackState[],
    getPitchSpeed: () => PitchSpeedSettings,
    onTick: (deltaSec: number, peakLevelDb: number) => void
  ): void {
    if (this.isPlaying || !this.customBuffer) return;
    if (!this.ctx) {
      this.ctx = new AudioContext();
    }
    if (this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
    this.isPlaying = true;
    this.startCustomBufferRealtimeEngine(getStems, getPitchSpeed, onTick);
  }

  private computeFormantPeakingCoeffs(
    sampleRate: number,
    semitones: number,
    enabled: boolean
  ): [number, number, number, number, number] {
    if (!enabled || Math.abs(semitones) < 0.05) {
      return [1, 0, 0, 0, 0];
    }
    const gainDb = Math.max(-7.5, Math.min(7.5, -semitones * 0.72));
    const A = Math.pow(10, gainDb / 40);
    const w0 = (2 * Math.PI * 1120) / Math.max(8000, sampleRate);
    const cosW0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * 1.15);

    const b0 = 1 + alpha * A;
    const b1 = -2 * cosW0;
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * cosW0;
    const a2 = 1 - alpha / A;

    const invA0 = 1 / a0;
    return [b0 * invA0, b1 * invA0, b2 * invA0, a1 * invA0, a2 * invA0];
  }

  private startCustomBufferRealtimeEngine(
    getStems: () => StemTrackState[],
    getPitchSpeed: () => PitchSpeedSettings,
    onTick: (deltaSec: number, peakLevelDb: number) => void
  ): void {
    if (!this.ctx || !this.customBuffer) return;

    const buffer = this.customBuffer;
    const chLeft = buffer.getChannelData(0);
    const chRight = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : chLeft;
    const sampleRate = buffer.sampleRate;
    const blockSize = 2048;
    const vocalGrainSize = 1600;
    const instGrainSize = 1280;

    // Initialize smooth gains to current target values so playback starts cleanly
    const initStems = getStems();
    const initAnySolo = initStems.some((s) => s.solo);
    const readTargetGain = (stemsList: StemTrackState[], anySoloFlag: boolean, id: StemId) => {
      const st = stemsList.find((x) => x.id === id);
      if (!st || st.muted || (anySoloFlag && !st.solo)) return 0;
      return st.volume / 100;
    };
    this.smoothLeadGain = readTargetGain(initStems, initAnySolo, 'lead_vocal');
    this.smoothBackingGain = readTargetGain(initStems, initAnySolo, 'backing_vocal');
    this.smoothInstGain = readTargetGain(initStems, initAnySolo, 'instrumental');
    this.smoothReverbGain = readTargetGain(initStems, initAnySolo, 'reverb_tail');
    this.smoothMasterGain = this.masterGain;

    const initPitch = getPitchSpeed();
    this.smoothPitchRatio = Math.pow(2, (initPitch.semitones + initPitch.cents / 100) / 12);
    this.bqZ1L = 0;
    this.bqZ2L = 0;
    this.bqZ1R = 0;
    this.bqZ2R = 0;
    this.pendingDeltaSec = 0;
    this.lastUiFlushMs = performance.now();

    const processor = this.ctx.createScriptProcessor(blockSize, 0, 2);

    // Decoupled UI ticker (~15 fps) so React state updates never stall the audio callback
    const scheduleUiFlush = () => {
      if (!this.isPlaying) return;
      const now = performance.now();
      if (now - this.lastUiFlushMs >= 64 && this.pendingDeltaSec > 0) {
        const delta = this.pendingDeltaSec;
        this.pendingDeltaSec = 0;
        this.lastUiFlushMs = now;
        onTick(delta, this.latestPeakDb);
      }
      this.uiRafId = window.requestAnimationFrame(scheduleUiFlush);
    };
    this.uiRafId = window.requestAnimationFrame(scheduleUiFlush);

    processor.onaudioprocess = (event) => {
      if (!this.isPlaying) return;

      const outL = event.outputBuffer.getChannelData(0);
      const outR = event.outputBuffer.getChannelData(1);

      const settings = getPitchSpeed();
      const stems = getStems();
      const anySolo = stems.some((s) => s.solo);

      const targetLead = readTargetGain(stems, anySolo, 'lead_vocal');
      const targetBacking = readTargetGain(stems, anySolo, 'backing_vocal');
      const targetInst = readTargetGain(stems, anySolo, 'instrumental');
      const targetReverb = readTargetGain(stems, anySolo, 'reverb_tail');
      const targetMaster = this.masterGain;

      const targetPitchRatio = Math.pow(2, (settings.semitones + settings.cents / 100) / 12);
      const speedRatio = Math.max(0.25, Math.min(2.0, settings.speed));
      const sep = this.separatedStems;
      const activeTotalSamples = sep ? sep.instrumental.left.length : buffer.length;

      const isUnityPitchSpeed =
        Math.abs(targetPitchRatio - 1.0) < 1e-4 &&
        Math.abs(this.smoothPitchRatio - 1.0) < 1e-4 &&
        Math.abs(speedRatio - 1.0) < 1e-4;

      const [b0, b1, b2, a1, a2] = this.computeFormantPeakingCoeffs(
        sampleRate,
        settings.semitones + settings.cents / 100,
        settings.formantPreservation && !isUnityPitchSpeed
      );

      const gainAlpha = 0.003;
      const pitchAlpha = 0.0025;
      let maxPeak = 0.0001;

      const vocalRef = sep ? sep.lead_vocal.left : chLeft;
      const instRef = sep ? sep.instrumental.left : chLeft;

      for (let i = 0; i < blockSize; i++) {
        this.smoothLeadGain += (targetLead - this.smoothLeadGain) * gainAlpha;
        this.smoothBackingGain += (targetBacking - this.smoothBackingGain) * gainAlpha;
        this.smoothInstGain += (targetInst - this.smoothInstGain) * gainAlpha;
        this.smoothReverbGain += (targetReverb - this.smoothReverbGain) * gainAlpha;
        this.smoothMasterGain += (targetMaster - this.smoothMasterGain) * gainAlpha;
        this.smoothPitchRatio += (targetPitchRatio - this.smoothPitchRatio) * pitchAlpha;

        let vocL = 0;
        let vocR = 0;
        let insL = 0;
        let insR = 0;

        if (isUnityPitchSpeed) {
          this.vocalPhaseA = 0;
          this.vocalSyncA = 0;
          this.vocalSyncB = 0;
          this.instPhaseA = 0;
          this.instSyncA = 0;
          this.instSyncB = 0;

          if (this.playheadSample >= activeTotalSamples - 2) {
            this.playheadSample = 0;
          }
          const pos = this.playheadSample;
          this.playheadSample += 1;

          if (sep) {
            if (this.smoothLeadGain > 1e-4) {
              vocL += cubicHermiteAt(sep.lead_vocal.left, pos) * this.smoothLeadGain;
              vocR += cubicHermiteAt(sep.lead_vocal.right, pos) * this.smoothLeadGain;
            }
            if (this.smoothBackingGain > 1e-4) {
              vocL += cubicHermiteAt(sep.backing_vocal.left, pos) * this.smoothBackingGain;
              vocR += cubicHermiteAt(sep.backing_vocal.right, pos) * this.smoothBackingGain;
            }
            if (this.smoothInstGain > 1e-4) {
              insL += cubicHermiteAt(sep.instrumental.left, pos) * this.smoothInstGain;
              insR += cubicHermiteAt(sep.instrumental.right, pos) * this.smoothInstGain;
            }
            if (this.smoothReverbGain > 1e-4) {
              insL += cubicHermiteAt(sep.reverb_tail.left, pos) * this.smoothReverbGain;
              insR += cubicHermiteAt(sep.reverb_tail.right, pos) * this.smoothReverbGain;
            }
          } else {
            const rawL = cubicHermiteAt(chLeft, pos);
            const rawR = cubicHermiteAt(chRight, pos);
            insL = rawL;
            insR = rawR;
          }
        } else {
          this.playheadSample += speedRatio;
          if (this.playheadSample >= activeTotalSamples - vocalGrainSize - 480) {
            this.playheadSample = 0;
            this.vocalSyncA = 0;
            this.vocalSyncB = 0;
            this.instSyncA = 0;
            this.instSyncB = 0;
          }

          // 1. Phase-locked Vocal WSOLA stream
          const vocRate = (this.smoothPitchRatio - speedRatio) / vocalGrainSize;
          const prevVocA = this.vocalPhaseA;
          let nextVocA = prevVocA + vocRate;
          if (vocRate >= 0 && nextVocA >= 1.0) {
            nextVocA -= Math.floor(nextVocA);
            const refPosB = this.playheadSample + 0.5 * vocalGrainSize + this.vocalSyncB;
            this.vocalSyncA = findBestWsolaOffset(
              vocalRef,
              refPosB,
              this.playheadSample + nextVocA * vocalGrainSize,
              360
            );
          } else if (vocRate < 0 && nextVocA < 0.0) {
            nextVocA = ((nextVocA % 1.0) + 1.0) % 1.0;
            const refPosB = this.playheadSample + 0.5 * vocalGrainSize + this.vocalSyncB;
            this.vocalSyncA = findBestWsolaOffset(
              vocalRef,
              refPosB,
              this.playheadSample + nextVocA * vocalGrainSize,
              360
            );
          }
          if (
            (vocRate >= 0 && prevVocA < 0.5 && nextVocA >= 0.5) ||
            (vocRate < 0 && prevVocA >= 0.5 && nextVocA < 0.5)
          ) {
            const nextVocB = (nextVocA + 0.5) % 1.0;
            const refPosA = this.playheadSample + 0.5 * vocalGrainSize + this.vocalSyncA;
            this.vocalSyncB = findBestWsolaOffset(
              vocalRef,
              refPosA,
              this.playheadSample + nextVocB * vocalGrainSize,
              360
            );
          }
          this.vocalPhaseA = nextVocA;
          const vocPhaseB = (nextVocA + 0.5) % 1.0;
          const vWinA = 0.5 * (1 - Math.cos(2 * Math.PI * nextVocA));
          const vWinB = 0.5 * (1 - Math.cos(2 * Math.PI * vocPhaseB));
          const vPosA = this.playheadSample + nextVocA * vocalGrainSize + this.vocalSyncA;
          const vPosB = this.playheadSample + vocPhaseB * vocalGrainSize + this.vocalSyncB;

          // 2. Phase-locked Instrumental WSOLA stream (tighter grain preserves drum attacks)
          const insRate = (this.smoothPitchRatio - speedRatio) / instGrainSize;
          const prevInsA = this.instPhaseA;
          let nextInsA = prevInsA + insRate;
          if (insRate >= 0 && nextInsA >= 1.0) {
            nextInsA -= Math.floor(nextInsA);
            const refPosB = this.playheadSample + 0.5 * instGrainSize + this.instSyncB;
            this.instSyncA = findBestWsolaOffset(
              instRef,
              refPosB,
              this.playheadSample + nextInsA * instGrainSize,
              300
            );
          } else if (insRate < 0 && nextInsA < 0.0) {
            nextInsA = ((nextInsA % 1.0) + 1.0) % 1.0;
            const refPosB = this.playheadSample + 0.5 * instGrainSize + this.instSyncB;
            this.instSyncA = findBestWsolaOffset(
              instRef,
              refPosB,
              this.playheadSample + nextInsA * instGrainSize,
              300
            );
          }
          if (
            (insRate >= 0 && prevInsA < 0.5 && nextInsA >= 0.5) ||
            (insRate < 0 && prevInsA >= 0.5 && nextInsA < 0.5)
          ) {
            const nextInsB = (nextInsA + 0.5) % 1.0;
            const refPosA = this.playheadSample + 0.5 * instGrainSize + this.instSyncA;
            this.instSyncB = findBestWsolaOffset(
              instRef,
              refPosA,
              this.playheadSample + nextInsB * instGrainSize,
              300
            );
          }
          this.instPhaseA = nextInsA;
          const insPhaseB = (nextInsA + 0.5) % 1.0;
          const iWinA = 0.5 * (1 - Math.cos(2 * Math.PI * nextInsA));
          const iWinB = 0.5 * (1 - Math.cos(2 * Math.PI * insPhaseB));
          const iPosA = this.playheadSample + nextInsA * instGrainSize + this.instSyncA;
          const iPosB = this.playheadSample + insPhaseB * instGrainSize + this.instSyncB;

          if (sep) {
            if (this.smoothLeadGain > 1e-4) {
              vocL +=
                (cubicHermiteAt(sep.lead_vocal.left, vPosA) * vWinA +
                  cubicHermiteAt(sep.lead_vocal.left, vPosB) * vWinB) *
                this.smoothLeadGain;
              vocR +=
                (cubicHermiteAt(sep.lead_vocal.right, vPosA) * vWinA +
                  cubicHermiteAt(sep.lead_vocal.right, vPosB) * vWinB) *
                this.smoothLeadGain;
            }
            if (this.smoothBackingGain > 1e-4) {
              vocL +=
                (cubicHermiteAt(sep.backing_vocal.left, vPosA) * vWinA +
                  cubicHermiteAt(sep.backing_vocal.left, vPosB) * vWinB) *
                this.smoothBackingGain;
              vocR +=
                (cubicHermiteAt(sep.backing_vocal.right, vPosA) * vWinA +
                  cubicHermiteAt(sep.backing_vocal.right, vPosB) * vWinB) *
                this.smoothBackingGain;
            }
            if (this.smoothInstGain > 1e-4) {
              insL +=
                (cubicHermiteAt(sep.instrumental.left, iPosA) * iWinA +
                  cubicHermiteAt(sep.instrumental.left, iPosB) * iWinB) *
                this.smoothInstGain;
              insR +=
                (cubicHermiteAt(sep.instrumental.right, iPosA) * iWinA +
                  cubicHermiteAt(sep.instrumental.right, iPosB) * iWinB) *
                this.smoothInstGain;
            }
            if (this.smoothReverbGain > 1e-4) {
              insL +=
                (cubicHermiteAt(sep.reverb_tail.left, iPosA) * iWinA +
                  cubicHermiteAt(sep.reverb_tail.left, iPosB) * iWinB) *
                this.smoothReverbGain;
              insR +=
                (cubicHermiteAt(sep.reverb_tail.right, iPosA) * iWinA +
                  cubicHermiteAt(sep.reverb_tail.right, iPosB) * iWinB) *
                this.smoothReverbGain;
            }
          } else {
            const rawL =
              cubicHermiteAt(chLeft, iPosA) * iWinA + cubicHermiteAt(chLeft, iPosB) * iWinB;
            const rawR =
              cubicHermiteAt(chRight, iPosA) * iWinA + cubicHermiteAt(chRight, iPosB) * iWinB;
            if (settings.formantPreservation) {
              const mid = 0.5 * (rawL + rawR);
              const side = 0.5 * (rawL - rawR);
              vocL = mid;
              vocR = mid;
              insL = side;
              insR = -side;
            } else {
              insL = rawL;
              insR = rawR;
            }
          }

          // Apply Formant EQ exclusively to the Vocal branch so drums/guitars stay crisp
          const filtVocL = b0 * vocL + this.bqZ1L;
          this.bqZ1L = b1 * vocL - a1 * filtVocL + this.bqZ2L;
          this.bqZ2L = b2 * vocL - a2 * filtVocL;
          vocL = filtVocL;

          const filtVocR = b0 * vocR + this.bqZ1R;
          this.bqZ1R = b1 * vocR - a1 * filtVocR + this.bqZ2R;
          this.bqZ2R = b2 * vocR - a2 * filtVocR;
          vocR = filtVocR;
        }

        const scaledL = (vocL + insL) * this.smoothMasterGain;
        const scaledR = (vocR + insR) * this.smoothMasterGain;
        const ceiling = this.limiterEnabled ? 0.966 : 0.999;
        const mixL = softLimitStudioSample(scaledL, ceiling);
        const mixR = softLimitStudioSample(scaledR, ceiling);

        outL[i] = mixL;
        outR[i] = mixR;

        const absMax = Math.max(Math.abs(mixL), Math.abs(mixR));
        if (absMax > maxPeak) maxPeak = absMax;
      }

      this.latestPeakDb = Math.max(-60, 20 * Math.log10(maxPeak));
      this.pendingDeltaSec += (blockSize / sampleRate) * speedRatio;
      // Background-tab safety flush when requestAnimationFrame is throttled
      if (this.pendingDeltaSec >= 0.25) {
        const delta = this.pendingDeltaSec;
        this.pendingDeltaSec = 0;
        this.lastUiFlushMs = performance.now();
        onTick(delta, this.latestPeakDb);
      }
    };

    processor.connect(this.ctx.destination);
    this.scriptProcessor = processor;
  }

  public stop(): void {
    this.isPlaying = false;
    if (this.uiRafId !== null) {
      window.cancelAnimationFrame(this.uiRafId);
      this.uiRafId = null;
    }
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor.onaudioprocess = null;
      this.scriptProcessor = null;
    }
  }

  /**
   * Renders one or multiple downloadable WAV / FLAC files according to the selected export target & format.
   */
  public renderExportAudioFiles(
    baseName: string,
    stems: StemTrackState[],
    settings: PitchSpeedSettings,
    target: 'master_mix' | 'all_stems' | 'karaoke_backing',
    format: BrowserAudioExportFormat
  ): { fileName: string; blob: Blob }[] {
    const ext = format === 'flac' ? 'flac' : 'wav';

    if (target === 'all_stems') {
      const stemOrder: StemId[] = [
        'lead_vocal',
        'backing_vocal',
        'instrumental',
        'reverb_tail',
      ];
      return stemOrder.map((stemId) => {
        const soloedStems = stems.map((s) => ({
          ...s,
          muted: s.id !== stemId,
          solo: s.id === stemId,
        }));
        return {
          fileName: `${baseName}_${stemId}.${ext}`,
          blob: this.renderSingleMixBlob(soloedStems, settings, format),
        };
      });
    }

    if (target === 'karaoke_backing') {
      const karaokeStems = stems.map((s) => ({
        ...s,
        solo: false,
        muted: s.id === 'lead_vocal' || s.id === 'reverb_tail',
      }));
      return [
        {
          fileName: `${baseName}_karaoke_with_backing.${ext}`,
          blob: this.renderSingleMixBlob(karaokeStems, settings, format),
        },
      ];
    }

    return [
      {
        fileName: `${baseName}.${ext}`,
        blob: this.renderSingleMixBlob(stems, settings, format),
      },
    ];
  }

  private renderSingleMixBlob(
    stems: StemTrackState[],
    settings: PitchSpeedSettings,
    format: BrowserAudioExportFormat
  ): Blob {
    if (!this.customBuffer) {
      throw new Error('Chưa có file âm thanh thực tế để xuất.');
    }

    const sampleRate = this.customBuffer.sampleRate;
    const anySolo = stems.some((s) => s.solo);
    const getStemGain = (id: StemId) => {
      const st = stems.find((x) => x.id === id);
      if (!st || st.muted || (anySolo && !st.solo)) return 0;
      return st.volume / 100;
    };

    const pitchRatio = Math.pow(2, (settings.semitones + settings.cents / 100) / 12);
    const speedRatio = Math.max(0.25, Math.min(2.0, settings.speed));

    const leadGain = getStemGain('lead_vocal');
    const backingGain = getStemGain('backing_vocal');
    const instGain = getStemGain('instrumental');
    const reverbGain = getStemGain('reverb_tail');

    const sep = this.separatedStems;
    const rawL = this.customBuffer.getChannelData(0);
    const rawR =
      this.customBuffer.numberOfChannels > 1
        ? this.customBuffer.getChannelData(1)
        : rawL;

    const srcLen = sep ? sep.instrumental.left.length : this.customBuffer.length;
    const outFrames = Math.max(1, Math.floor(srcLen / speedRatio));
    const outL = new Float32Array(outFrames);
    const outR = new Float32Array(outFrames);

    const isUnityPitchSpeed =
      Math.abs(pitchRatio - 1.0) < 1e-4 && Math.abs(speedRatio - 1.0) < 1e-4;

    if (isUnityPitchSpeed) {
      if (sep) {
        for (let i = 0; i < outFrames; i++) {
          const idx = Math.min(srcLen - 1, i);
          outL[i] =
            sep.lead_vocal.left[idx] * leadGain +
            sep.backing_vocal.left[idx] * backingGain +
            sep.instrumental.left[idx] * instGain +
            sep.reverb_tail.left[idx] * reverbGain;
          outR[i] =
            sep.lead_vocal.right[idx] * leadGain +
            sep.backing_vocal.right[idx] * backingGain +
            sep.instrumental.right[idx] * instGain +
            sep.reverb_tail.right[idx] * reverbGain;
        }
      } else {
        for (let i = 0; i < outFrames; i++) {
          const idx = Math.min(srcLen - 1, i);
          outL[i] = rawL[idx] * this.masterGain;
          outR[i] = rawR[idx] * this.masterGain;
        }
      }
      return encodeStereoAudioBlob(outL, outR, sampleRate, format);
    }

    // Full-length Dual-Stream Phase-Locked WSOLA rendering with Vocal-Only Formant Compensation
    const vocalGrainSize = 1600;
    const instGrainSize = 1280;
    const vocRate = (pitchRatio - speedRatio) / vocalGrainSize;
    const insRate = (pitchRatio - speedRatio) / instGrainSize;
    const vocalRef = sep ? sep.lead_vocal.left : rawL;
    const instRef = sep ? sep.instrumental.left : rawL;
    const [b0, b1, b2, a1, a2] = this.computeFormantPeakingCoeffs(
      sampleRate,
      settings.semitones + settings.cents / 100,
      settings.formantPreservation
    );

    let playhead = 0;
    let vocPhaseA = 0;
    let vocSyncA = 0;
    let vocSyncB = 0;
    let insPhaseA = 0;
    let insSyncA = 0;
    let insSyncB = 0;
    let z1L = 0;
    let z2L = 0;
    let z1R = 0;
    let z2R = 0;

    for (let i = 0; i < outFrames; i++) {
      const prevVocA = vocPhaseA;
      let nextVocA = prevVocA + vocRate;
      if (vocRate >= 0 && nextVocA >= 1.0) {
        nextVocA -= Math.floor(nextVocA);
        vocSyncA = findBestWsolaOffset(
          vocalRef,
          playhead + 0.5 * vocalGrainSize + vocSyncB,
          playhead + nextVocA * vocalGrainSize,
          360
        );
      } else if (vocRate < 0 && nextVocA < 0.0) {
        nextVocA = ((nextVocA % 1.0) + 1.0) % 1.0;
        vocSyncA = findBestWsolaOffset(
          vocalRef,
          playhead + 0.5 * vocalGrainSize + vocSyncB,
          playhead + nextVocA * vocalGrainSize,
          360
        );
      }
      if (
        (vocRate >= 0 && prevVocA < 0.5 && nextVocA >= 0.5) ||
        (vocRate < 0 && prevVocA >= 0.5 && nextVocA < 0.5)
      ) {
        const nextVocB = (nextVocA + 0.5) % 1.0;
        vocSyncB = findBestWsolaOffset(
          vocalRef,
          playhead + 0.5 * vocalGrainSize + vocSyncA,
          playhead + nextVocB * vocalGrainSize,
          360
        );
      }
      vocPhaseA = nextVocA;
      const vocPhaseB = (nextVocA + 0.5) % 1.0;
      const vWinA = 0.5 * (1 - Math.cos(2 * Math.PI * nextVocA));
      const vWinB = 0.5 * (1 - Math.cos(2 * Math.PI * vocPhaseB));
      const vPosA = playhead + nextVocA * vocalGrainSize + vocSyncA;
      const vPosB = playhead + vocPhaseB * vocalGrainSize + vocSyncB;

      const prevInsA = insPhaseA;
      let nextInsA = prevInsA + insRate;
      if (insRate >= 0 && nextInsA >= 1.0) {
        nextInsA -= Math.floor(nextInsA);
        insSyncA = findBestWsolaOffset(
          instRef,
          playhead + 0.5 * instGrainSize + insSyncB,
          playhead + nextInsA * instGrainSize,
          300
        );
      } else if (insRate < 0 && nextInsA < 0.0) {
        nextInsA = ((nextInsA % 1.0) + 1.0) % 1.0;
        insSyncA = findBestWsolaOffset(
          instRef,
          playhead + 0.5 * instGrainSize + insSyncB,
          playhead + nextInsA * instGrainSize,
          300
        );
      }
      if (
        (insRate >= 0 && prevInsA < 0.5 && nextInsA >= 0.5) ||
        (insRate < 0 && prevInsA >= 0.5 && nextInsA < 0.5)
      ) {
        const nextInsB = (nextInsA + 0.5) % 1.0;
        insSyncB = findBestWsolaOffset(
          instRef,
          playhead + 0.5 * instGrainSize + insSyncA,
          playhead + nextInsB * instGrainSize,
          300
        );
      }
      insPhaseA = nextInsA;
      const insPhaseB = (nextInsA + 0.5) % 1.0;
      const iWinA = 0.5 * (1 - Math.cos(2 * Math.PI * nextInsA));
      const iWinB = 0.5 * (1 - Math.cos(2 * Math.PI * insPhaseB));
      const iPosA = playhead + nextInsA * instGrainSize + insSyncA;
      const iPosB = playhead + insPhaseB * instGrainSize + insSyncB;

      let vocL = 0;
      let vocR = 0;
      let insL = 0;
      let insR = 0;

      if (sep) {
        if (leadGain > 1e-4) {
          vocL +=
            (cubicHermiteAt(sep.lead_vocal.left, vPosA) * vWinA +
              cubicHermiteAt(sep.lead_vocal.left, vPosB) * vWinB) *
            leadGain;
          vocR +=
            (cubicHermiteAt(sep.lead_vocal.right, vPosA) * vWinA +
              cubicHermiteAt(sep.lead_vocal.right, vPosB) * vWinB) *
            leadGain;
        }
        if (backingGain > 1e-4) {
          vocL +=
            (cubicHermiteAt(sep.backing_vocal.left, vPosA) * vWinA +
              cubicHermiteAt(sep.backing_vocal.left, vPosB) * vWinB) *
            backingGain;
          vocR +=
            (cubicHermiteAt(sep.backing_vocal.right, vPosA) * vWinA +
              cubicHermiteAt(sep.backing_vocal.right, vPosB) * vWinB) *
            backingGain;
        }
        if (instGain > 1e-4) {
          insL +=
            (cubicHermiteAt(sep.instrumental.left, iPosA) * iWinA +
              cubicHermiteAt(sep.instrumental.left, iPosB) * iWinB) *
            instGain;
          insR +=
            (cubicHermiteAt(sep.instrumental.right, iPosA) * iWinA +
              cubicHermiteAt(sep.instrumental.right, iPosB) * iWinB) *
            instGain;
        }
        if (reverbGain > 1e-4) {
          insL +=
            (cubicHermiteAt(sep.reverb_tail.left, iPosA) * iWinA +
              cubicHermiteAt(sep.reverb_tail.left, iPosB) * iWinB) *
            reverbGain;
          insR +=
            (cubicHermiteAt(sep.reverb_tail.right, iPosA) * iWinA +
              cubicHermiteAt(sep.reverb_tail.right, iPosB) * iWinB) *
            reverbGain;
        }
      } else {
        const l = cubicHermiteAt(rawL, iPosA) * iWinA + cubicHermiteAt(rawL, iPosB) * iWinB;
        const r = cubicHermiteAt(rawR, iPosA) * iWinA + cubicHermiteAt(rawR, iPosB) * iWinB;
        if (settings.formantPreservation) {
          const mid = 0.5 * (l + r);
          const side = 0.5 * (l - r);
          vocL = mid;
          vocR = mid;
          insL = side;
          insR = -side;
        } else {
          insL = l;
          insR = r;
        }
      }

      const filtVocL = b0 * vocL + z1L;
      z1L = b1 * vocL - a1 * filtVocL + z2L;
      z2L = b2 * vocL - a2 * filtVocL;
      vocL = filtVocL;

      const filtVocR = b0 * vocR + z1R;
      z1R = b1 * vocR - a1 * filtVocR + z2R;
      z2R = b2 * vocR - a2 * filtVocR;
      vocR = filtVocR;

      outL[i] = (vocL + insL) * this.masterGain;
      outR[i] = (vocR + insR) * this.masterGain;
      playhead += speedRatio;
    }

    return encodeStereoAudioBlob(outL, outR, sampleRate, format);
  }
}
