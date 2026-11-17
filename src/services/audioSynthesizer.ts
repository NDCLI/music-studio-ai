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

/**
 * Real-time Multi-Stem & Granular Formant Pitch/Time Engine.
 * Operates exclusively on real decoded audio/video tracks and separated stems
 * with full-duration export (WAV 16/24-bit, WAV 32-bit Float, FLAC 24-bit).
 */
export class StudioPreviewAudioEngine {
  private ctx: AudioContext | null = null;
  private isPlaying = false;
  private customBuffer: AudioBuffer | null = null;
  private separatedStems: SeparatedStemsMap | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private formantFilterNode: BiquadFilterNode | null = null;
  private playheadSample = 0;
  private grainPhaseA = 0;
  private grainPhaseB = 0.5;
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
      this.playheadSample = Math.max(
        0,
        Math.min(
          this.customBuffer.length - 1,
          Math.floor(sec * this.customBuffer.sampleRate)
        )
      );
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

  private startCustomBufferRealtimeEngine(
    getStems: () => StemTrackState[],
    getPitchSpeed: () => PitchSpeedSettings,
    onTick: (deltaSec: number, peakLevelDb: number) => void
  ): void {
    if (!this.ctx || !this.customBuffer) return;

    const buffer = this.customBuffer;
    const chLeft = buffer.getChannelData(0);
    const chRight = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : chLeft;
    const totalSamples = buffer.length;
    const sampleRate = buffer.sampleRate;
    const blockSize = 2048;
    const grainSize = 2400;

    const processor = this.ctx.createScriptProcessor(blockSize, 0, 2);
    const formantFilter = this.ctx.createBiquadFilter();
    formantFilter.type = 'peaking';
    formantFilter.frequency.value = 1150;
    formantFilter.Q.value = 1.4;

    processor.onaudioprocess = (event) => {
      if (!this.isPlaying) return;

      const outL = event.outputBuffer.getChannelData(0);
      const outR = event.outputBuffer.getChannelData(1);

      const settings = getPitchSpeed();
      const stems = getStems();
      const anySolo = stems.some((s) => s.solo);

      const getStemGain = (id: StemId) => {
        const st = stems.find((x) => x.id === id);
        if (!st || st.muted || (anySolo && !st.solo)) return 0;
        return st.volume / 100;
      };

      const leadGain = getStemGain('lead_vocal');
      const backingGain = getStemGain('backing_vocal');
      const instGain = getStemGain('instrumental');
      const reverbGain = getStemGain('reverb_tail');

      const pitchRatio = Math.pow(2, (settings.semitones + settings.cents / 100) / 12);
      const speedRatio = Math.max(0.25, Math.min(2.0, settings.speed));
      const grainRate = (pitchRatio - speedRatio) / grainSize;

      if (settings.formantPreservation && Math.abs(settings.semitones) > 0) {
        formantFilter.gain.value = -settings.semitones * 0.85;
      } else {
        formantFilter.gain.value = 0;
      }

      const sep = this.separatedStems;
      let maxPeak = 0.0001;

      for (let i = 0; i < blockSize; i++) {
        this.playheadSample += speedRatio;
        if (this.playheadSample >= totalSamples - grainSize - 2) {
          this.playheadSample = 0;
        }

        this.grainPhaseA = (this.grainPhaseA + grainRate + 1.0) % 1.0;
        this.grainPhaseB = (this.grainPhaseA + 0.5) % 1.0;

        const winA = 0.5 * (1 - Math.cos(2 * Math.PI * this.grainPhaseA));
        const winB = 0.5 * (1 - Math.cos(2 * Math.PI * this.grainPhaseB));

        const rawPosA = Math.min(
          totalSamples - 2,
          Math.max(0, this.playheadSample + this.grainPhaseA * grainSize)
        );
        const rawPosB = Math.min(
          totalSamples - 2,
          Math.max(0, this.playheadSample + this.grainPhaseB * grainSize)
        );
        const idxA = Math.floor(rawPosA);
        const fracA = rawPosA - idxA;
        const idxB = Math.floor(rawPosB);
        const fracB = rawPosB - idxB;

        const lerp = (arr: Float32Array, idx: number, f: number) =>
          arr[idx] + (arr[idx + 1] - arr[idx]) * f;

        let sumL = 0;
        let sumR = 0;

        if (sep) {
          if (leadGain > 0.001) {
            sumL += (lerp(sep.lead_vocal.left, idxA, fracA) * winA + lerp(sep.lead_vocal.left, idxB, fracB) * winB) * leadGain;
            sumR += (lerp(sep.lead_vocal.right, idxA, fracA) * winA + lerp(sep.lead_vocal.right, idxB, fracB) * winB) * leadGain;
          }
          if (backingGain > 0.001) {
            sumL += (lerp(sep.backing_vocal.left, idxA, fracA) * winA + lerp(sep.backing_vocal.left, idxB, fracB) * winB) * backingGain;
            sumR += (lerp(sep.backing_vocal.right, idxA, fracA) * winA + lerp(sep.backing_vocal.right, idxB, fracB) * winB) * backingGain;
          }
          if (instGain > 0.001) {
            sumL += (lerp(sep.instrumental.left, idxA, fracA) * winA + lerp(sep.instrumental.left, idxB, fracB) * winB) * instGain;
            sumR += (lerp(sep.instrumental.right, idxA, fracA) * winA + lerp(sep.instrumental.right, idxB, fracB) * winB) * instGain;
          }
          if (reverbGain > 0.001) {
            sumL += (lerp(sep.reverb_tail.left, idxA, fracA) * winA + lerp(sep.reverb_tail.left, idxB, fracB) * winB) * reverbGain;
            sumR += (lerp(sep.reverb_tail.right, idxA, fracA) * winA + lerp(sep.reverb_tail.right, idxB, fracB) * winB) * reverbGain;
          }
        } else {
          const rawL = lerp(chLeft, idxA, fracA) * winA + lerp(chLeft, idxB, fracB) * winB;
          const rawR = lerp(chRight, idxA, fracA) * winA + lerp(chRight, idxB, fracB) * winB;

          const mid = 0.5 * (rawL + rawR);
          const side = 0.5 * (rawL - rawR);

          const vocalComponent = mid * leadGain * 0.75;
          const backingComponentL = side * backingGain * 0.9;
          const backingComponentR = -side * backingGain * 0.9;
          const instComponentL = (rawL - mid * 0.55) * instGain;
          const instComponentR = (rawR - mid * 0.55) * instGain;
          const revComponent = side * reverbGain * 0.35;

          sumL = vocalComponent + backingComponentL + instComponentL + revComponent;
          sumR = vocalComponent + backingComponentR + instComponentR - revComponent;
        }

        const scaledL = sumL * this.masterGain;
        const scaledR = sumR * this.masterGain;
        const mixL = this.limiterEnabled
          ? Math.max(-0.966, Math.min(0.966, scaledL))
          : Math.max(-1.0, Math.min(1.0, scaledL));
        const mixR = this.limiterEnabled
          ? Math.max(-0.966, Math.min(0.966, scaledR))
          : Math.max(-1.0, Math.min(1.0, scaledR));

        outL[i] = mixL;
        outR[i] = mixR;

        const absMax = Math.max(Math.abs(mixL), Math.abs(mixR));
        if (absMax > maxPeak) maxPeak = absMax;
      }

      const db = Math.max(-60, 20 * Math.log10(maxPeak));
      onTick((blockSize / sampleRate) * speedRatio, db);
    };

    processor.connect(formantFilter);
    formantFilter.connect(this.ctx.destination);
    this.scriptProcessor = processor;
    this.formantFilterNode = formantFilter;
  }

  public stop(): void {
    this.isPlaying = false;
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor.onaudioprocess = null;
      this.scriptProcessor = null;
    }
    if (this.formantFilterNode) {
      this.formantFilterNode.disconnect();
      this.formantFilterNode = null;
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
          const l = rawL[i];
          const r = rawR[i];
          const mid = 0.5 * (l + r);
          const side = 0.5 * (l - r);
          outL[i] = mid * leadGain * 0.75 + side * backingGain * 0.9 + (l - mid * 0.55) * instGain + side * reverbGain * 0.35;
          outR[i] = mid * leadGain * 0.75 - side * backingGain * 0.9 + (r - mid * 0.55) * instGain - side * reverbGain * 0.35;
        }
      }
      return encodeStereoAudioBlob(outL, outR, sampleRate, format);
    }

    // Full-length Granular Hann Overlap-Add Pitch + Time rendering (preserves true duration / speedRatio)
    const grainSize = 2400;
    const grainRate = (pitchRatio - speedRatio) / grainSize;
    let playhead = 0;
    let grainPhaseA = 0;

    for (let i = 0; i < outFrames; i++) {
      const grainPhaseB = (grainPhaseA + 0.5) % 1.0;
      const winA = 0.5 * (1 - Math.cos(2 * Math.PI * grainPhaseA));
      const winB = 0.5 * (1 - Math.cos(2 * Math.PI * grainPhaseB));

      const idxA = Math.min(srcLen - 1, Math.max(0, Math.floor(playhead + grainPhaseA * grainSize)));
      const idxB = Math.min(srcLen - 1, Math.max(0, Math.floor(playhead + grainPhaseB * grainSize)));

      if (sep) {
        outL[i] =
          (sep.lead_vocal.left[idxA] * winA + sep.lead_vocal.left[idxB] * winB) * leadGain +
          (sep.backing_vocal.left[idxA] * winA + sep.backing_vocal.left[idxB] * winB) * backingGain +
          (sep.instrumental.left[idxA] * winA + sep.instrumental.left[idxB] * winB) * instGain +
          (sep.reverb_tail.left[idxA] * winA + sep.reverb_tail.left[idxB] * winB) * reverbGain;
        outR[i] =
          (sep.lead_vocal.right[idxA] * winA + sep.lead_vocal.right[idxB] * winB) * leadGain +
          (sep.backing_vocal.right[idxA] * winA + sep.backing_vocal.right[idxB] * winB) * backingGain +
          (sep.instrumental.right[idxA] * winA + sep.instrumental.right[idxB] * winB) * instGain +
          (sep.reverb_tail.right[idxA] * winA + sep.reverb_tail.right[idxB] * winB) * reverbGain;
      } else {
        const l = rawL[idxA] * winA + rawL[idxB] * winB;
        const r = rawR[idxA] * winA + rawR[idxB] * winB;
        const mid = 0.5 * (l + r);
        const side = 0.5 * (l - r);
        outL[i] = mid * leadGain * 0.75 + side * backingGain * 0.9 + (l - mid * 0.55) * instGain + side * reverbGain * 0.35;
        outR[i] = mid * leadGain * 0.75 - side * backingGain * 0.9 + (r - mid * 0.55) * instGain - side * reverbGain * 0.35;
      }

      playhead += speedRatio;
      grainPhaseA = (grainPhaseA + grainRate + 1.0) % 1.0;
    }

    return encodeStereoAudioBlob(outL, outR, sampleRate, format);
  }
}
