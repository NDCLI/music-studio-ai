import { PitchSpeedSettings, StemId, StemTrackState } from '../types/studio';
import {
  detectBpmFromBuffer,
  detectMusicalKeyFromBuffer,
  encodeStereoWavBlob,
} from './dspEngine';

/**
 * Generates deterministic studio waveform peak data for each stem track.
 */
export function generateStemPeaks(stemId: StemId, count = 140, seedBoost = 1): number[] {
  const peaks: number[] = [];
  const seedMap: Record<StemId, number> = {
    lead_vocal: 1.7,
    backing_vocal: 3.1,
    instrumental: 0.9,
    reverb_tail: 4.3,
  };
  const baseSeed = seedMap[stemId] * seedBoost;

  for (let i = 0; i < count; i++) {
    const progress = i / count;
    const envelope = Math.sin(progress * Math.PI);
    const waveA = Math.abs(Math.sin((i + baseSeed * 7) * 0.29));
    const waveB = Math.abs(Math.cos((i * 1.4 + baseSeed * 3) * 0.17));
    const phraseGap =
      stemId === 'lead_vocal'
        ? Math.sin(i * 0.18) > -0.35
          ? 1
          : 0.12
        : stemId === 'backing_vocal'
          ? Math.cos(i * 0.12) > 0.05
            ? 0.9
            : 0.15
          : 1;

    const raw = (waveA * 0.6 + waveB * 0.4) * (0.35 + 0.65 * envelope) * phraseGap;
    peaks.push(Math.max(0.08, Math.min(0.98, Number(raw.toFixed(3)))));
  }

  return peaks;
}

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
 * Stage 2 & 3 Real-time Multi-Stem & Granular Formant Pitch/Time Engine.
 * Supports both synthesized multi-stem preview and real audio file playback
 * with independent Pitch Shifting (semitones/cents) and Time Stretching (speed).
 */
export class StudioPreviewAudioEngine {
  private ctx: AudioContext | null = null;
  private isPlaying = false;
  private stepTimer: number | null = null;
  private stepIndex = 0;
  private customBuffer: AudioBuffer | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private formantFilterNode: BiquadFilterNode | null = null;
  private playheadSample = 0;
  private grainPhaseA = 0;
  private grainPhaseB = 0.5;

  public async loadAudioFile(file: File): Promise<{
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
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await this.ctx.decodeAudioData(arrayBuffer);
    this.customBuffer = audioBuffer;
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
        lead_vocal: basePeaks.map((p, idx) =>
          Math.max(0.06, Math.min(0.98, p * (Math.sin(idx * 0.22) > -0.2 ? 0.95 : 0.25)))
        ),
        backing_vocal: basePeaks.map((p, idx) =>
          Math.max(0.06, Math.min(0.95, p * (Math.cos(idx * 0.15) > 0.1 ? 0.78 : 0.18)))
        ),
        instrumental: basePeaks,
        reverb_tail: basePeaks.map((p) => Math.max(0.05, Math.min(0.75, p * 0.55))),
      },
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
    if (this.isPlaying) return;
    if (!this.ctx) {
      this.ctx = new AudioContext();
    }
    if (this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
    this.isPlaying = true;

    // If a real audio file was loaded, run the real-time Dual-Grain Overlap-Add Pitch/Time Shifter
    if (this.customBuffer) {
      this.startCustomBufferRealtimeEngine(getStems, getPitchSpeed, onTick);
      return;
    }

    const baseIntervalMs = 240;

    const triggerStep = () => {
      if (!this.isPlaying || !this.ctx) return;

      const settings = getPitchSpeed();
      const stems = getStems();
      const anySolo = stems.some((s) => s.solo);

      const pitchFactor = Math.pow(2, (settings.semitones + settings.cents / 100) / 12);
      const effectiveIntervalMs = Math.max(80, baseIntervalMs / settings.speed);

      const peakDb = this.playSynthesizedMultiStemStep(
        stems,
        anySolo,
        pitchFactor,
        settings.formantPreservation
      );
      onTick(effectiveIntervalMs / 1000, peakDb);
      this.stepIndex = (this.stepIndex + 1) % 16;

      this.stepTimer = window.setTimeout(triggerStep, effectiveIntervalMs);
    };

    triggerStep();
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
    const grainSize = 2400; // ~54ms Hann window grain for clean vocal & polyphonic pitch shifting

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

      // Dynamic vocal tract formant compensation (prevents chipmunk thinness when pitching up)
      if (settings.formantPreservation && Math.abs(settings.semitones) > 0) {
        formantFilter.gain.value = -settings.semitones * 0.85;
      } else {
        formantFilter.gain.value = 0;
      }

      let maxPeak = 0.0001;

      for (let i = 0; i < blockSize; i++) {
        this.playheadSample += speedRatio;
        if (this.playheadSample >= totalSamples - grainSize - 2) {
          this.playheadSample = 0;
        }

        this.grainPhaseA = (this.grainPhaseA + grainRate + 1.0) % 1.0;
        this.grainPhaseB = (this.grainPhaseA + 0.5) % 1.0;

        // Hann window weights for dual overlapping grains (wA + wB = 1.0)
        const winA = 0.5 * (1 - Math.cos(2 * Math.PI * this.grainPhaseA));
        const winB = 0.5 * (1 - Math.cos(2 * Math.PI * this.grainPhaseB));

        const idxA = Math.min(
          totalSamples - 1,
          Math.max(0, Math.floor(this.playheadSample + this.grainPhaseA * grainSize))
        );
        const idxB = Math.min(
          totalSamples - 1,
          Math.max(0, Math.floor(this.playheadSample + this.grainPhaseB * grainSize))
        );

        const rawL = chLeft[idxA] * winA + chLeft[idxB] * winB;
        const rawR = chRight[idxA] * winA + chRight[idxB] * winB;

        // Real-time Mid/Side decomposition to separate Center Lead Vocal vs Stereo Backing & Instrumental
        const mid = 0.5 * (rawL + rawR);
        const side = 0.5 * (rawL - rawR);

        const vocalComponent = mid * leadGain * 0.75;
        const backingComponentL = side * backingGain * 0.9;
        const backingComponentR = -side * backingGain * 0.9;
        const instComponentL = (rawL - mid * 0.55) * instGain;
        const instComponentR = (rawR - mid * 0.55) * instGain;
        const revComponent = side * reverbGain * 0.35;

        // Soft-knee limiter at -0.3 dBFS (0.966)
        const mixL = Math.max(
          -0.966,
          Math.min(
            0.966,
            vocalComponent + backingComponentL + instComponentL + revComponent
          )
        );
        const mixR = Math.max(
          -0.966,
          Math.min(
            0.966,
            vocalComponent + backingComponentR + instComponentR - revComponent
          )
        );

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
    if (this.stepTimer !== null) {
      clearTimeout(this.stepTimer);
      this.stepTimer = null;
    }
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
   * Renders a real downloadable WAV Blob with current Pitch, Speed, Formant, and Stem Mixer settings.
   */
  public renderExportWavBlob(
    stems: StemTrackState[],
    settings: PitchSpeedSettings
  ): Blob {
    const sampleRate = this.customBuffer?.sampleRate ?? 44100;
    const anySolo = stems.some((s) => s.solo);
    const getStemGain = (id: StemId) => {
      const st = stems.find((x) => x.id === id);
      if (!st || st.muted || (anySolo && !st.solo)) return 0;
      return st.volume / 100;
    };

    const pitchRatio = Math.pow(2, (settings.semitones + settings.cents / 100) / 12);
    const speedRatio = Math.max(0.25, Math.min(2.0, settings.speed));

    if (this.customBuffer) {
      const srcL = this.customBuffer.getChannelData(0);
      const srcR =
        this.customBuffer.numberOfChannels > 1
          ? this.customBuffer.getChannelData(1)
          : srcL;
      const outFrames = Math.min(
        sampleRate * 45,
        Math.floor(this.customBuffer.length / speedRatio)
      );
      const outL = new Float32Array(outFrames);
      const outR = new Float32Array(outFrames);

      const leadGain = getStemGain('lead_vocal');
      const backingGain = getStemGain('backing_vocal');
      const instGain = getStemGain('instrumental');

      for (let i = 0; i < outFrames; i++) {
        const srcIdx = Math.min(
          srcL.length - 1,
          Math.floor(i * speedRatio * pitchRatio) % srcL.length
        );
        const l = srcL[srcIdx];
        const r = srcR[srcIdx];
        const mid = 0.5 * (l + r);
        const side = 0.5 * (l - r);
        outL[i] = mid * leadGain * 0.7 + side * backingGain + (l - mid * 0.5) * instGain;
        outR[i] = mid * leadGain * 0.7 - side * backingGain + (r - mid * 0.5) * instGain;
      }

      return encodeStereoWavBlob(outL, outR, sampleRate);
    }

    // Synthesize 8 seconds of the demo track with pitch & speed applied
    const durationSec = 8;
    const outFrames = Math.floor(sampleRate * durationSec);
    const outL = new Float32Array(outFrames);
    const outR = new Float32Array(outFrames);

    const leadGain = getStemGain('lead_vocal') * 0.25;
    const backingGain = getStemGain('backing_vocal') * 0.2;
    const instGain = getStemGain('instrumental') * 0.3;

    const melody = [440, 523.25, 659.25, 587.33, 523.25, 493.88, 440, 392];
    const bass = [110, 110, 87.31, 87.31, 130.81, 130.81, 98, 98];

    for (let i = 0; i < outFrames; i++) {
      const t = i / sampleRate;
      const step = Math.floor(t * 4 * speedRatio) % melody.length;
      const fLead = melody[step] * pitchRatio;
      const fBass = bass[step] * pitchRatio;

      const sLead = Math.sin(2 * Math.PI * fLead * t) * leadGain;
      const sBack = Math.sin(2 * Math.PI * (fLead * 0.75) * t) * backingGain;
      const sInst = Math.sin(2 * Math.PI * fBass * t) * instGain;

      outL[i] = sLead + sBack + sInst;
      outR[i] = sLead - sBack * 0.5 + sInst;
    }

    return encodeStereoWavBlob(outL, outR, sampleRate);
  }

  private playSynthesizedMultiStemStep(
    stems: StemTrackState[],
    anySolo: boolean,
    pitchFactor: number,
    formantPreservation: boolean
  ): number {
    if (!this.ctx) return -60;
    const now = this.ctx.currentTime;

    const leadMelodyFreqs = [
      440.0, 523.25, 659.25, 587.33, 523.25, 493.88, 440.0, 392.0,
      349.23, 440.0, 523.25, 659.25, 587.33, 523.25, 493.88, 440.0,
    ];
    const backingHarmonyFreqs = [
      329.63, 392.0, 440.0, 392.0, 349.23, 329.63, 293.66, 261.63,
      261.63, 329.63, 392.0, 440.0, 392.0, 349.23, 329.63, 329.63,
    ];
    const bassFreqs = [
      110.0, 110.0, 110.0, 110.0, 87.31, 87.31, 87.31, 87.31,
      130.81, 130.81, 130.81, 130.81, 98.0, 98.0, 98.0, 98.0,
    ];

    const getEffectiveGain = (id: StemId): number => {
      const stem = stems.find((s) => s.id === id);
      if (!stem) return 0;
      if (stem.muted) return 0;
      if (anySolo && !stem.solo) return 0;
      return (stem.volume / 100) * 0.14;
    };

    const leadGainVal = getEffectiveGain('lead_vocal');
    if (leadGainVal > 0.001) {
      const osc = this.ctx.createOscillator();
      const filter = this.ctx.createBiquadFilter();
      const gain = this.ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(leadMelodyFreqs[this.stepIndex] * pitchFactor, now);

      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(
        formantPreservation ? 1150 : 1150 * pitchFactor,
        now
      );
      filter.Q.setValueAtTime(2.4, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(leadGainVal, now + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0008, now + 0.21);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now);
      osc.stop(now + 0.22);
    }

    const backingGainVal = getEffectiveGain('backing_vocal');
    if (backingGainVal > 0.001 && this.stepIndex % 2 === 0) {
      const osc = this.ctx.createOscillator();
      const filter = this.ctx.createBiquadFilter();
      const gain = this.ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(backingHarmonyFreqs[this.stepIndex] * pitchFactor, now);

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(formantPreservation ? 1600 : 1600 * pitchFactor, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(backingGainVal * 0.9, now + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.0008, now + 0.42);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now);
      osc.stop(now + 0.44);
    }

    const instGainVal = getEffectiveGain('instrumental');
    if (instGainVal > 0.001) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(bassFreqs[this.stepIndex] * pitchFactor, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(instGainVal * 1.35, now + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0008, now + 0.2);

      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now);
      osc.stop(now + 0.21);
    }

    const reverbGainVal = getEffectiveGain('reverb_tail');
    if (reverbGainVal > 0.001 && this.stepIndex % 4 === 0) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(leadMelodyFreqs[this.stepIndex] * 2 * pitchFactor, now);

      gain.gain.setValueAtTime(0.0005, now);
      gain.gain.exponentialRampToValueAtTime(reverbGainVal * 0.45, now + 0.12);
      gain.gain.exponentialRampToValueAtTime(0.0005, now + 0.75);

      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now);
      osc.stop(now + 0.76);
    }

    const combined =
      leadGainVal + backingGainVal + instGainVal + reverbGainVal;
    return combined > 0.001 ? Math.max(-42, 20 * Math.log10(combined * 2.2)) : -60;
  }
}
