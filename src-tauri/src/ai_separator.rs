//! Local AI Stem Separation Engine using ONNX Runtime (`ort` load-dynamic).
//! Supports 3-stage separation:
//! 1. Instrumental vs. All Vocals (`UVR-MDX-NET-Inst_HQ_4.onnx` / `Mel-Band-RoFormer`)
//! 2. Lead Vocal vs. Backing Vocal (`UVR_MDXNET_KARA_2.onnx` / `Mel-Band-RoFormer-Karaoke`)
//! 3. Dry Lead Vocal vs. Reverb Tail (`Reverb_HQ_By_FoxJoy.onnx`)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StftModelConfig {
    pub model_filename: String,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub dim_f: usize,
    pub dim_t: usize,
    pub compensate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationPipelineConfig {
    pub stage1_inst_model: StftModelConfig,
    pub stage2_karaoke_model: StftModelConfig,
    pub stage3_dereverb_model: StftModelConfig,
}

impl SeparationPipelineConfig {
    /// Returns the exact STFT parameters for the fast MDX-Net 3-stage pipeline.
    pub fn default_mdx_pipeline() -> Self {
        Self {
            stage1_inst_model: StftModelConfig {
                model_filename: "UVR-MDX-NET-Inst_HQ_4.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 5120,
                hop_length: 1024,
                dim_f: 2560,
                dim_t: 256,
                compensate: 1.019,
            },
            stage2_karaoke_model: StftModelConfig {
                model_filename: "UVR_MDXNET_KARA_2.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 5120,
                hop_length: 1024,
                dim_f: 2048,
                dim_t: 256,
                compensate: 1.065,
            },
            stage3_dereverb_model: StftModelConfig {
                model_filename: "Reverb_HQ_By_FoxJoy.onnx".to_string(),
                sample_rate: 44_100,
                n_fft: 7680,
                hop_length: 1024,
                dim_f: 3072,
                dim_t: 256,
                compensate: 1.028,
            },
        }
    }
}
