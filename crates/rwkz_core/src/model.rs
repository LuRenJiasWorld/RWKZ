use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sha2::{Digest, Sha256};

use crate::arithmetic;
use crate::error::{Error, Result};

// ─── Model management ────────────────────────────────────────────────────────

/// Info about a discovered GGUF model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model base name, e.g. "rwkv7-0.1b-g1"
    pub name: String,
    /// Quantization type: "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16",
    /// "IQ3_XXS", "IQ4_XS", etc.
    pub quantization: String,
    /// Absolute path to the .gguf file
    pub path: PathBuf,
    /// SHA256 fingerprint of the model (covers first 256KB + last 256KB + file size)
    pub fingerprint: [u8; 32],
}

/// Compute a compact SHA256 fingerprint for a GGUF model file.
/// Hashes: first 256KB + last 256KB + file size (as u64 LE).
/// Covers metadata + key weights, sufficient to uniquely identify any model variant.
pub fn compute_fingerprint(path: &Path) -> Result<[u8; 32]> {
    let file_len = std::fs::metadata(path)?.len();
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();

    // Hash first 256 KB (covers GGUF header, metadata, and early weights)
    let head_len = 262144.min(file_len as usize);
    let mut buf = vec![0u8; head_len];
    f.read_exact(&mut buf)?;
    hasher.update(&buf);

    // Hash last 256 KB (output layer + tail)
    let tail_len = 262144.min(file_len as usize);
    if file_len > tail_len as u64 {
        f.seek(SeekFrom::End(-(tail_len as i64)))?;
        let mut buf = vec![0u8; tail_len];
        f.read_exact(&mut buf)?;
        hasher.update(&buf);
    }

    // Hash file size to distinguish same-content different-length files
    hasher.update(&file_len.to_le_bytes());

    let hash: [u8; 32] = hasher.finalize().into();
    Ok(hash)
}

/// All recognized quantization suffixes in GGUF filenames.
/// Ordered by precision (lower rank = smaller/lower quality).
const ALL_QUANT_SUFFIXES: &[&str] = &[
    // Legacy quant types
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16",
    // K-quant types
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_k_s", "q4_k_m",
    "q5_k_s", "q5_k_m",
    "q6_k",
    // IQ (importance-aware) quant types
    "iq1_s", "iq2_xxs", "iq2_xs", "iq2_s",
    "iq3_xxs", "iq3_xs", "iq3_s",
    "iq4_xs", "iq4_nl",
];

/// Parse quantization type from a GGUF filename.
/// Returns the canonical UPPERCASE form (e.g. "IQ3_XXS", "Q4_0", "F16").
fn parse_quant_from_filename(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    let name = lower.trim_end_matches(".gguf");

    for q in ALL_QUANT_SUFFIXES {
        if name.ends_with(&format!("-{q}")) {
            return Some(q.to_uppercase());
        }
    }
    None
}

/// Parse model base name from a GGUF filename.
/// Expected pattern: `<name>-<quant>.gguf` → returns `<name>`
fn parse_model_name(filename: &str) -> String {
    let name = filename.trim_end_matches(".gguf");
    for q in ALL_QUANT_SUFFIXES {
        if let Some(prefix) = name.strip_suffix(&format!("-{q}")) {
            return prefix.to_string();
        }
    }
    // Fallback: use whole filename without extension
    name.to_string()
}

/// Scan a directory for GGUF model files and build ModelInfo for each.
pub fn discover_models(models_dir: &Path) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    let dir = std::fs::read_dir(models_dir)?;

    for entry in dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let lower = fname.to_lowercase();

        if !lower.ends_with(".gguf") {
            continue;
        }

        let quantization = match parse_quant_from_filename(fname) {
            Some(q) => q,
            None => continue, // skip non-standard names
        };

        let name = parse_model_name(fname);

        let fingerprint = compute_fingerprint(&path)?;

        models.push(ModelInfo {
            name,
            quantization,
            path: path.canonicalize().unwrap_or(path),
            fingerprint,
        });
    }

    Ok(models)
}

/// Find a model by exact fingerprint match.
pub fn find_by_fingerprint<'a>(models: &'a [ModelInfo], fp: &[u8; 32]) -> Option<&'a ModelInfo> {
    models.iter().find(|m| m.fingerprint == *fp)
}

/// Quality ordering: higher rank = more precision.
/// IQ (importance-aware) types are ranked above legacy types at
/// comparable bit-widths because importance weighting preserves quality better.
fn quality_rank(q: &str) -> usize {
    match q {
        // 2-bit
        "IQ1_S"   => 0,
        "IQ2_XXS" => 1,
        "IQ2_XS"  => 2,
        "Q2_K"    => 3,
        "IQ2_S"   => 4,
        // 3-bit
        "Q3_K_S"  => 5,
        "IQ3_XXS" => 6,
        "Q3_K_M"  => 7,
        "IQ3_XS"  => 8,
        "Q3_K_L"  => 9,
        "IQ3_S"   => 10,
        // 4-bit
        "Q4_0"    => 11,
        "Q4_K_S"  => 12,
        "Q4_1"    => 13,
        "IQ4_XS"  => 14,
        "Q4_K_M"  => 15,
        "IQ4_NL"  => 16,
        // 5-bit
        "Q5_0"    => 17,
        "Q5_K_S"  => 18,
        "Q5_1"    => 19,
        "Q5_K_M"  => 20,
        // 6-8 bit
        "Q6_K"    => 21,
        "Q8_0"    => 22,
        "F16"     => 23,
        _ => 11, // default to Q4_0 level for unknown types
    }
}

/// Select the model that best matches the requested quantization.
/// Finds the model whose quality rank is closest ≤ the requested rank.
/// Falls back to the highest available if none match.
pub fn select_model<'a>(models: &'a [ModelInfo], q: &str) -> Option<&'a ModelInfo> {
    if models.is_empty() {
        return None;
    }
    let target = quality_rank(&q.to_uppercase());
    // Try exact match first
    if let Some(m) = models.iter().find(|m| m.quantization.eq_ignore_ascii_case(q)) {
        return Some(m);
    }
    // Try closest ≤ target
    let mut best: Option<&ModelInfo> = None;
    let mut best_rank = -1i32;
    for m in models {
        let rank = quality_rank(&m.quantization) as i32;
        if rank <= target as i32 && rank > best_rank {
            best_rank = rank;
            best = Some(m);
        }
    }
    // Fallback to max available
    best.or_else(|| {
        models
            .iter()
            .max_by_key(|m| quality_rank(&m.quantization))
    })
}

// ─── LMPredictor ─────────────────────────────────────────────────────────────

/// LLM predictor wrapping RWKV model for compression.
/// Supports v5, v7, and quantized v7 (GGUF) model architectures.
pub enum LMPredictor {
    V5(V5Predictor),
    V7(V7Predictor),
    V7Quantized(V7QuantizedPredictor),
}

pub struct V5Predictor {
    model: candle_transformers::models::rwkv_v5::Model,
    state: candle_transformers::models::rwkv_v5::State,
    device: Device,
    config: candle_transformers::models::rwkv_v5::Config,
}

pub struct V7Predictor {
    model: candle_transformers::models::rwkv_v7::Model,
    state: candle_transformers::models::rwkv_v7::State,
    device: Device,
    config: candle_transformers::models::rwkv_v7::Config,
}

pub struct V7QuantizedPredictor {
    model: crate::quantized_rwkv_v7::Model,
    state: candle_transformers::models::rwkv_v7::State,
    device: Device,
    config: candle_transformers::models::rwkv_v7::Config,
    pub fingerprint: [u8; 32],
    pub model_name: String,
    pub quantization: String,
}

impl LMPredictor {
    /// Load an RWKV model from a safetensors or pytorch file.
    /// Auto-detects v5 vs v7 based on config.
    pub fn from_file(model_path: &str, config_path: &str) -> Result<Self> {
        let device = Device::Cpu;

        let config_str =
            std::fs::read_to_string(config_path).map_err(|e| Error::Model(e.to_string()))?;

        if config_str.contains("\"version\"") || config_str.contains("\"model_type\": \"rwkv7\"") {
            let config: candle_transformers::models::rwkv_v7::Config =
                serde_json::from_str(&config_str).map_err(|e| Error::Model(e.to_string()))?;
            Self::load_v7(model_path, config, &device)
        } else {
            let config: candle_transformers::models::rwkv_v5::Config =
                serde_json::from_str(&config_str).map_err(|e| Error::Model(e.to_string()))?;
            Self::load_v5(model_path, config, &device)
        }
    }

    /// Load with hardcoded v7 config (for models without config.json).
    pub fn from_file_v7_builtin(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let config = candle_transformers::models::rwkv_v7::Config {
            version: candle_transformers::models::rwkv_v7::ModelVersion::V7,
            vocab_size: 65536,
            hidden_size: 768,
            num_hidden_layers: 12,
            head_size: 64,
            intermediate_size: None,
            rescale_every: 0,
        };
        Self::load_v7(model_path, config, &device)
    }

    /// Load a quantized v7 model from a GGUF file, computing its fingerprint.
    pub fn from_gguf(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        Self::load_gguf(model_path, &device)
    }

    /// Load from GGUF with fingerprint + metadata.
    pub fn from_gguf_with_info(path: &Path) -> Result<(Self, ModelInfo)> {
        let device = Device::Cpu;

        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.gguf");
        let model_name = parse_model_name(fname);
        let quantization = parse_quant_from_filename(fname).unwrap_or_else(|| "Q8_0".to_string());
        let fingerprint = compute_fingerprint(path)?;

        let predictor = Self::load_gguf(path.to_str().unwrap(), &device)?;

        let info = ModelInfo {
            name: model_name,
            quantization,
            path: path.to_path_buf(),
            fingerprint,
        };

        Ok((predictor, info))
    }

    fn load_gguf(model_path: &str, device: &Device) -> Result<Self> {
        let path = Path::new(model_path);
        let config = candle_transformers::models::rwkv_v7::Config {
            version: candle_transformers::models::rwkv_v7::ModelVersion::V7,
            vocab_size: 65536,
            hidden_size: 768,
            num_hidden_layers: 12,
            head_size: 64,
            intermediate_size: None,
            rescale_every: 0,
        };

        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            model_path, device,
        )
        .map_err(|e| Error::Model(e.to_string()))?;

        let model = crate::quantized_rwkv_v7::Model::new(&config, vb)
            .map_err(|e| Error::Model(e.to_string()))?;
        let state = candle_transformers::models::rwkv_v7::State::new(&config, device)
            .map_err(|e| Error::Model(e.to_string()))?;

        // Compute metadata from filename
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.gguf");
        let model_name = parse_model_name(fname);
        let quantization =
            parse_quant_from_filename(fname).unwrap_or_else(|| "Q8_0".to_string());
        let fingerprint = compute_fingerprint(path)?;

        Ok(Self::V7Quantized(V7QuantizedPredictor {
            model,
            state,
            device: device.clone(),
            config,
            fingerprint,
            model_name,
            quantization,
        }))
    }

    /// Get the model fingerprint (available for GGUF-loaded models).
    /// For safetensors models, returns zeros.
    pub fn fingerprint(&self) -> [u8; 32] {
        match self {
            Self::V7Quantized(p) => p.fingerprint,
            _ => [0u8; 32],
        }
    }

    /// Get the model name (e.g. "rwkv7-0.1b-g1").
    pub fn model_name(&self) -> &str {
        match self {
            Self::V7Quantized(p) => &p.model_name,
            _ => "rwkv7-unknown",
        }
    }

    /// Get the quantization type (e.g. "Q8_0", "F16").
    pub fn quantization(&self) -> &str {
        match self {
            Self::V7Quantized(p) => &p.quantization,
            _ => "F32",
        }
    }

    fn load_v5(
        model_path: &str,
        config: candle_transformers::models::rwkv_v5::Config,
        device: &Device,
    ) -> Result<Self> {
        let vb = if model_path.ends_with(".safetensors") || model_path.ends_with(".st") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)
                    .map_err(|e| Error::Model(e.to_string()))?
            }
        } else {
            VarBuilder::from_pth(model_path, DType::F32, device)
                .map_err(|e| Error::Model(e.to_string()))?
        };

        let model = candle_transformers::models::rwkv_v5::Model::new(&config, vb)
            .map_err(|e| Error::Model(e.to_string()))?;
        let state = candle_transformers::models::rwkv_v5::State::new(1, &config, device)
            .map_err(|e| Error::Model(e.to_string()))?;

        Ok(Self::V5(V5Predictor {
            model,
            state,
            device: device.clone(),
            config,
        }))
    }

    fn load_v7(
        model_path: &str,
        config: candle_transformers::models::rwkv_v7::Config,
        device: &Device,
    ) -> Result<Self> {
        let vb = if model_path.ends_with(".safetensors") || model_path.ends_with(".st") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)
                    .map_err(|e| Error::Model(e.to_string()))?
            }
        } else {
            VarBuilder::from_pth(model_path, DType::F32, device)
                .map_err(|e| Error::Model(e.to_string()))?
        };

        let model = candle_transformers::models::rwkv_v7::Model::new(&config, vb)
            .map_err(|e| Error::Model(e.to_string()))?;
        let state = candle_transformers::models::rwkv_v7::State::new(&config, device)
            .map_err(|e| Error::Model(e.to_string()))?;

        Ok(Self::V7(V7Predictor {
            model,
            state,
            device: device.clone(),
            config,
        }))
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::V5(p) => p.config.vocab_size,
            Self::V7(p) => p.config.vocab_size,
            Self::V7Quantized(p) => p.config.vocab_size,
        }
    }

    /// Feed a token and get the CDF for the next token prediction.
    pub fn predict_next_cdf(&mut self, token: u32) -> Result<Vec<u32>> {
        let logits = match self {
            Self::V5(p) => {
                let input = Tensor::new(&[[token]], &p.device)
                    .map_err(|e| Error::Model(e.to_string()))?;
                let logits = p
                    .model
                    .forward(&input, &mut p.state)
                    .map_err(|e| Error::Model(e.to_string()))?;
                logits
                    .squeeze(0)
                    .and_then(|t| t.squeeze(0))
                    .and_then(|t| t.to_dtype(DType::F32))
                    .map_err(|e| Error::Model(e.to_string()))?
            }
            Self::V7(p) => {
                let input = Tensor::new(&[[token]], &p.device)
                    .map_err(|e| Error::Model(e.to_string()))?;
                let logits = p
                    .model
                    .forward(&input, &mut p.state, &[token])
                    .map_err(|e| Error::Model(e.to_string()))?;
                logits.to_dtype(DType::F32).map_err(|e| Error::Model(e.to_string()))?
            }
            Self::V7Quantized(p) => {
                let input = Tensor::new(&[[token]], &p.device)
                    .map_err(|e| Error::Model(e.to_string()))?;
                let logits = p
                    .model
                    .forward(&input, &mut p.state, &[token])
                    .map_err(|e| Error::Model(e.to_string()))?;
                logits.to_dtype(DType::F32).map_err(|e| Error::Model(e.to_string()))?
            }
        };

        let logits_vec = logits
            .to_vec1::<f32>()
            .map_err(|e| Error::Model(e.to_string()))?;

        let probs = arithmetic::softmax(&logits_vec);
        let cdf = arithmetic::build_cdf_from_probs(&probs);

        Ok(cdf)
    }

    /// Reset model state for a new block.
    pub fn reset_state(&mut self) -> Result<()> {
        match self {
            Self::V5(p) => {
                p.state = candle_transformers::models::rwkv_v5::State::new(
                    1, &p.config, &p.device,
                )
                .map_err(|e| Error::Model(e.to_string()))?;
            }
            Self::V7(p) => {
                p.state = candle_transformers::models::rwkv_v7::State::new(&p.config, &p.device)
                    .map_err(|e| Error::Model(e.to_string()))?;
            }
            Self::V7Quantized(p) => {
                p.state = candle_transformers::models::rwkv_v7::State::new(&p.config, &p.device)
                    .map_err(|e| Error::Model(e.to_string()))?;
            }
        }
        Ok(())
    }
}
