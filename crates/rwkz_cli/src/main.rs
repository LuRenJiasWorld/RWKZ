use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::{Path, PathBuf};

use rwkz_core::model;

// ─── HuggingFace auto-download ──────────────────────────────────────────────

/// Default HuggingFace repository hosting the RWKZ models and tokenizer.
const DEFAULT_HF_REPO: &str = "LuRenJiasWorld/RWKV-v7-0.1B-G1-GGUF";

/// Ensure a GGUF model file exists in `models_dir`.
/// If not found locally, downloads it from HuggingFace.
/// Returns the local path to the model file.
fn ensure_model_available(
    model_name: &str,    // e.g. "rwkv7-0.1b-g1"
    quantization: &str,  // e.g. "Q8_0"
    models_dir: &Path,
) -> Result<PathBuf> {
    let quant_lower = quantization.to_lowercase();
    let filename = format!("{model_name}-{quant_lower}.gguf");
    let local_path = models_dir.join(&filename);

    if local_path.exists() {
        return Ok(local_path);
    }

    std::fs::create_dir_all(models_dir)?;

    eprintln!("Model not found locally. Downloading {filename} from HuggingFace...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(DEFAULT_HF_REPO.to_string());

    // hf-hub downloads to cache with resume support, then we copy to models/
    let cached_path = repo.get(&filename)?;
    std::fs::copy(&cached_path, &local_path)?;
    eprintln!("Downloaded to {}", local_path.display());

    Ok(local_path)
}

/// Ensure the tokenizer JSON file exists in `models_dir`.
/// If not found locally, downloads it from HuggingFace.
fn ensure_tokenizer_available(models_dir: &Path) -> Result<PathBuf> {
    let filename = "rwkv_vocab_v20230424.json";
    let local_path = models_dir.join(filename);

    if local_path.exists() {
        return Ok(local_path);
    }

    std::fs::create_dir_all(models_dir)?;

    eprintln!("Tokenizer not found. Downloading {filename} from HuggingFace...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(DEFAULT_HF_REPO.to_string());

    let cached_path = repo.get(filename)?;
    std::fs::copy(&cached_path, &local_path)?;
    eprintln!("Downloaded to {}", local_path.display());

    Ok(local_path)
}

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "rwkz", version, about = "LLM-based text compression")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a text file
    Compress {
        /// Input text file
        input: PathBuf,
        /// Output compressed file
        output: PathBuf,
        /// Quantization level: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16,
        /// IQ3_XXS, IQ4_XS, etc. (default: Q4_0)
        #[arg(long, default_value = "Q4_0")]
        q: String,
        /// Custom model path (overrides auto-discovery)
        #[arg(long)]
        model: Option<PathBuf>,
        /// Tokenizer path (JSON) — auto-downloaded if not provided
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Block size in bytes
        #[arg(long, default_value_t = 50 * 1024 * 1024)]
        block_size: usize,
    },
    /// Decompress a compressed file
    Decompress {
        /// Input compressed file
        input: PathBuf,
        /// Output text file
        output: PathBuf,
        /// Custom model path (overrides auto-discovery)
        #[arg(long)]
        model: Option<PathBuf>,
        /// Tokenizer path (JSON) — auto-downloaded if not provided
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
    /// Show info about a compressed file
    Info {
        /// Compressed file
        input: PathBuf,
    },
}

fn models_dir() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models"
    ))
}

/// Print a short model info line to stderr.
fn print_model_info(info: &model::ModelInfo) {
    let fp_short: String = info.fingerprint.iter().take(6).map(|b| format!("{b:02x}")).collect();
    eprintln!(
        "Using model: {} ({}) [{}...]",
        info.name, info.quantization, fp_short
    );
}

/// Load predictor for compression: auto-discover locally, fall back to HuggingFace download.
///
/// `model_name` is the expected base name (e.g. "rwkv7-0.1b-g1").
fn load_predictor_for_compress(
    q: &str,
    model_path: Option<&PathBuf>,
    model_name: &str,
) -> Result<(rwkz_core::model::LMPredictor, model::ModelInfo)> {
    // Path A: explicit --model flag
    if let Some(path) = model_path {
        let (predictor, info) = model::LMPredictor::from_gguf_with_info(path)?;
        print_model_info(&info);
        return Ok((predictor, info));
    }

    let models_dir = models_dir();

    // Path B: auto-discover locally
    let models = model::discover_models(&models_dir).unwrap_or_default();
    if let Some(info) = model::select_model(&models, q) {
        let (predictor, info) = model::LMPredictor::from_gguf_with_info(&info.path)?;
        print_model_info(&info);
        return Ok((predictor, info));
    }

    // Path C: download from HuggingFace
    let local_path = ensure_model_available(model_name, q, &models_dir)?;
    let (predictor, info) = model::LMPredictor::from_gguf_with_info(&local_path)?;
    print_model_info(&info);
    Ok((predictor, info))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress {
            input,
            output,
            q,
            model,
            tokenizer,
            block_size,
        } => {
            let models_dir = models_dir();
            let tokenizer_path = match tokenizer {
                Some(ref p) => p.clone(),
                None => ensure_tokenizer_available(&models_dir)?,
            };
            eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
            let tokenizer = rwkz_core::tokenizer::TextTokenizer::from_file(
                tokenizer_path.to_str().unwrap(),
            )?;

            let (predictor, model_info) = load_predictor_for_compress(
                &q,
                model.as_ref(),
                "rwkv7-0.1b-g1",
            )?;

            let input_text = std::fs::read_to_string(&input)?;
            eprintln!("Input: {} bytes", input_text.len());

            let mut compressor =
                rwkz_core::compressor::Compressor::new(predictor, tokenizer, model_info)
                    .with_block_size(block_size);

            let out_file = std::fs::File::create(&output)?;
            let mut writer = std::io::BufWriter::new(out_file);

            eprintln!("Compressing...");
            let start = std::time::Instant::now();
            let stats = compressor.compress(input_text.as_bytes(), &mut writer)?;
            writer.flush()?;
            let elapsed = start.elapsed();

            eprintln!("Done in {:.2}s", elapsed.as_secs_f64());
            eprintln!("Original: {} bytes", stats.original_size);
            eprintln!("Compressed: {} bytes", stats.compressed_size);
            eprintln!("Ratio: {:.2} bpb", stats.bits_per_byte());
            eprintln!("Blocks: {}", stats.blocks);
        }
        Commands::Decompress {
            input,
            output,
            model,
            tokenizer,
        } => {
            let models_dir = models_dir();

            let tokenizer_path = match tokenizer {
                Some(ref p) => p.clone(),
                None => ensure_tokenizer_available(&models_dir)?,
            };
            eprintln!("Loading tokenizer from {:?}...", tokenizer_path);
            let tokenizer = rwkz_core::tokenizer::TextTokenizer::from_file(
                tokenizer_path.to_str().unwrap(),
            )?;

            // Read file header first to get model info
            let mut file = std::fs::File::open(&input)?;
            let header = rwkz_core::format::FileHeader::read_from(&mut file)?;
            drop(file);

            eprintln!(
                "File header: {} ({}) version={}",
                header.model_name, header.quantization, header.version
            );

            let predictor = if let Some(ref model_path) = model {
                // Path A: explicit --model flag
                let (predictor, info) = model::LMPredictor::from_gguf_with_info(model_path)?;
                print_model_info(&info);
                predictor
            } else {
                // Path B: try local fingerprint match
                let models = model::discover_models(&models_dir).unwrap_or_default();
                match model::find_by_fingerprint(&models, &header.fingerprint) {
                    Some(info) => {
                        let (predictor, _) = model::LMPredictor::from_gguf_with_info(&info.path)?;
                        print_model_info(info);
                        predictor
                    }
                    None => {
                        // Path C: download from HuggingFace
                        let local_path = ensure_model_available(
                            &header.model_name,
                            &header.quantization,
                            &models_dir,
                        )?;
                        let (predictor, info) = model::LMPredictor::from_gguf_with_info(&local_path)?;
                        print_model_info(&info);
                        predictor
                    }
                }
            };

            let in_file = std::fs::File::open(&input)?;
            let mut reader = std::io::BufReader::new(in_file);

            let out_file = std::fs::File::create(&output)?;
            let mut writer = std::io::BufWriter::new(out_file);

            eprintln!("Decompressing...");
            let start = std::time::Instant::now();
            let mut decompressor =
                rwkz_core::decompressor::Decompressor::new(predictor, tokenizer);
            decompressor.decompress(&mut reader, &mut writer)?;
            writer.flush()?;
            let elapsed = start.elapsed();

            eprintln!("Done in {:.2}s", elapsed.as_secs_f64());
        }
        Commands::Info { input } => {
            let file = std::fs::File::open(&input)?;
            let mut reader = std::io::BufReader::new(file);

            match rwkz_core::format::FileHeader::read_from(&mut reader) {
                Ok(header) => {
                    println!("File: {:?}", input);
                    println!("Version: {}", header.version);
                    if header.version >= 2 {
                        println!("Model: {} ({})", header.model_name, header.quantization);
                        if header.fingerprint != [0u8; 32] {
                            let fp_str: String = header
                                .fingerprint
                                .iter()
                                .map(|b| format!("{b:02x}"))
                                .collect();
                            println!("Fingerprint: {}", fp_str);
                        }
                    } else {
                        println!("Model: {}", header.model_name);
                    }
                    println!("Block size: {} bytes", header.block_size);
                }
                Err(e) => {
                    eprintln!("Error reading file header: {e}");
                }
            }
        }
    }

    Ok(())
}
