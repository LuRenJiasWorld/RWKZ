use rwkz_core::tokenizer::TextTokenizer;
use rwkz_core::model::{LMPredictor, ModelInfo};
use rwkz_core::compressor::Compressor;
use rwkz_core::decompressor::Decompressor;

const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

/// Dummy model info for tests that use safetensors (no GGUF fingerprint available).
fn dummy_model_info() -> ModelInfo {
    ModelInfo {
        name: "rwkv7-g1d-0.1b".into(),
        quantization: "F32".into(),
        path: "models/rwkv7-g1d-0.1b.safetensors".into(),
        fingerprint: [0u8; 32],
    }
}

#[test]
fn test_full_roundtrip() {
    let text = std::fs::read_to_string(&format!("{MODEL_DIR}/../../alice29.txt"))
        .expect("Failed to read alice29.txt");

    // Use first 500 chars for speed
    let text = &text[..500];

    let predictor = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    ).expect("Failed to load model");

    let tokenizer = TextTokenizer::from_file(&format!("{MODEL_DIR}/rwkv_vocab_v20230424.json"))
        .expect("Failed to load tokenizer");

    // Compress
    let mut compressor = Compressor::new(predictor, tokenizer, dummy_model_info());
    let mut compressed = Vec::new();
    let stats = compressor.compress(
        text.as_bytes(),
        &mut compressed,
    ).expect("Compression failed");

    eprintln!("Original: {} bytes, Compressed: {} bytes, bpb: {:.3}",
        stats.original_size, stats.compressed_size, stats.bits_per_byte());

    // Reload model and tokenizer for decompression (fresh state)
    let predictor2 = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    ).expect("Failed to load model 2");
    let tokenizer2 = TextTokenizer::from_file(&format!("{MODEL_DIR}/rwkv_vocab_v20230424.json"))
        .expect("Failed to load tokenizer 2");

    // Decompress
    let mut decompressor = Decompressor::new(predictor2, tokenizer2);
    let mut decompressed = Vec::new();
    decompressor.decompress(
        &mut compressed.as_slice(),
        &mut decompressed,
    ).expect("Decompression failed");

    let decompressed_text = String::from_utf8(decompressed).expect("Invalid UTF-8");
    assert_eq!(text, &decompressed_text, "Roundtrip mismatch!");
}

#[test]
fn test_cdf_determinism() {
    let mut p1 = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    ).expect("Failed to load model 1");

    let mut p2 = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    ).expect("Failed to load model 2");

    let text = std::fs::read_to_string(&format!("{MODEL_DIR}/../../alice29.txt"))
        .expect("Failed to read alice29.txt");
    let tokenizer = TextTokenizer::from_file(&format!("{MODEL_DIR}/rwkv_vocab_v20230424.json"))
        .expect("Failed to load tokenizer");
    let tokens: Vec<u32> = tokenizer.encode(&text).expect("encode failed").into_iter().take(20).collect();

    for (i, &token) in tokens.iter().enumerate() {
        let cdf1 = p1.predict_next_cdf(token).expect("p1 cdf failed");
        let cdf2 = p2.predict_next_cdf(token).expect("p2 cdf failed");

        let mismatches: Vec<(usize, u32, u32)> = cdf1.iter().zip(cdf2.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();

        if !mismatches.is_empty() {
            eprintln!("Token[{}]={}: CDF DIFFERS at {} positions!", i, token, mismatches.len());
            for (idx, a, b) in mismatches.iter().take(5) {
                eprintln!("  CDF[{}]: {} vs {} (diff: {})", idx, a, b, (*a as i64 - *b as i64).abs());
            }
            panic!("CDF determinism check failed at token {}", i);
        }
    }
}
