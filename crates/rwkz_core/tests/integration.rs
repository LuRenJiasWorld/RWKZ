use rwkz_core::tokenizer::TextTokenizer;
use rwkz_core::model::LMPredictor;

const MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");

#[test]
fn test_tokenizer_roundtrip() {
    let tok = TextTokenizer::from_file(&format!("{MODEL_DIR}/rwkv_vocab_v20230424.json"))
        .expect("Failed to load tokenizer");

    let texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "RWKV is an RNN with transformer-level performance.",
    ];

    for text in texts {
        let tokens = tok.encode(text).expect("encode failed");
        assert!(!tokens.is_empty(), "empty tokens for '{}'", text);
        let decoded = tok.decode(&tokens).expect("decode failed");
        assert_eq!(text, decoded, "roundtrip mismatch for '{}'", text);
    }
}

#[test]
fn test_model_v7_load_and_predict() {
    let mut predictor = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    )
    .expect("Failed to load v7 model");

    assert_eq!(predictor.vocab_size(), 65536);

    // Feed token 0 and get CDF
    let cdf = predictor.predict_next_cdf(0).expect("predict failed");
    assert_eq!(cdf.len(), 65536 + 1, "CDF length mismatch");

    // CDF should be monotonically non-decreasing
    for i in 1..cdf.len() {
        assert!(cdf[i] >= cdf[i - 1], "CDF not monotonic at index {i}");
    }

    println!("CDF max: {}, first 5: {:?}", cdf.last().unwrap(), &cdf[..5]);
}

#[test]
fn test_model_v7_deterministic() {
    let mut p1 = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    )
    .expect("Failed to load model");

    let mut p2 = LMPredictor::from_file_v7_builtin(
        &format!("{MODEL_DIR}/rwkv7-g1d-0.1b.safetensors"),
    )
    .expect("Failed to load model");

    // Same token sequence should produce same CDFs
    for token in [0u32, 100, 500, 1000] {
        let cdf1 = p1.predict_next_cdf(token).expect("p1 predict failed");
        let cdf2 = p2.predict_next_cdf(token).expect("p2 predict failed");
        assert_eq!(cdf1, cdf2, "CDFs differ for token {token} — not deterministic!");
    }
}
