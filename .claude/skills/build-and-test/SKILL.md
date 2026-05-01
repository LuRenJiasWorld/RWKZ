---
name: build-and-test
description: Build, test, and CI workflow for RWKZ. Use when asked to build, run tests, verify compilation, or check for warnings.
allowed-tools: Bash, Read, Edit, Write
---

# Build and Test

## Quick Reference

```bash
# Build release
cargo build --release

# Run all lib tests (fast, no model needed)
cargo test --lib

# Run integration tests (needs safetensors model)
cargo test --test integration

# Run e2e tests (needs safetensors + alice29.txt)
cargo test --test e2e

# Check for warnings (if clippy available)
cargo clippy --all-targets
```

## Prerequisites

- Rust 1.85+ (2024 edition)
- `models/rwkv_vocab_v20230424.json` — tokenizer (committed, 1.4MB)
- `models/rwkv7-g1d-0.1b.safetensors` — F32 model for integration/e2e tests (NOT committed, 365MB)
- `alice29.txt` at project parent directory — test data for e2e tests

## Build Steps

1. Verify Cargo.toml is workspace root (not a crate)
2. Build from workspace root:
   ```bash
   cd RWKZ
   cargo build --release
   ```
3. Binary is at `target/release/rwkz_cli` (the crate is named `rwkz_cli`)

## Test Stages

### Stage 1: Unit Tests (always run first, no dependencies)
```
cargo test --lib
```
Tests: arithmetic roundtrip, softmax, CDF partition, format header/block/CRC32, invalid magic detection.
Expected: 9/9 passed, <1 second.

### Stage 2: Integration Tests (needs safetensors model)
```
cargo test --test integration
```
Tests: tokenizer roundtrip, model v7 load+predict, model v7 determinism.
Expected: 2/3 passed. `test_model_v7_load_and_predict` may fail with "CDF not monotonic at index 65536" — this is a known pre-existing precision edge case.

### Stage 3: E2E Tests (needs safetensors + alice29.txt)
```
cargo test --test e2e
```
Tests: full compress→decompress roundtrip (500 chars), CDF determinism across two model instances.
Expected: 2/2 passed, ~20 minutes total (inference-bound).

## Troubleshooting

### "No GGUF model files found"
The `models/` directory needs at least one `.gguf` file. Download from HuggingFace (`zhiyuan8/RWKV-v7-0.1B-G1-GGUF`) and place in `models/`.

### Integration test fails with file-not-found
Ensure `models/rwkv7-g1d-0.1b.safetensors` exists. This file is NOT committed to git (365MB).

### Build is slow
First build compiles all candle dependencies (~5-7 min). Incremental builds are fast (<10s).
