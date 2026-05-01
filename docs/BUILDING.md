# Building From Source

This guide covers building RWKZ from source, including prerequisites, feature flags, and troubleshooting.

## Prerequisites

- **Rust** 1.85 or later (2024 edition required)
- **Cargo** (included with Rust)

Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Quick Build

```bash
git clone <repository-url>
cd RWKZ

# Debug build (faster compile, slower runtime)
cargo build

# Release build (slower compile, optimized runtime)
cargo build --release
```

The binary is at `target/release/rwkz`.

## Workspace Structure

```
RWKZ/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── rwkz_core/        # Compression library
│   │   └── Cargo.toml      #   Dependencies: candle-core, candle-nn, candle-transformers
│   └── rwkz_cli/         # CLI binary
│       └── Cargo.toml      #   Dependencies: rwkz_core, clap
└── models/                 # Model files (not built by Cargo)
```

## Running Tests

```bash
# Unit tests (arithmetic coding, format, etc.)
cargo test --lib

# Integration tests (model loading, determinism)
cargo test --test integration

# End-to-end tests (compress → decompress roundtrip)
cargo test --test e2e

# Run all tests
cargo test
```

### Test Dependencies

The integration and e2e tests require:

1. **Model file**: `models/rwkv7-g1d-0.1b.safetensors` (365 MB, F32 safetensors)
2. **Tokenizer**: `models/rwkv_vocab_v20230424.json` (1.4 MB)
3. **Test data**: `alice29.txt` at the project root

If any of these are missing, the affected tests will fail with a file-not-found error.

## Feature Flags

Currently, RWKZ has no feature flags. All functionality is always compiled.

### Candle Features

candle is used with default features, which includes:
- CPU BLAS acceleration (no GPU support)
- GGUF format parsing (via `candle-transformers`)
- Quantization types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

## Dependency Overview

| Dependency | Version | Purpose |
|---|---|---|
| `candle-core` | 0.10 | Tensor operations |
| `candle-nn` | 0.10 | Neural network layers (Embedding, Module trait) |
| `candle-transformers` | 0.10 | RWKV v5/v7 models, GGUF VarBuilder, tokenizer |
| `clap` | 4 | CLI argument parsing |
| `crc32fast` | 1.4 | Block integrity checks |
| `sha2` | 0.10 | Model fingerprint (SHA256) |
| `serde` / `serde_json` | 1 | Config deserialization |
| `anyhow` | 1 | Error handling in CLI |

## Platform Support

RWKZ is designed for **Linux x86_64**. It should work on:

- **Linux** (primary target) — full support
- **macOS** (x86_64 / ARM) — should work, not tested
- **Windows** — may work with MSVC toolchain, not tested

### CPU Requirements

- x86_64 with SSE4.2 or later
- No GPU required — everything runs on CPU
- ~500 MB RAM minimum (for 0.1B model + inference state)

## Model File Setup

RWKZ discovers models by scanning `models/` for `.gguf` files. Files must follow the naming convention:

```
<model-name>-<quantization>.gguf
```

Examples:
- `rwkv7-0.1b-g1-q4_0.gguf`
- `rwkv7-0.1b-g1-q8_0.gguf`
- `rwkv7-0.1b-g1-f16.gguf`

### Downloading Models

Pre-quantized models can be downloaded from HuggingFace:

```bash
# Example: download F16 model (may be large)
# Source: zhiyuan8/RWKV-v7-0.1B-G1-GGUF

# Place in models/ directory
mv rwkv7-0.1B-g1-F16.gguf models/rwkv7-0.1b-g1-f16.gguf
```

### Generating Quantized Models From F16

If you have the F16 source model, generate lower-precision versions:

```bash
# Requires llama.cpp's llama-quantize
for q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; do
    ./llama-quantize \
        models/rwkv7-0.1b-g1-f16.gguf \
        models/rwkv7-0.1b-g1-${q,,}.gguf \
        $q
done
```

## Troubleshooting

### "No GGUF model files found"

Make sure you have at least one `.gguf` file in the `models/` directory:

```bash
ls models/*.gguf
```

### "Model not found!" on decompress

The compressed file's fingerprint doesn't match any available model. Either:
- Download the model that was used for compression
- Use `rwkz info <file>` to see which model is needed
- Use `--model <path>` to specify the exact model file

### Build errors about missing dependencies

```bash
# Clean build cache
cargo clean
cargo build
```

### Slow performance

- Build with `--release` flag — debug builds are ~10× slower
- Use a smaller quantization (Q4_0) if memory is tight
- Check that your CPU supports SSE4.2 (virtually all x86_64 CPUs do)
