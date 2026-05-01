# RWKZ

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)

A lossless text compressor based on next-token prediction. A small RWKV v7 language model reads the input token by token, predicting a probability distribution over each subsequent token. An arithmetic coder uses these predictions to encode the text — accurate predictions need fewer bits.

This is a reimplementation of [Fabrice Bellard's ts_zip](https://bellard.org/rwkz/), written in Rust. It uses [candle](https://github.com/huggingface/candle) for model inference and [GGUF](https://github.com/ggml-org/llama.cpp) for model distribution.

---

## How It Works

```
Input Text -> Tokenizer -> RWKV v7 (predict next token) -> Arithmetic Coder -> .rkz file
```

The model outputs a probability for each token in the vocabulary. The arithmetic coder maps these probabilities to a compressed bitstream. Decompression runs the same model with the same state transitions, reconstructing the original tokens from the bitstream.

Both compression and decompression are deterministic: 32-bit integer arithmetic, F32 CPU inference (IEEE 754), and a non-sampling tokenizer guarantee byte-for-byte identical output on any machine.

---

## Quick Start

```bash
# Download and install (Linux and macOS)
curl -fsSL https://raw.githubusercontent.com/LuRenJiasWorld/RWKZ/master/install.sh | bash

# Compress a file (missing models are fetched from HuggingFace automatically)
rwkz compress input.txt output.rkz --q Q4_K_M

# Decompress
rwkz decompress output.rkz restored.txt
```

Or build from source:

```bash
git clone https://github.com/LuRenJiasWorld/RWKZ.git && cd RWKZ
cargo build --release
```

No manual model download is needed. When a model file is not found in `models/`, rwkz downloads it from [HuggingFace](https://huggingface.co/LuRenJiasWorld/RWKV-v7-0.1B-G1-GGUF).

---

## Model Files

All models are derived from the RWKV v7 0.1B G1 checkpoint ([BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1)), quantized to GGUF format with [llama.cpp](https://github.com/ggml-org/llama.cpp). Weights are dequantized to F32 at load time — for a model this small, F32 matrix multiplication on CPU is faster than running the quantization kernel on every forward pass.

The following quantization levels are available. bpb values are measured on a 2,000-byte sample of English prose; due to fixed header overhead, full-file bpb is lower than shown here.

| Level | File | Size | bpb (2KB) | Notes |
|-------|------|------|-----------|-------|
| Q2_K | `rwkv7-0.1b-g1-q2_k.gguf` | 96 MB | 4.12 | Minimum size; measurable quality loss |
| Q3_K_M | `rwkv7-0.1b-g1-q3_k_m.gguf` | 110 MB | 3.38 | |
| Q4_0 | `rwkv7-0.1b-g1-q4_0.gguf` | 127 MB | 3.10 | Legacy format |
| **Q4_K_M** | `rwkv7-0.1b-g1-q4_k_m.gguf` | **127 MB** | **3.02** | **Recommended** |
| Q5_K_M | `rwkv7-0.1b-g1-q5_k_m.gguf` | 143 MB | 3.06 | |
| Q8_0 | `rwkv7-0.1b-g1-q8_0.gguf` | 203 MB | 3.05 | |
| F16 | `rwkv7-0.1b-g1-f16.gguf` | 369 MB | 3.04 | No quantization; reference |

At 4-bit precision and above, compression quality converges: Q4_K_M through F16 differ by less than 0.05 bpb. Q4_K_M is recommended because it matches Q4_0 in size while producing measurably better compression.

Full table and additional levels (Q3_K_S, Q4_1, Q5_0, Q5_1, Q6_K) are documented in [BENCHMARK.md](docs/BENCHMARK.md).

---

## Compression Performance

Test environment: Intel Core i7-7700K (4C/8T), 16 GB RAM. Input: `alice29.txt` (Alice's Adventures in Wonderland, 152 KB). Detailed data in [BENCHMARK.md](docs/BENCHMARK.md).

### 20 KB sample (RWKZ Q8_0 vs traditional compressors)

| Tool | Compressed | bpb | Time |
|------|-----------|-----|------|
| bzip2 -6 | 8,061 B | 3.22 | 0.01 s |
| brotli -6 | 8,886 B | 3.55 | 0.01 s |
| **RWKZ Q8_0** | **6,776 B** | **2.71** | 337 s |

At 20 KB, RWKZ produces ~16% smaller output than bzip2. It is roughly 30,000 times slower on this CPU.

### Full file (traditional compressors only)

| Tool | Compressed | bpb |
|------|-----------|-----|
| bzip2 -6 | 43,202 B | 2.27 |
| xz -6 | 48,500 B | 2.55 |
| brotli -6 | 51,967 B | 2.73 |
| zstd -6 | 53,112 B | 2.79 |
| gzip -6 | 54,423 B | 2.86 |
| lz4 -6 | 64,055 B | 3.37 |

RWKZ has not been run on the full 152 KB file (estimated ~50 minutes on this CPU). The 0.1B model is too small to match bzip2 at full-file scale; larger RWKV models (1.5B or above) would be needed to close that gap.

### Quantization quality convergence

On the 2,000-byte test, 4-bit quantizations and above all land within 3.01-3.06 bpb — indistinguishable from F16. Below 4-bit, compression degrades: Q3_K_M reaches 3.38 bpb and Q2_K reaches 4.12 bpb. The effective floor for this model size is about 4 bits per weight.

---

## Design

- **Dequantize to F32 at load time.** F32 BLAS matrix multiplication on CPU is faster than running the quantization kernel for every forward pass on a model this small. RAM usage is approximately 400 MB.
- **Model fingerprint.** SHA256 of the first and last 256 KB of the model file, plus its size. The decompressor reads the expected fingerprint from the file header and verifies it against the loaded model. This prevents using the wrong quantization level without detection.
- **CRC32 per block.** Input is split into blocks (default 50 MB). Each compressed block includes a CRC32 checksum of the original data, verified during decompression.
- **Deterministic.** The arithmetic coder uses 32-bit integer math. Model inference runs F32 on CPU. The tokenizer has no sampling. Compressing the same file twice with the same model produces the same output.

---

## Project Structure

```
RWKZ/
├── Cargo.toml                 # Workspace manifest
├── crates/
│   ├── rwkz_core/             # Core library (compression engine)
│   │   └── src/
│   │       ├── arithmetic.rs        # 32-bit arithmetic coder and CDF builder
│   │       ├── compressor.rs        # Block-based compression pipeline
│   │       ├── decompressor.rs      # Decompression with fingerprint verification
│   │       ├── format.rs            # .rkz v2 binary file format
│   │       ├── model.rs             # Model discovery, fingerprint, quantization
│   │       ├── quantized_rwkv_v7.rs # RWKV v7 model from GGUF weights
│   │       └── tokenizer.rs         # RWKV BPE tokenizer
│   └── rwkz_cli/              # CLI binary (clap)
│       └── src/main.rs              # compress / decompress / info commands
├── docs/                      # Architecture, benchmarks, format spec
├── scripts/quantization/      # Quantization generation script
└── models/                    # Model files (auto-downloaded, not tracked in git)
```

---

## Building From Source

```
git clone https://github.com/LuRenJiasWorld/RWKZ.git && cd RWKZ
cargo build --release
cargo test --lib                 # 9 unit tests, no model needed
```

Requires Rust 1.85 or later. Model files download automatically on first use. See [BUILDING.md](docs/BUILDING.md) for details.

---

## Documentation

- [ARCHITECTURE](docs/ARCHITECTURE.md) — Design philosophy and data flow
- [BENCHMARK](docs/BENCHMARK.md) — Performance data and cross-tool comparison
- [BUILDING](docs/BUILDING.md) — Build prerequisites and troubleshooting
- [FORMAT](docs/FORMAT.md) — .rkz v2 file format specification
- [QUANTIZATION](docs/QUANTIZATION.md) — Quantization background and results
- [TEST_REPORT](docs/TEST_REPORT.md) — Test suite results

---

## License

Apache License 2.0. Code and model weights share the same license. See [LICENSE](LICENSE).

---

## Acknowledgments

- [Fabrice Bellard](https://bellard.org/) for the original [ts_zip](https://bellard.org/rwkz/) design
- [BlinkDL](https://github.com/BlinkDL) for the RWKV model architecture and training
- [HuggingFace](https://huggingface.co/) for the [candle](https://github.com/huggingface/candle) ML framework
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF format and quantization tools
- [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf), whose contributions were essential to completing this project

![DeepSeek V4 Pro](resources/deepseek-v4-pro.jpg)
