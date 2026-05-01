# Architecture

This document describes the internal architecture of RWKZ — how the pieces fit together and the design decisions behind them.

## Overview

RWKZ is a **two-stage pipeline**: a language model predicts probabilities, and an arithmetic coder uses those probabilities to produce a compact bitstream. The roles reverse for decompression.

```
                     Compression Pipeline
                    ═══════════════════════

  Input ──▶ Tokenizer ──▶ Predictor ──▶ Arithmetic ──▶ .rkz File
  Text        (text→ids)    (RWKV v7)     Encoder

                    Decompression Pipeline
                    ═══════════════════════

  .rkz File ──▶ Arithmetic ──▶ Predictor ──▶ Tokenizer ──▶ Output
                Decoder        (RWKV v7)     (ids→text)    Text
```

The pipeline is **serial by design** — the model cannot predict token N+1 without knowing token N, and the arithmetic coder must use the same CDF for both encoding and decoding.

## Crate Architecture

```
┌─────────────────────┐
│     rwkz_cli      │  CLI binary — argument parsing, orchestration
│  (depends on core)  │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│    rwkz_core      │  Library — all compression logic
│                     │
│  ┌────────────────┐ │
│  │   model.rs     │ │  Model loading, fingerprint, discovery
│  │                 │ │  LMPredictor enum: V5 | V7 | V7Quantized
│  ├────────────────┤ │
│  │ quantized_     │ │  RWKV v7 model impl for GGUF weights
│  │ rwkv_v7.rs     │ │  TimeMix (attention) + ChannelMix (FFN)
│  ├────────────────┤ │
│  │ arithmetic.rs  │ │  32-bit arithmetic encoder/decoder
│  ├────────────────┤ │
│  │ compressor.rs  │ │  Compression orchestration + Stats
│  ├────────────────┤ │
│  │ decompressor.rs│ │  Decompression + model fingerprint matching
│  ├────────────────┤ │
│  │ format.rs      │ │  File format: header v2, block layout, CRC32
│  ├────────────────┤ │
│  │ tokenizer.rs   │ │  Text ←→ token ID conversion (RWKV vocab)
│  ├────────────────┤ │
│  │ error.rs       │ │  Unified Error type
│  └────────────────┘ │
└─────────────────────┘
```

## Key Design Decisions

### 1. Dequantize at Load Time

For the 0.1B parameter model (~400 MB F32), we dequantize all GGUF weights to F32 during model construction. This means:

- **Model loading takes ~0.5 seconds** (reading 200 MB, converting to F32)
- **Inference runs at F32 matmul speed** (candle's optimized BLAS kernel)
- **Memory usage is ~400 MB** (acceptable for 0.1B)

The alternative — keeping weights as Q8_0 and using candle's quantization kernel for every forward pass — was ~3× slower on CPU. For larger models (1B+), the memory/speed trade-off would favor keeping weights quantized.

### 2. Model Fingerprint

Each GGUF file has a unique SHA256 fingerprint computed from:

- First 256 KB (GGUF header + metadata + early layer weights)
- Last 256 KB (output layer + tail weights)
- Total file size (as u64 LE)

This is stored in the `.rkz` file header (32 bytes). During decompression, the loaded model's fingerprint must match the file header's fingerprint exactly — preventing silent corruption from using the wrong model.

The fingerprint computation reads only ~512 KB of a potentially 400 MB file, completing in <1 ms.

### 3. Block-Based Processing

Input is split into **blocks** (default 50 MB). Between blocks, the model state is reset. This design:

- **Limits error propagation** — a single-bit error only corrupts one block
- **Enables parallel decompression** — blocks are independent (future work)
- **Bounds memory usage** — only one block's compressed data is in memory at a time

Each block stores: original size (8 bytes), token count (4 bytes), CRC32 checksum (4 bytes), compressed data length (8 bytes), compressed data (variable).

### 4. Quantized RWKV v7 Model

The `quantized_rwkv_v7.rs` module implements the RWKV v7 architecture specifically for GGUF weight loading. Key details:

**GGUF Weight Layout**: GGUF stores weights as `[out_dim, in_dim]` (llama.cpp convention). candle's GGUF loader calls `dimensions.reverse()`, resulting in `[in_dim, out_dim]` in candle. We use `get_no_shape()` + `QMatMul::from_arc()` to bypass shape checks and handle this directly.

**RWKV v7 Architecture**:
- **TimeMix (Attention)**: Token shift → linear projections → LoRA decay → value residual → delta-rule state update → GroupNorm → bonus term → output projection
- **ChannelMix (FFN)**: Token shift → linear key → squared ReLU activation → linear value

**Fused Lerp**: Six token-shift lerp vectors (`x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g`) are stored contiguously in `time_mix_lerp_fused.weight` with shape `[hidden_size, 1, 1, 6]`. We reshape to `[6, hidden_size]` and split by index.

**LoRA Dimensions**: Four LoRA intermediate dimensions (`d_decay`, `d_aaa`, `d_mv`, `d_gate`) are inferred from weight shapes at load time via `infer_lora_dims()`.

### 5. File Format v2

The `.rkz` file consists of:

```
┌──────────────────────────────────────────┐
│ Header (120 bytes)                       │
│  MAGIC[4]  = "RWKZ\x01"                  │
│  version[2] = 2 (u16 LE)                │
│  model_name[48]   = null-padded ASCII   │
│  quantization[16] = null-padded ASCII   │
│  fingerprint[32]  = SHA256              │
│  block_size[4]    = u32 LE              │
│  reserved[14]     = zeros               │
├──────────────────────────────────────────┤
│ Block 1                                  │
│  original_size[8]   = u64 LE            │
│  num_tokens[4]      = u32 LE            │
│  crc32[4]           = u32 LE            │
│  compressed_len[8]  = u64 LE            │
│  compressed_data[variable]              │
├──────────────────────────────────────────┤
│ Block 2...                               │
├──────────────────────────────────────────┤
│ EOF marker (8 bytes of zeros)            │
└──────────────────────────────────────────┘
```

See [FORMAT.md](FORMAT.md) for the complete specification.

### 6. Determinism

The entire compression/decompression pipeline is **deterministic**:

- Same model + same input → same compressed output
- Same model + same compressed output → same decompressed output
- Arithmetic coding uses fixed-precision 32-bit integer math (no floating point)
- Model inference runs on CPU with F32 precision (IEEE 754 — no non-deterministic GPU operations)
- Tokenizer is deterministic (no stochastic sampling)

Tests in `tests/integration.rs` verify determinism by loading two model instances and confirming their CDFs match bit-for-bit on the same token sequence.

## Dependency Graph

```
rwkz_cli
├── clap (argument parsing)
├── anyhow (error handling)
└── rwkz_core
    ├── candle-core (tensor operations)
    ├── candle-nn (neural network building blocks)
    ├── candle-transformers (RWKV v5/v7 models, tokenizer, GGUF loader)
    ├── crc32fast (block integrity checks)
    ├── sha2 (model fingerprint)
    └── serde / serde_json (config parsing)
```

## Performance Characteristics

Measured on **i7-7700K @ 4.20GHz (4 cores)**, RWKV v7 0.1B model:

| Operation | Time (measured) | Notes |
|---|---|---|
| Model loading (Q8_0 GGUF) | ~0.3 s | 200 MB file → 400 MB F32 in RAM |
| Model fingerprint compute | <1 ms | Reads 512 KB of GGUF file |
| Full model inference (one token) | ~50–60 ms | 0.1B model, CPU-only F32 matmul |
| Arithmetic coding per token | <0.01 ms | Negligible |
| Block CRC32 verification | ~1 ms / MB | |
| 2,000 byte compress (Q8_0) | ~32 s | ~500 tokens × ~60 ms |
| 20,000 byte compress (Q8_0) | ~337 s | ~5,000 tokens × ~60 ms |
| 2,000 byte decompress (Q8_0) | ~30 s | Same forward pass for every token |

The dominant cost is **model inference** (~99% of runtime). The arithmetic coder, CRC32, and I/O are negligible by comparison.

For a 0.1B model on CPU, each token prediction takes ~50–60 ms. At ~4 tokens per word of English text, this translates to ~200–250 ms per word compressed. Larger models or GPU inference would significantly improve bpb at the cost of higher latency.
