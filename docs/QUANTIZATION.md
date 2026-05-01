# Quantization Guide

This document explains how model quantization works in RWKZ, the available levels, and the quality/speed trade-offs.

## What Is Quantization?

Quantization reduces the precision of model weights to save storage space. Instead of storing each weight as a 32-bit float (F32), we store it at lower precision:

| Format | Bits per weight | Example weight = 0.123456 |
|--------|----------------|---------------------------|
| F32 | 32 | 0.123456 (exact) |
| F16 | 16 | 0.12347 (3.3 decimal digits) |
| Q8_0 | 8 | 0.1235 (block-scale 8-bit) |
| Q5_1 | 5 | ~0.12 (block-scale 5-bit) |
| Q4_0 | 4 | ~0.12 (block-scale 4-bit) |

The key insight: language model weights are **noise-tolerant**. Small perturbations from quantization produce nearly identical predictions.

## Available Levels

All levels are generated from the same source model (`rwkv7-0.1b-g1`, F16 GGUF). The **weights are identical** across levels — only the precision differs.

| Level | Model Size | Relative to F16 | Best For |
|-------|-----------|-----------------|----------|
| **Q4_0** | 127 MB | 34% | Smallest distribution, good quality |
| **Q4_1** | 135 MB | 37% | Q4_0 + minimum value preservation |
| **Q5_0** | 143 MB | 39% | 5-bit weights |
| **Q5_1** | 151 MB | 41% | 5-bit + minimum values |
| **Q8_0** | 203 MB | 55% | Near-lossless, recommended general use |
| **F16** | 369 MB | 100% | Reference / maximum quality |

## How Quantization Levels Are Generated

We use llama.cpp's `llama-quantize` tool to convert the F16 source model:

```bash
# Download source F16 GGUF from HuggingFace
# Model: zhiyuan8/RWKV-v7-0.1B-G1-GGUF

# Generate all levels
for q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; do
    llama-quantize \
        models/rwkv7-0.1b-g1-F16.gguf \
        models/rwkv7-0.1b-g1-${q,,}.gguf \
        $q
done
```

## How RWKZ Uses Quantized Weights

RWKZ dequantizes all weights to F32 at load time:

```
GGUF file (Q8_0)  ──▶  Dequantize  ──▶  F32 Tensor  ──▶  F32 matmul

Model load time: ~0.5s  (one-time cost)
Inference: F32 speed   (every forward pass)
Memory: ~400 MB        (F32 weights for 0.1B model)
```

**Why not run quantized inference?** For a 0.1B parameter model, F32 BLAS matmul on CPU is ~3× faster than the Q8_0 quantization kernel. The memory cost (~400 MB) is acceptable. For larger models (1B+), keeping weights quantized during inference would be preferable.

## Model Selection Behavior

### At Compress Time

```
rwkz compress input.txt out.rkz --q Q5_1
```

1. Scans `models/` for all `.gguf` files
2. Parses quantization from filename (e.g., `rwkv7-0.1b-g1-q5_1.gguf` → `Q5_1`)
3. Finds the **best available model ≤ requested quality**:
   - If `Q5_1` exists → uses it
   - If only `Q4_0`, `Q4_1`, `Q8_0` exist → uses `Q4_1` (closest ≤ Q5_1)
   - Falls back to the highest quality available if none match
4. Writes its fingerprint to the `.rkz` file header

### At Decompress Time

```
rwkz decompress out.rkz restored.txt
```

1. Reads fingerprint from `.rkz` header
2. Scans `models/` for all `.gguf` files
3. Finds the model whose fingerprint matches exactly
4. Loads it and verifies the fingerprint matches again
5. If no match → error with available models and expected fingerprint

### Custom Model Path

Both compress and decompress accept `--model <PATH>`:

```bash
rwkz compress input.txt out.rkz --model ~/custom-model.gguf
rwkz decompress out.rkz restored.txt --model ~/custom-model.gguf
```

When `--model` is provided, auto-discovery is skipped and the specified file is loaded directly. Fingerprint is still computed and stored/verified.

## Cross-Quantization Safety

The fingerprint mechanism **prevents** using the wrong quantization for decompression:

```
$ rwkz compress input.txt out.rkz --q Q4_0
  (writes Q4_0 fingerprint to header)

$ rwkz decompress out.rkz restored.txt
  Error: Model not found!
    File requires: rwkv7-0.1b-g1 (Q4_0), fp=3a7f21b9...
    Available:
      Q8_0 models/rwkv7-0.1b-g1-q8_0.gguf fp=8b2c...
      F16 models/rwkv7-0.1b-g1-f16.gguf fp=1e9d...
```

This ensures compressed files are always decompressed with the exact same model that compressed them.

## Expected Quality Impact

For text compression, the quality impact of quantization is minimal because:

1. **Compression is self-consistent.** The same model is used for both encoding and decoding, so the arithmetic coder never observes a "wrong" probability — it always encodes with the distribution the decoder will also use.

2. **RWKV is robust.** The delta-rule state update and LoRA parameterization make the model less sensitive to individual weight perturbations than transformer architectures.

3. **Softmax saturates.** Small weight differences spread across 65,536 vocabulary entries typically change token rankings minimally.

### Measured bpb (bits per byte)

Tested on **i7-7700K @ 4.20GHz (4 cores)** with `alice29.txt`:

**Small sample (2,000 bytes)** — header overhead dominates:

| Level | Compressed Size | bpb | Compress Time | Roundtrip |
|-------|----------------|-----|--------------|-----------|
| Q4_0 | 798 B | 3.19 | ~32 s | PASS |
| Q4_1 | 834 B | 3.34 | ~34 s | PASS |
| Q5_0 | 784 B | 3.14 | ~34 s | PASS |
| Q5_1 | 780 B | 3.12 | ~38 s | PASS |
| Q8_0 | 786 B | 3.14 | ~32 s | PASS |
| F16 | 785 B | 3.14 | ~30 s | PASS |

**Medium sample (20,000 bytes)** — model warms up:

| Level | Compressed Size | bpb | Compress Time | Roundtrip |
|-------|----------------|-----|--------------|-----------|
| Q8_0 | 6,776 B | 2.71 | ~337 s | PASS |

> **Note**: On 2 KB inputs, the 120-byte v2 header accounts for ~0.48 bpb of overhead. As input size increases toward 50 MB blocks, the amortized overhead becomes negligible (<0.001 bpb).

### Analysis

- All 6 quantization levels produce **nearly identical bpb** within ±0.1 of each other — confirming that quantization has negligible impact on compression quality
- **Decompression speed** is nearly identical to compression speed (the model runs the same forward pass in both directions)
- The 0.1B model is **too small** to match the best traditional compressors on large inputs. Larger 1.5B+ RWKV models would likely surpass bzip2.

For full cross-tool comparison with gzip/bzip2/xz/brotli/zstd/lz4, see [BENCHMARK.md](BENCHMARK.md).
