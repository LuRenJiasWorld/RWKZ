---
license: apache-2.0
language:
  - en
  - zh
  - ja
  - ko
  - fr
  - ar
  - es
  - pt
tags:
  - rwkv
  - gguf
  - text-compression
  - lossless-compression
  - rwkz
  - rwkv7
base_model: BlinkDL/rwkv7-g1
pipeline_tag: text-generation
---

# RWKV-v7-0.1B-G1-GGUF

GGUF-quantized RWKV v7 0.1B model files for use with [**rwkz**](https://github.com/LuRenJiasWorld/RWKZ) — an LLM-powered lossless text compressor.

These are the **same weights** as [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1), converted to GGUF format with llama.cpp tooling. No fine-tuning or modification — just quantization for smaller download and faster load.

## Intended Use: Lossless Text Compression

This model powers rwkz, a reimplementation of [Fabrice Bellard's ts_zip](https://bellard.org/rwkz/). The RWKV v7 model predicts the next token in a text, and an arithmetic coder uses those predictions to compress the actual tokens. The better the model predicts, the fewer bits it takes.

```
Text → Tokenizer → RWKV v7 (predict) → Arithmetic Coder → .rkz compressed file
```

**These are not chat / instruct models.** They are base RWKV v7 language models, suitable only for next-token prediction. For chat, use [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1) directly.

## Available Quantization Levels

All generated from the same F16 source using [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-quantize`. Weights are dequantized to F32 at load time for fast CPU inference.

| Level | File | Size | bpb (2KB) | Status |
|-------|------|------|-----------|--------|
| Q2_K | `rwkv7-0.1b-g1-q2_k.gguf` | 96 MB | 4.12 | Minimum viable |
| Q3_K_S | `rwkv7-0.1b-g1-q3_k_s.gguf` | 110 MB | 3.38 | Size-conscious |
| Q3_K_M | `rwkv7-0.1b-g1-q3_k_m.gguf` | 110 MB | 3.38 | Size-conscious |
| Q4_0 | `rwkv7-0.1b-g1-q4_0.gguf` | 127 MB | 3.10 | Legacy baseline |
| Q4_1 | `rwkv7-0.1b-g1-q4_1.gguf` | 135 MB | 3.24 | Legacy |
| **Q4_K_M** | `rwkv7-0.1b-g1-q4_k_m.gguf` | **127 MB** | **3.02** | **⭐ Recommended** |
| Q5_0 | `rwkv7-0.1b-g1-q5_0.gguf` | 143 MB | 3.04 | Legacy |
| Q5_1 | `rwkv7-0.1b-g1-q5_1.gguf` | 151 MB | 3.02 | Legacy |
| Q5_K_M | `rwkv7-0.1b-g1-q5_k_m.gguf` | 143 MB | 3.06 | Quality-focused |
| Q6_K | `rwkv7-0.1b-g1-q6_k.gguf` | 160 MB | 3.01 | High quality |
| Q8_0 | `rwkv7-0.1b-g1-q8_0.gguf` | 203 MB | 3.05 | High quality |
| F16 | `rwkv7-0.1b-g1-f16.gguf` | 369 MB | 3.04 | Reference |

### Which Level Should I Use?

```
bpb (lower = better compression)
4.2 ┤ Q2_K ●  (96 MB)
3.6 ┤
3.4 ┤ Q3_K ●  (110 MB)   ← 15% smaller model, 13% worse bpb than 4-bit
3.2 ┤ Q4_1 ●  (135 MB)
     │ Q4_0 ●  (127 MB)
3.0 ┤ ● ● ● ● ● ●          ← Q4_K_M through F16: all ~3.01–3.06 bpb
     └────────────────────────────────────
         96   110  127  143  203  369 MB
```

**Q4_K_M is the sweet spot** — same size as Q4_0 (127 MB) but achieves 3.02 bpb vs 3.10. Going to higher bit-widths buys almost nothing. Below 4-bit, compression quality degrades noticeably.

### K-Quant vs Legacy

At comparable sizes, K-quant types consistently outperform legacy types:

| Same size | Better choice | Why |
|-----------|--------------|-----|
| Both 127 MB | Q4_K_M over Q4_0 | 3.02 vs 3.10 bpb |
| Both 143 MB | Q5_K_M over Q5_0 | 3.06 vs 3.04 bpb |

Always prefer K-quant when available.

## Compression Performance vs Traditional Tools

Tested on `alice29.txt` (152 KB, English prose, i7-7700K CPU).

### 20 KB sample

| Tool | Size | bpb | Time |
|------|------|-----|------|
| bzip2 -6 | 8,061 B | 3.22 | 0.01 s |
| brotli -6 | 8,886 B | 3.55 | 0.01 s |
| **rwkz Q8_0** | **6,776 B** | **2.71** | 337 s |

rwkz achieves **16% better compression than bzip2** at 20 KB — at the cost of being ~30,000× slower.

### Full file (152 KB, traditional compressors only)

| Tool | Size | bpb |
|------|------|-----|
| **bzip2 -6** | 43,202 B | 2.27 |
| xz -6 | 48,500 B | 2.55 |
| brotli -6 | 51,967 B | 2.73 |
| zstd -6 | 53,112 B | 2.79 |
| gzip -6 | 54,423 B | 2.86 |
| lz4 -6 | 64,055 B | 3.37 |

> rwkz hasn't been run on the full 152 KB file (estimated ~51 min on this CPU). With larger RWKV models (1.5B+), bpb is expected to drop below bzip2.

## Reproducing Quantization

This repository includes everything needed to reproduce all quantization levels from the F16 source:

```bash
# Download the generation script
wget https://huggingface.co/LuRenJiasWorld/RWKV-v7-0.1B-G1-GGUF/resolve/main/scripts/generate_quants.sh

# Run with llama.cpp tools in PATH
LLAMA_CPP_BIN_DIR=/path/to/llama.cpp/build/bin ./generate_quants.sh
```

What's included:
- **`imatrix/rwkv7-0.1b-g1.imatrix`** — Pre-computed importance matrix (WikiText calibration, ~500KB)
- **`calibration/calibration.txt`** — Calibration data used (WikiText excerpt)
- **`scripts/generate_quants.sh`** — One-shot script to regenerate all quantization levels

## Model Details

| Property | Value |
|----------|-------|
| Architecture | RWKV v7 (base variant, no DeepEmbed) |
| Parameters | 191 M |
| Hidden size | 768 |
| Layers | 12 |
| Head size | 64 |
| FFN size | 3,072 |
| Vocab size | 65,536 |
| Context length | 1,048,576 |
| Tokenizer | RWKV BPE (`rwkv_vocab_v20230424.json`) |

## Using with rwkz

```bash
# Compress (model auto-downloads if not cached locally)
rwkz compress input.txt output.rkz --q Q4_K_M

# Decompress (fingerprint-matched automatically)
rwkz decompress output.rkz restored.txt
```

rwkz automatically selects the best available quantization level. Pass `--model` to use a specific GGUF file directly.

## License

Model weights: Apache 2.0 (same as [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1)).

The [rwkz](https://github.com/LuRenJiasWorld/RWKZ) compressor: Apache 2.0 License.

## Acknowledgments

- [BlinkDL](https://github.com/BlinkDL) — RWKV v7 model architecture and training
- [Fabrice Bellard](https://bellard.org/) — Original ts_zip / rwkz concept
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF format, quantization tools
- [bartowski](https://huggingface.co/bartowski) — GGUF distribution best practices
- [ikawrakow](https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp) — Calibration datasets
