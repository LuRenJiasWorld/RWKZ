# Benchmark

Test environment and full compression performance comparison.

## Test Hardware

| Component | Detail |
|-----------|--------|
| CPU | Intel Core i7-7700K @ 4.20GHz |
| Cores | 4 physical / 8 logical |
| RAM | 16 GB |
| OS | Linux |
| Rust | 1.85+ (2024 edition) |

## Test Data: alice29.txt

Full text of "Alice's Adventures in Wonderland" (Lewis Carroll).

- **Size**: 152,089 bytes
- **Language**: English prose
- **Structure**: 12 chapters, mixed dialogue and narration

## Traditional Compressors (full file, 152,089 bytes)

All tools at compression level 6.

| Compressor | Size | bpb | Ratio | Compress | Decompress |
|-----------|------|-----|-------|----------|------------|
| **bzip2** | 43,202 B | **2.27** | 28.4% | 0.02 s | 0.01 s |
| xz | 48,500 B | 2.55 | 31.9% | 0.09 s | 0.01 s |
| brotli | 51,967 B | 2.73 | 34.2% | 0.03 s | 0.01 s |
| zstd | 53,112 B | 2.79 | 34.9% | 0.02 s | 0.01 s |
| gzip | 54,423 B | 2.86 | 35.8% | 0.02 s | <0.01 s |
| lz4 | 64,055 B | 3.37 | 42.1% | 0.01 s | <0.01 s |

```
bpb (bits per byte): lower = better compression
Ratio: compressed / original
```

## RWKZ: All Quantization Levels (2,000 byte sample)

Header overhead (~120 bytes / 0.48 bpb) dominates at this size.

| Level | Model Size | Compressed | bpb | Compress | Decompress | Roundtrip |
|-------|-----------|-----------|-----|----------|------------|-----------|
| Q2_K | 96 MB | 1,029 B | 4.12 | 28.4 s | 54.8 s | ✅ |
| Q3_K_S | 110 MB | 845 B | 3.38 | 34.1 s | — | — |
| Q3_K_M | 110 MB | 845 B | 3.38 | 31.0 s | 35.5 s | ✅ |
| Q4_0 | 127 MB | 774 B | 3.10 | 31.7 s | 38.6 s | ✅ |
| Q4_1 | 135 MB | 810 B | 3.24 | 31.5 s | — | — |
| **Q4_K_M** | **127 MB** | **756 B** | **3.02** | 31.2 s | 35.7 s | ✅ |
| Q5_0 | 143 MB | 760 B | 3.04 | 33.9 s | — | — |
| Q5_1 | 151 MB | 756 B | 3.02 | 30.9 s | — | — |
| Q5_K_M | 143 MB | 764 B | 3.06 | 34.8 s | — | — |
| Q6_K | 160 MB | 753 B | 3.01 | 30.3 s | — | — |
| Q8_0 | 203 MB | 762 B | 3.05 | 44.4 s | — | ✅ |
| F16 | 369 MB | 761 B | 3.04 | 54.3 s | 32.7 s | ✅ |

> **Recommendation: Q4_K_M** — same size as Q4_0 (127 MB) but achieves bpb 3.02 vs 3.10.
> Q3_K_M offers 15% smaller files at a 13% bpb penalty.

### IQ Quantization Types

IQ (importance-aware) types are NOT currently supported by candle's GGUF reader:

| Level | Size | Status |
|-------|------|--------|
| IQ1_S | 80 MB | ❌ unsupported dtype |
| IQ2_XXS | 85 MB | ❌ unsupported dtype |
| IQ2_XS | 87 MB | ❌ unsupported dtype |
| IQ3_XXS | 100 MB | ❌ unsupported dtype |
| IQ4_XS | 118 MB | ❌ unsupported dtype |

These require candle to add GGUF dtype support for IQ formats before they can be used. The code already recognizes their filenames (model discovery + quantization parsing).

## K-Quant vs Legacy Quant

At comparable sizes, K-quant types consistently outperform legacy types:

| Same size, better quality |
|---------------------------|
| Q4_K_M (3.02 bpb) > Q4_0 (3.10 bpb) — both 127 MB |
| Q5_K_M (3.06 bpb) ~ Q5_0 (3.04 bpb) — both 143 MB ¹ |

> ¹ Q5_0 slightly edges Q5_K_M in this test; within measurement noise (±0.02).

## Quality vs Size Trade-off

```
bpb
4.2 ─┤
     │ Q2_K ●  (96 MB)
3.6 ─┤
     │
3.4 ─┤  Q3_K ●  (110 MB)
     │
3.2 ─┤ Q4_1 ●  (135 MB)
     │
3.0 ─┤ ● ● ● ● ●  ← Q4_K_M, Q5_0, Q5_1, Q6_K, F16 all at ~3.01-3.06
     │      (all 4-bit+ converge to F16 quality)
     └───┼────┼────┼────┼────┼────
        96   110  127  143  203  369 MB
```

**Convergence point: 4-bit and above.** Q4_K_M through F16 produce near-identical bpb. Going lower to Q3_K trades 13% worse compression for 15% smaller model. Q2_K shows significant degradation.

## 20,000 byte sample — Q8_0

| Metric | Value |
|--------|-------|
| Compressed | 6,776 B |
| bpb | 2.71 |
| Compress time | 337 s (5.6 min) |
| Decompress time | 326 s (5.4 min) |

## Head-to-Head (20 KB, Q8_0 vs brotli)

Same 20 KB input, representative level for both:

| Tool | Size | bpb | Compress | Decompress |
|------|------|-----|----------|------------|
| bzip2 -6 | 8,061 B | 3.22 | 0.01 s | <0.01 s |
| brotli -6 | 8,886 B | 3.55 | 0.01 s | <0.01 s |
| **RWKZ Q8_0** | **6,776 B** | **2.71** | 337 s | 326 s |

> RWKZ Q8_0 achieves ~16% better compression than bzip2 at 20 KB, at ~30,000× the compute cost.

## Extrapolated Estimates

Projecting from the 20 KB Q8_0 result (2.71 bpb), expected performance on the full 152,089 byte file:

| Metric | Estimate |
|--------|----------|
| Compressed size | ~51,600 B |
| bpb | ~2.71 |
| Compress time | ~51 min |
| Decompress time | ~50 min |

The 0.1B model's bpb converges as input grows — the 120-byte header cost amortizes to <0.001 bpb at 50 MB blocks. With a larger 1.5B+ RWKV model, bpb is expected to drop below bzip2 on the full file.

## Speed-Compression Trade-off

```
bzip2  ██████████████████████████████████████████████████▏ 2.27 bpb  0.02 s
xz     ███████████████████████████████████████████████████▎ 2.55 bpb  0.09 s
brotli ████████████████████████████████████████████████████▍ 2.73 bpb  0.03 s
RWKZ █████████████████████████████████████████████████████ 2.71 bpb  337 s *
zstd   █████████████████████████████████████████████████████▌ 2.79 bpb  0.02 s
gzip   ██████████████████████████████████████████████████████▊ 2.86 bpb  0.02 s
lz4    ███████████████████████████████████████████████████████████ 3.37 bpb  0.01 s

* 20 KB sample. Higher bpb = worse compression. Bar width = compressed size ratio.
```

## Model Footprint

| Level | Download | RAM (dequantized) | Suitable for |
|-------|----------|-------------------|-------------|
| Q2_K | 96 MB | ~300 MB | Minimum viable |
| Q3_K_M | 110 MB | ~330 MB | Size-conscious |
| Q4_0 | 127 MB | ~300 MB | Legacy standard |
| **Q4_K_M** | **127 MB** | **~330 MB** | **Recommended** |
| Q5_K_M | 143 MB | ~350 MB | Quality-focused |
| Q8_0 | 203 MB | ~400 MB | High quality |
| F16 | 369 MB | ~400 MB | Reference |

All levels use the same inference engine; quality differences between 4-bit+ levels are negligible (<0.1 bpb).

## Interpretation

1. **The 0.1B model is too small** to beat the best traditional compressors. At 20 KB it edges past brotli (2.71 vs 3.55), but bzip2 still wins on the full 152 KB file (2.27 vs estimated 2.71). A 1.5B model would likely surpass bzip2.

2. **4-bit is the quality floor.** Q4_K_M through F16 produce near-identical bpb (±0.05). Below 4-bit, bpb rises: Q3_K ~3.38 (+13%), Q2_K ~4.12 (+37%).

3. **K-quant beats legacy.** Q4_K_M matches Q4_0 in size (127 MB) but delivers better compression (3.02 vs 3.10 bpb). Always prefer K-quant over legacy at the same bit-width.

4. **Decompress ≈ Compress.** The model does the same forward pass in both directions — there's no "fast decode" mode. This is inherent to LLM-based compression.

5. **CPU-only is slow.** At ~60 ms/token on i7-7700K, a 50 MB block would take ~2 days. GPU inference or larger CPU would make RWKZ practical for real workloads.

6. **bzip2 remains the baseline to beat** for text compression. RWKZ's advantage would come from larger models, not from the 0.1B architecture.

## Generating Quantized Models

See `scripts/quantization/generate_quants.sh` for the full quantization pipeline:

```bash
# All levels
./scripts/quantization/generate_quants.sh

# Single level
./scripts/quantization/generate_quants.sh Q4_K_M
```

Tools needed: `llama-imatrix` + `llama-quantize` from [llama.cpp](https://github.com/ggml-org/llama.cpp).
Calibration data: `wiki.train.raw` from [ikawrakow/validation-datasets-for-llama.cpp](https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp).
