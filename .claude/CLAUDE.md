# RWKZ — Agent Development Guide

## Project Identity

RWKZ is a Rust reimplementation of Fabrice Bellard's LLM-based lossless text compression.
It uses RWKV v7 (0.1B parameters) to predict text token-by-token, then encodes with arithmetic coding.

- **Language**: Rust 2024 edition (requires 1.85+)
- **ML Framework**: [candle](https://github.com/huggingface/candle) 0.10
- **Model Format**: GGUF (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16)
- **License**: MIT

## Architecture Overview

```
  Input ──▶ Tokenizer ──▶ RWKV v7 ──▶ Arithmetic ──▶ .rkz File
  Text        (text→ids)    (predict)     Encoder

  .rkz File ──▶ Arithmetic ──▶ RWKV v7 ──▶ Tokenizer ──▶ Output
                Decoder        (predict)     (ids→text)    Text
```

The pipeline is **serial**: token N+1 can't be predicted without token N, so both encoder and decoder follow the same path.

## Workspace Structure

```
RWKZ/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── rwkz_core/              # Core library
│   │   ├── src/
│   │   │   ├── arithmetic.rs          # 32-bit arithmetic encoder/decoder + softmax + CDF builder
│   │   │   ├── compressor.rs          # Block-based compression pipeline
│   │   │   ├── decompressor.rs        # Block decompression + model fingerprint verification
│   │   │   ├── error.rs               # Error enum (Io, Model, Tokenizer, Format, Compression)
│   │   │   ├── format.rs              # .rkz v2 file format (header 120B, block 24B+data)
│   │   │   ├── model.rs               # LMPredictor enum (V5/V7/V7Quantized), ModelInfo, discovery
│   │   │   ├── quantized_rwkv_v7.rs   # RWKV v7 model for GGUF weights (Q4_0→F16)
│   │   │   └── tokenizer.rs           # RWKV BPE tokenizer wrapper
│   │   └── tests/
│   │       ├── integration.rs         # Model load + determinism (safetensors only)
│   │       └── e2e.rs                 # Full compress→decompress roundtrip
│   └── rwkz_cli/               # CLI binary
│       └── src/main.rs                # clap argument parsing, model discovery, orchestration
├── docs/
│   ├── ARCHITECTURE.md / FORMAT.md / QUANTIZATION.md / BENCHMARK.md / TEST_REPORT.md / BUILDING.md
├── models/                       # GGUF files (gitignored) + tokenizer (committed)
└── .claude/                      # Agent guidance (this file + skills/)
```

## Key Design Decisions

### 1. Dequantize at Load Time

All GGUF weights are dequantized to F32 during model construction (`QLinear::new` calls `.dequantize()` on every weight). Rationale:

- For 0.1B model, F32 BLAS matmul is ~3× faster than keeping Q8_0 quantized tensors and using the quantization kernel on every forward pass
- RAM cost: ~400 MB (acceptable for 0.1B)
- Load time: ~0.3s (one-time cost)

**Never** try to keep weights quantized during inference — it's slower on CPU for models this small.

### 2. GGUF Weight Layout

GGUF stores tensors as `[out_dim, in_dim]` (llama.cpp convention). candle's GGUF loader calls `dimensions.reverse()` on load, then the VarBuilder getters handle it. Our code uses `vb.get_no_shape(name)?.dequantize(dev)?` which returns the dequantized F32 tensor with whatever shape candle decoded it to.

For all QLinear layers: we dequantize the weight and store it as a plain `Tensor`. Forward pass transposes it to `[in_dim, out_dim]` for the matmul: `xs.matmul(&self.weight.t()?)`.

### 3. Model Fingerprint

SHA256(first 256KB + last 256KB + file_size as u64 LE). Computed in `<1ms` (reads only 512KB of a 400MB file). Stored in .rkz v2 header (32 bytes). Decompressor must match fingerprint exactly — prevents using wrong quantization level.

### 4. File Format v2 (120-byte header)

```
MAGIC[4] "RWKZ\x01" | version[2]=2 | model_name[48] | quantization[16] | fingerprint[32] | block_size[4] | reserved[14]
```

Only v2 is supported — no backward compatibility with v1. Each block has CRC32 verification.

### 5. Loop-based Serial Inference

Both compression and decompression run one token at a time through the model — there's no batching or parallel inference. This is inherent to the algorithm: each token's prediction depends on the previous token. The model state (`StatePerLayer::att_kv`, `att_x_prev`, `ffn_x_prev`) is mutated in-place.

### 6. Determinism

The entire pipeline is deterministic:
- Arithmetic coder: 32-bit fixed-precision integer math, no floating point
- Model: F32 on CPU = IEEE 754 deterministic
- Tokenizer: no stochastic sampling

Integration tests verify determinism by loading two model instances and comparing CDF outputs token-by-token.

## Core Data Flow

### Compression (compressor.rs)

```
1. Read block of text (up to block_size, default 50MB)
2. Tokenize: text → Vec<u32>
3. For each token:
   a. model.predict_next_cdf(prev_token) → CDF (len=65537)
   b. encoder.encode_symbol(token, &cdf)
4. Block header: original_size[8] + num_tokens[4] + crc32[4] + compressed_len[8] + data[variable]
5. Write file: header + blocks + EOF marker
```

### Decompression (decompressor.rs)

```
1. Read header → extract fingerprint, model_name, quantization
2. Verify loaded model fingerprint matches header
3. For each block:
   a. decoder = Decoder::new(compressed_data)
   b. For each token: decoder.decode_symbol(&cdf) → token
   c. Verify CRC32
```

### Model Inference (quantized_rwkv_v7.rs)

Each token through the model triggers two sub-layers per block (12 blocks for 0.1B):

**TimeMix (Attention):**
1. Token shift: `xr = x + (x_prev - x) ⊙ x_r` ... (6 lerp vectors from fused lerp tensor)
2. Linear projections: r, k, v = receptance(xr), key(xk), value(xv)
3. LoRA decay: `w = exp(-0.606531 × σ(w0 + tanh(xw·w1)·w2))`
4. Value residual (layer 1+): `v = v + (v_first - v) ⊙ σ(v0 + v1(xv)·v2)`
5. Delta-rule state update: `state = state ⊙ w + state·ab + vk`
6. GroupNorm on output
7. Bonus term: `(r ⊙ k ⊙ r_k).sum() · v`
8. Output projection: `output(gated)`

**ChannelMix (FFN):**
1. Token shift: `k = x + (x_prev - x) ⊙ x_k`
2. Squared ReLU: `key(k).relu().sqr()`
3. Linear value: `value(activated_k)`

## GGUF Tensor Naming Convention

(from `quantized_rwkv_v7.rs` top comment):

| GGUF Tensor | Shape | Role |
|------------|-------|------|
| `token_embd.weight` | `[vocab, hidden]` | Token embedding matrix |
| `token_embd_norm.{weight,bias}` | `[hidden]` | Pre-norm (layer 0, optional) |
| `blk.{i}.time_mix_lerp_fused.weight` | `[hidden, 1, 1, 6]` | Fused lerp (split into x_r/w/k/v/a/g) |
| `blk.{i}.time_mix_w{0,1,2}.weight` | varies | LoRA decay weights |
| `blk.{i}.time_mix_a{0,1,2}.weight` | varies | ICL rate weights |
| `blk.{i}.time_mix_v{0,1,2}.weight` | varies | Value residual weights (layer 1+) |
| `blk.{i}.time_mix_g{1,2}.weight` | varies | Gate weights |
| `blk.{i}.time_mix_k_k.weight` | `[head_size]` | Key normalization scale |
| `blk.{i}.time_mix_k_a.weight` | `[head_size]` | Key ICL scale |
| `blk.{i}.time_mix_r_k.weight` | `[head_size]` | Bonus term scale |
| `blk.{i}.time_mix_{receptance,key,value,output}.weight` | `[hidden, hidden]` | Main linear projections |
| `blk.{i}.time_mix_ln.{weight,bias}` | `[hidden]` | GroupNorm scale/bias |
| `blk.{i}.attn_norm.{weight,bias}` | `[hidden]` | Layer norm 1 |
| `blk.{i}.attn_norm_2.{weight,bias}` | `[hidden]` | Layer norm 2 |
| `blk.{i}.channel_mix_lerp_k.weight` | `[hidden]` | FFN token shift lerp |
| `blk.{i}.channel_mix_{key,value}.weight` | `[intermediate, hidden]` / `[hidden, intermediate]` | FFN linear layers |
| `output_norm.{weight,bias}` | `[hidden]` | Final layer norm |
| `output.weight` | `[vocab, hidden]` | Output projection head |

## Common Pitfalls (Discovered Through Debugging)

### Tensor Math
- `&Tensor + &Tensor` returns `Result<Tensor>` — **always use `?`**
- Use `broadcast_mul/sub/add/div()`, not `*`/`-`/`+` operators
- After `.t()?`, call `.contiguous()?` before `index_select()` (non-contiguous tensors fail index_select)
- `Tensor::get(i)?` on dim-0 slices along the first dimension, returns a subtensor

### Fused Lerp Tensor
`time_mix_lerp_fused.weight` shape is `[hidden_size, 1, 1, 6]`:
```rust
let fused = vb.get_no_shape("time_mix_lerp_fused.weight")?.dequantize(dev)?;
let fused = fused.reshape((hidden_size, 6))?.t()?; // [6, hidden_size]
let x_r = fused.get(0)?; // first lerp vector of size [hidden_size]
```
This is the ONLY GGUF tensor with a non-standard shape.

### Pre-norm (ln0) Application
`token_embd_norm` is the pre-norm applied in `Model::forward` **before** entering blocks. It is applied in `Model::forward`, NOT in `Block::forward`. Applying it in both places would double-normalize — this was a real bug that caused a ~30% quality degradation.

### LoRA Dimension Inference
LoRA intermediate dimensions are inferred from weight shapes at load time:
```rust
let d_decay = vb.get_no_shape("time_mix_w1.weight")?.shape().dims()[0]; // [out_dim, in_dim], so dims[0] = out_dim
```
These are `(d_decay, d_aaa, d_mv, d_gate)` — stored but not currently used (the model construction uses them implicitly through the weight shapes).

### Different Training Runs
The `rwkv7-0.1b-g1` (GGUF) and `rwkv7-g1d-0.1b` (safetensors) are **different training runs** of the same architecture. Their weights differ — embedding values, logits, and therefore CDFs don't match. Don't waste time trying to make them match; they're supposed to differ.

### CDF Monotonicity Edge Case
`build_cdf_from_probs()` with a 65,536-entry probability vector can produce `cdf[65536] < cdf[65535]` due to f64 precision accumulation. This is a pre-existing numerical edge case, not a new bug. It only manifests in the safetensors integration test, not in GGUF usage.

## Model Quantization Levels

6 levels generated from the same F16 source via `llama-quantize`:

| Level | Size | Quality Rank |
|-------|------|-------------|
| Q4_0 | 127 MB | 0 (default) |
| Q4_1 | 135 MB | 1 |
| Q5_0 | 143 MB | 2 |
| Q5_1 | 151 MB | 3 |
| Q8_0 | 203 MB | 4 |
| F16 | 369 MB | 5 |

Filename convention: `{model_name}-{quant_lower}.gguf` (e.g. `rwkv7-0.1b-g1-q4_0.gguf`).
Quantization parsed from filename by `parse_quant_from_filename()`.

## Model Selection Logic

**Compress** (`--q Q5_1`):
1. `discover_models(models_dir)` — scan `models/` for `.gguf` files, parse quant from filename
2. `select_model(models, "Q5_1")` — find best available ≤ requested quality (rank-based)
3. Fall back to highest quality available if no match found

**Decompress**:
1. Read fingerprint from file header
2. `discover_models()` + `find_by_fingerprint()` — exact fingerprint match required
3. If no match found: error listing available models and expected fingerprint

## Test Suite

| Test | Requires | Time |
|------|----------|------|
| `cargo test --lib` (9 tests) | Nothing | <1s |
| `cargo test --test integration` (3 tests) | `rwkv7-g1d-0.1b.safetensors` + tokenizer | ~5min |
| `cargo test --test e2e` (2 tests) | Same as integration + `alice29.txt` | ~20min |

Tests NOT committed to git: model files (too large), test fixtures (runtime artifacts).

## Known Issues

1. `test_model_v7_load_and_predict` fails occasionally — CDF monotonicity at index 65536 (pre-existing precision edge case, not a regression)
2. Q4_0 on inputs >20KB may produce 0-byte output (memory pressure suspected — needs investigation)
3. Only supports RWKV v7 base variant (no v7a/v7b DeepEmbed/DEA)

## Commit Conventions

- All commit messages MUST be in English. No Chinese or other languages in commit titles or bodies.
- Format: `type: description` (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- Include which files changed and what changed in them
