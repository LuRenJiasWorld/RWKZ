# RWKZ — Migration Notes for Claude Code

This document is the **handoff from a previous Claude session** that built this project. Start here in the new session.

## What Is RWKZ

RWKZ (RWKV Zip) is a reimplementation of Fabrice Bellard's ts_zip: LLM-based lossless text compression. A small RWKV v7 language model (0.1B params) predicts the next token, and an arithmetic coder encodes the actual token using that probability distribution.

```
Input Text → Tokenizer → RWKV v7 (predict next token) → Arithmetic Encoder → .rkz file
.rkz file → Arithmetic Decoder → RWKV v7 (predict next token) → Tokenizer → Output Text
```

## Project Layout

```
RWKZ/
├── Cargo.toml                    # Workspace root
├── crates/rwkz_core/             # Core library (all compression logic)
│   ├── src/
│   │   ├── arithmetic.rs         # 32-bit arithmetic encoder/decoder + softmax
│   │   ├── compressor.rs         # Block-based compression pipeline
│   │   ├── decompressor.rs       # Block decompression + fingerprint verification
│   │   ├── error.rs              # Error enum
│   │   ├── format.rs             # .rkz v2 file format (120-byte header)
│   │   ├── model.rs              # LMPredictor enum + ModelInfo + discovery
│   │   ├── quantized_rwkv_v7.rs  # RWKV v7 model for GGUF weights
│   │   └── tokenizer.rs          # RWKV BPE tokenizer wrapper
│   └── tests/
│       ├── integration.rs        # Model load + determinism (needs safetensors)
│       └── e2e.rs                # Full roundtrip (needs safetensors + alice29.txt)
├── crates/rwkz_cli/              # CLI binary
│   └── src/main.rs               # clap parsing, model discovery, orchestration
├── docs/                         # English documentation
│   ├── ARCHITECTURE.md           # Design philosophy, data flow, 6 key decisions
│   ├── FORMAT.md                 # .rkz v2 file format specification
│   ├── QUANTIZATION.md           # Quantization theory + measured data
│   ├── BENCHMARK.md              # Cross-tool compression comparison
│   ├── BUILDING.md               # Build from source + troubleshooting
│   └── TEST_REPORT.md            # Full test suite results
├── models/                       # Model + tokenizer (GGUF gitignored)
├── .claude/                      # Agent guidance
│   ├── CLAUDE.md                 # Project dev guide (architecture, pitfalls, GGUF names)
│   └── skills/                   # 5 agent skills
└── src/main.rs                   # Workspace wrapper
```

## Build

### Prerequisites
- Rust 1.85+ (2024 edition)

### First Build
```bash
cd /mnt/data/Project/RWKZ
cargo build --release
```

The first build downloads all candle dependencies (~5-7 min).
Binary is at `target/release/rwkz`.

### Running Tests
```bash
cargo test --lib                    # 9/9 — no model needed, <1s
cargo test --test integration       # 2/3 — needs safetensors model
cargo test --test e2e               # 2/2 — needs safetensors + alice29.txt
```

### Test Dependencies (not committed)
- `models/rwkv7-g1d-0.1b.safetensors` (365 MB) — for integration/e2e tests
- `alice29.txt` at project root — for e2e tests

## Model Setup

### Download
Models are published on HuggingFace under `zhiyuan8/RWKV-v7-0.1B-G1-GGUF`.

6 quantization levels available:

| File | Size | Quality Rank |
|------|------|-------------|
| `rwkv7-0.1b-g1-q4_0.gguf` | 127 MB | 0 (default) |
| `rwkv7-0.1b-g1-q4_1.gguf` | 135 MB | 1 |
| `rwkv7-0.1b-g1-q5_0.gguf` | 143 MB | 2 |
| `rwkv7-0.1b-g1-q5_1.gguf` | 151 MB | 3 |
| `rwkv7-0.1b-g1-q8_0.gguf` | 203 MB | 4 |
| `rwkv7-0.1b-g1-f16.gguf` | 369 MB | 5 |

Filename convention: `{name}-{quant_lowercase}.gguf`

Place in `models/`. GGUF files are gitignored.
`models/rwkv_vocab_v20230424.json` (1.4 MB) is committed — this is the tokenizer.

### Quick Start
```bash
# Compress (auto-discovers models in models/)
rwkz compress input.txt output.rkz --q Q8_0

# Decompress (auto-matches model by fingerprint)
rwkz decompress output.rkz restored.txt

# Use specific model file
rwkz compress input.txt output.rkz --model ./my-model.gguf

# Show file info
rwkz info output.rkz
```

## Key Architecture Decisions

### 1. Dequantize at Load Time
All GGUF weights dequantized to F32 during model load. For 0.1B:
- ~0.3s load time, ~400 MB RAM
- F32 BLAS matmul ~3× faster than Q8_0 quantization kernel on CPU

### 2. Model Fingerprint
SHA256(first 256KB + last 256KB + file_size as u64 LE). Stored in .rkz v2 header.
Decompressor requires exact fingerprint match — prevents wrong quantization level.

### 3. File Format v2
120-byte header: `RWKZ\x01`(4) + version(2) + model_name(48) + quantization(16) + fingerprint(32) + block_size(4) + reserved(14)

### 4. GGUF Weight Layout
GGUF stores tensors as `[out_dim, in_dim]`. candle reverses dimensions on load.
Our code uses `vb.get_no_shape(name)?.dequantize(dev)?` for all weights.
`QLinear` struct wraps F32 tensor, transposes in `forward()` for matmul.

### 5. Determinism
Entire pipeline deterministic: arithmetic coder uses 32-bit integer math, model is F32 CPU (IEEE 754), tokenizer has no sampling.

## GGUF Tensor Naming Convention

| GGUF Tensor | Shape | Role |
|------------|-------|------|
| `token_embd.weight` | `[vocab, hidden]` | Embedding |
| `token_embd_norm.{weight,bias}` | `[hidden]` | Pre-norm (layer 0) |
| `blk.{i}.time_mix_lerp_fused.weight` | `[hidden, 1, 1, 6]` | Fused lerp (split into 6 vectors) |
| `blk.{i}.time_mix_w{0,1,2}.weight` | varies | LoRA decay |
| `blk.{i}.time_mix_a{0,1,2}.weight` | varies | ICL rate |
| `blk.{i}.time_mix_v{0,1,2}.weight` | varies | Value residual |
| `blk.{i}.time_mix_g{1,2}.weight` | varies | Gate |
| `blk.{i}.time_mix_k_k.weight` | `[head_size]` | Key norm scale |
| `blk.{i}.time_mix_k_a.weight` | `[head_size]` | Key ICL scale |
| `blk.{i}.time_mix_r_k.weight` | `[head_size]` | Bonus term scale |
| `blk.{i}.time_mix_{receptance,key,value,output}.weight` | `[hidden, hidden]` | Linear projections |
| `blk.{i}.time_mix_ln.{weight,bias}` | `[hidden]` | GroupNorm |
| `blk.{i}.attn_norm.{weight,bias}` | `[hidden]` | Layer norm 1 |
| `blk.{i}.attn_norm_2.{weight,bias}` | `[hidden]` | Layer norm 2 |
| `blk.{i}.channel_mix_lerp_k.weight` | `[hidden]` | FFN shift lerp |
| `blk.{i}.channel_mix_{key,value}.weight` | varies | FFN linear |
| `output_norm.{weight,bias}` | `[hidden]` | Final norm |
| `output.weight` | `[vocab, hidden]` | Output head |

## Common Pitfalls

### Tensor Operations
- `&Tensor + &Tensor` returns `Result<Tensor>` — always use `?`
- Use `broadcast_mul/sub/add/div()`, not `*`/`-`/`+` operators
- After `.t()?`, call `.contiguous()?` before `index_select()`

### Fused Lerp Tensor
`time_mix_lerp_fused.weight` shape is `[hidden_size, 1, 1, 6]`:
```rust
let fused = fused.reshape((hidden_size, 6))?.t()?; // [6, hidden_size]
let x_r = fused.get(0)?; // first lerp vector
```

### Pre-norm (ln0)
Applied in `Model::forward`, NOT in `Block::forward`. Double application = ~30% quality loss.

### Different Training Runs
`rwkv7-0.1b-g1` (GGUF) and `rwkv7-g1d-0.1b` (safetensors) are **different runs**. Weights differ. This is expected.

## Known Issues

1. `test_model_v7_load_and_predict` may fail: CDF monotonicity at index 65536 (pre-existing f64 precision edge case)
2. Q4_0 on inputs >20KB may produce 0-byte output (suspected memory issue)
3. Only supports RWKV v7 base variant (no v7a/v7b DeepEmbed/DEA)

## Benchmark Data (i7-7700K, 4 cores)

### Traditional Compressors (alice29.txt, 152 KB)
| Tool | bpb | Time |
|------|-----|------|
| bzip2 -6 | 2.27 | 0.02 s |
| xz -6 | 2.55 | 0.09 s |
| brotli -6 | 2.73 | 0.03 s |
| zstd -6 | 2.79 | 0.02 s |
| gzip -6 | 2.86 | 0.02 s |
| lz4 -6 | 3.37 | 0.01 s |

### RWKZ All Quantization Levels (2 KB sample)
All levels produce near-identical bpb (±0.1). On 20 KB Q8_0: 2.71 bpb, 337 s compress.

## Git History
- `4caa108` fix: binary named rwkz
- `fd4c882` feat: initial commit (32 files, 69K lines)

## Agent Skill Reference

| Skill | When to invoke |
|-------|---------------|
| `build-and-test` | Build, run tests, CI |
| `model-quantization` | Generate quantized GGUF from F16 |
| `debug-gguf` | Debug weight loading / tensor shape issues |
| `benchmark` | Run cross-tool compression benchmarks |
| `compress-decompress` | Test pipeline correctness |

Invoke with the Skill tool: description matches skill name trigger.
Full skill files at `.claude/skills/<name>/SKILL.md`.
