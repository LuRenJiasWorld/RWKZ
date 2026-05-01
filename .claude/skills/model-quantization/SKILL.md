---
name: model-quantization
description: Generate quantized GGUF models from F16 source using llama-quantize. Use when asked to create quantization levels, convert model formats, or regenerate model files.
allowed-tools: Bash, Read
---

# Model Quantization

## Overview

RWKZ supports 6 GGUF quantization levels, all generated from the same F16 source model using llama.cpp's `llama-quantize` tool.

## Source Model

- **Name**: `zhiyuan8/RWKV-v7-0.1B-G1-GGUF`
- **File**: `rwkv7-0.1B-g1-F16.gguf` (369 MB)
- **Architecture**: RWKV v7, 0.1B parameters (vocab=65536, hidden=768, layers=12, head_size=64)

## Naming Convention

```
{model_name}-{quant_lowercase}.gguf
```

Examples:
- `rwkv7-0.1b-g1-q4_0.gguf`
- `rwkv7-0.1b-g1-q8_0.gguf`
- `rwkv7-0.1b-g1-f16.gguf`

The code parses quantization from filename via `parse_quant_from_filename()` in `model.rs`. The function looks for `-q4_0`, `-f16` etc. suffix before `.gguf`.

## Generating All Levels

From the F16 source model, generate the 5 quantized variants:

```bash
for q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0; do
    ./llama-quantize \
        models/rwkv7-0.1b-g1-f16.gguf \
        models/rwkv7-0.1b-g1-${q,,}.gguf \
        $q
done
```

## Expected Output Sizes

| Level | File Size | Command |
|-------|-----------|---------|
| F16 | 369 MB | (source, rename to match convention) |
| Q8_0 | 203 MB | `llama-quantize in.gguf out.gguf Q8_0` |
| Q5_1 | 151 MB | `llama-quantize in.gguf out.gguf Q5_1` |
| Q5_0 | 143 MB | `llama-quantize in.gguf out.gguf Q5_0` |
| Q4_1 | 135 MB | `llama-quantize in.gguf out.gguf Q4_1` |
| Q4_0 | 127 MB | `llama-quantize in.gguf out.gguf Q4_0` |

## Verifying Quantization

After generation, verify each model loads and produces correct output:

```bash
for q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 F16; do
    q_lower=$(echo "$q" | tr '[:upper:]' '[:lower:]')
    cargo run --release -- compress test.txt /tmp/test_${q}.rkz --q $q
    cargo run --release -- decompress /tmp/test_${q}.rkz /tmp/restored_${q}.txt \
        --model models/rwkv7-0.1b-g1-${q_lower}.gguf
    diff test.txt /tmp/restored_${q}.txt && echo "$q: PASS" || echo "$q: FAIL"
done
```

## Model Discovery

The code auto-discovers models by scanning `models/` for `.gguf` files and parsing quantization from filename. No config or registration needed — just place the file in `models/` with the correct naming convention.

Each model gets a SHA256 fingerprint (first 256KB + last 256KB + file size) stored in the compressed file header. This ensures decompression always uses the exact same model.
