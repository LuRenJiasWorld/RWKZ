---
name: compress-decompress
description: Test and debug the compress/decompress pipeline. Use when verifying roundtrip correctness, testing cross-quantization safety, or debugging output issues.
allowed-tools: Bash, Read
---

# Compress / Decompress Pipeline

## Quick Verification

Test that a specific model can compress and decompress losslessly:

```bash
BIN="cargo run --release --"
MODEL_DIR="models"

# Create test input
echo "Hello, World! This is a test of rwkz compression." > /tmp/test.txt

# Compress
$BIN compress /tmp/test.txt /tmp/test.rkz --q Q8_0

# Decompress
$BIN decompress /tmp/test.rkz /tmp/restored.txt --model "$MODEL_DIR/rwkv7-0.1b-g1-q8_0.gguf"

# Verify
diff /tmp/test.txt /tmp/restored.txt && echo "PASS: lossless roundtrip"
```

## Testing All Quantization Levels

```bash
for Q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 F16; do
    q_lower=$(echo "$Q" | tr '[:upper:]' '[:lower:]')
    $BIN compress /tmp/test.txt /tmp/test_${Q}.rkz --q "$Q" 2>&1 | grep -E "Done|Error"
    $BIN decompress /tmp/test_${Q}.rkz /tmp/restored_${Q}.txt \
        --model "$MODEL_DIR/rwkv7-0.1b-g1-${q_lower}.gguf" 2>&1 | grep -E "Done|Error"
    diff /tmp/test.txt /tmp/restored_${Q}.txt && echo "$Q: PASS" || echo "$Q: FAIL"
done
```

## Cross-Quantization Safety Test

Verify that a compressed file requires the exact same model for decompression:

```bash
# Compress with Q4_0
$BIN compress /tmp/test.txt /tmp/test_q4.rkz --q Q4_0

# Try to decompress with Q8_0 (should FAIL)
$BIN decompress /tmp/test_q4.rkz /tmp/restored.txt \
    --model "$MODEL_DIR/rwkv7-0.1b-g1-q8_0.gguf" 2>&1

# Expected output:
# Error: Model fingerprint mismatch:
#   file expects: ... (Q4_0)
#   loaded model: ... (Q8_0)
```

## Inspecting a Compressed File

```bash
$BIN info /tmp/test.rkz

# Example output:
# File: /tmp/test.rkz
# Version: 2
# Model: rwkv7-0.1b-g1 (Q8_0)
# Fingerprint: d3a6c76ce886...
# Block size: 52428800 bytes
```

## Using Custom Model Paths

```bash
# Compress with explicit model (bypasses auto-discovery)
$BIN compress /tmp/test.txt /tmp/test.rkz --model /path/to/custom.gguf

# Decompress with explicit model
$BIN decompress /tmp/test.rkz /tmp/restored.txt --model /path/to/custom.gguf
```

## Expected bpb by Input Size

The 120-byte v2 header adds overhead that dominates on small inputs:

| Input Size | Header Overhead (bpb) | Expected bpb (Q8_0) |
|-----------|----------------------|---------------------|
| 100 B | 9.60 | ~12.0 |
| 500 B | 1.92 | ~5.0 |
| 2,000 B | 0.48 | ~3.1 |
| 20,000 B | 0.05 | ~2.7 |
| 50 MB | <0.001 | ~2.5 (estimated) |

## Custom Tokenizer

```bash
$BIN compress input.txt output.rkz --tokenizer /path/to/vocab.json
$BIN decompress output.rkz restored.txt --tokenizer /path/to/vocab.json
```

Default tokenizer path: `models/rwkv_vocab_v20230424.json`

## Block Size

```bash
# Compress with smaller blocks (useful for testing)
$BIN compress input.txt output.rkz --block-size 1048576  # 1 MB
```

Default: 50 MB (52,428,800 bytes). Smaller blocks mean more frequent state resets, which slightly reduces compression quality.

## Common Issues

### "No GGUF model files found"
Place at least one `.gguf` file in `models/`. Files must follow naming convention `{name}-{quant}.gguf`.

### "Model not found!" during decompress
The file header's fingerprint doesn't match any available model. Either:
- Download the exact model used for compression
- Use `$BIN info` to see which model is needed
- Use `--model <path>` to specify the model file directly

### Compressed output is 0 bytes
May happen with Q4_0 on larger inputs (>20KB). Try Q5_0 or higher quantization.

### "Model fingerprint mismatch"
This is by design — prevents using the wrong quantization level. See cross-quantization safety test above.
