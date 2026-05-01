---
name: debug-gguf
description: Debug GGUF weight loading issues. Use when model produces unexpected output, weights differ between formats, or you suspect a tensor dimension bug.
allowed-tools: Bash, Read, Write, Edit
---

# Debug GGUF Weight Loading

## When to Use This Skill

- Model output doesn't match expected values
- Embedding values differ between safetensors and GGUF loading
- Model produces garbage or very high bpb
- You suspect a weight loading or dimension bug

## Diagnostic Checklist (priority order)

### 1. Verify It's the Same Training Run

**Different training runs have different weights.** Before debugging, confirm you're comparing the same run:

| Source | File | Model Run |
|--------|------|-----------|
| HuggingFace safetensors | `rwkv7-g1d-0.1b.safetensors` | Run "G1D" |
| HuggingFace GGUF | `rwkv7-0.1b-g1-F16.gguf` | Run "G1" |

These are **different training runs** — embedding values, logits, and CDFs will NOT match. This is expected, not a bug.

The correct comparison is: F16 GGUF vs Q8_0 GGUF (same run, different quantizations). These should produce nearly identical output.

### 2. Compare Raw Weight Bytes (not dequantized values)

```python
# Read raw F16 bytes from GGUF
from gguf import GGUFReader
import numpy as np

reader = GGUFReader('models/rwkv7-0.1b-g1-f16.gguf')
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        data = np.frombuffer(tensor.data, dtype=np.float16)
        print(f"Raw shape: {tensor.tensor_shape}")
        print(f"Values: {data[0:5].astype(np.float32)}")
```

### 3. Check Embedding First

Embeddings are the simplest to verify — they're just a lookup table. Compare `token_embd.weight` values between formats. If these differ, the weight loading is correct but the models are different training runs.

GGUF embedding shape: `[vocab_size, hidden_size]` = `[65536, 768]` for 0.1B model. After candle's `dimensions.reverse()`, it may be `[768, 65536]`.

### 4. Verify Pre-norm is Applied Once

`token_embd_norm` (ln0) should be applied in `Model::forward`, NOT in `Block::forward`. Check the code — if it's applied in both places, the output will be significantly degraded. This was a real bug that caused ~30% quality loss.

In `quantized_rwkv_v7.rs`, the pre-norm is applied at lines 465-467:
```rust
if let (Some(w), Some(b)) = (&self.ln0_weight, &self.ln0_bias) {
    xs = layer_norm(&xs, w, b, 1e-5)?;
}
```
And `Block::forward` does NOT apply ln0 — it starts directly with layer_norm 1 (attn_norm).

### 5. Check Fused Lerp Splitting

The 6 token-shift lerp vectors are stored contiguously in `time_mix_lerp_fused.weight` with shape `[hidden_size, 1, 1, 6]`. Split logic:
```rust
let fused = vb.get_no_shape("time_mix_lerp_fused.weight")?.dequantize(dev)?;
let fused = fused.reshape((hidden_size, 6))?.t()?; // → [6, hidden_size]
let x_r = fused.get(0)?; // [hidden_size]
// ... x_w, x_k, x_v, x_a, x_g
```

Common mistake: `.get(i)?` on `[hidden_size, 6]` returns `[6]` vectors. Must transpose to `[6, hidden_size]` first.

## GGUF Weight Loading Pattern

All weights are loaded with the same pattern:

**1D vectors** (layer norms, bias, scalar params):
```rust
let weight = vb.get_no_shape("tensor_name.weight")?.dequantize(dev)?;
// Returns Tensor of shape [d] — use directly
```

**2D matrices** (linear layers):
```rust
let linear = QLinear::new(&vb, "layer_name")?;
// Internally: vb.get_no_shape("layer_name.weight")?.dequantize(dev)?
// Forward: xs.matmul(&self.weight.t()?)
```

The `QLinear` struct wraps a dequantized F32 weight and transposes in forward pass.

## Common Errors and Fixes

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `QMatMul` shape mismatch | GGUF stores `[out, in]` but QMatMul expects `[in, out]` | Use `dequantize()` + manual transpose instead of `QMatMul` |
| `get_no_shape` returns wrong tensor | Wrong tensor name (missing `.weight` suffix) | GGUF names always end with `.weight` |
| `index_select` fails | Non-contiguous tensor after transpose | Call `.contiguous()?` |
| Layer norm applied twice | ln0 in both Model and Block | Remove from Block::forward |
| Fused lerp gives wrong sizes | `.get(i)?` on wrong axis | Transpose `[hidden, 6]` → `[6, hidden]` first |

## Quick Sanity Test

After loading a GGUF model, verify basic correctness:

```rust
// Feed token 0 and check CDF shape + monotonicity
let cdf = predictor.predict_next_cdf(0)?;
assert_eq!(cdf.len(), 65537);  // vocab_size + 1
for i in 1..cdf.len() {
    assert!(cdf[i] >= cdf[i - 1], "CDF not monotonic at {i}");
}
```

## Comparing Two Models

To verify Q8_0 and F16 produce identical results (same training run):

1. Load both models
2. Feed same sequence of tokens
3. Compare CDFs token-by-token
4. Expect near-identical CDFs (max difference < 0.1%)

If significantly different, the loading code or the quantization has a bug.
