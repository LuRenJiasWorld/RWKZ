---
name: benchmark
description: Run compression benchmarks comparing rwkz against traditional compressors. Use when asked to measure performance, compare bpb, or generate benchmark data.
allowed-tools: Bash, Read, Write
---

# Benchmark

## Overview

Benchmark rwkz against traditional compressors (gzip, bzip2, xz, zstd, lz4, brotli) to compare compression ratio and speed. Results go into `docs/BENCHMARK.md`.

## Test Environment

Record the test hardware before running benchmarks:

```bash
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/cpuinfo | grep "cpu cores" | head -1
```

## Full Cross-Tool Comparison

### Step 1: Check Available Tools

```bash
for tool in gzip bzip2 xz zstd lz4 brotli; do
    which $tool && echo "available" || echo "MISSING"
done
```

If any tool is missing, install it via the system package manager.

### Step 2: Run Traditional Compressors

```bash
INPUT=alice29.txt
INPUT_SIZE=$(wc -c < "$INPUT")

for tool in gzip bzip2 xz zstd lz4 brotli; do
    output="/tmp/alice_${tool}.compressed"
    decomp="/tmp/alice_${tool}.restored"

    # Compress (level 6)
    time cat "$INPUT" | $tool -6 > "$output"

    # Decompress
    time cat "$output" | $tool -d > "$decomp"

    # Verify
    cmp -s "$INPUT" "$decomp" && echo "$tool: PASS" || echo "$tool: FAIL"

    # Metrics
    comp_size=$(wc -c < "$output")
    bpb=$(echo "scale=4; $comp_size * 8 / $INPUT_SIZE" | bc)
    echo "$tool  size=${comp_size}B  bpb=${bpb}"
done
```

### Step 3: Run rwkz Per Quantization Level

**Small sample (2,000 bytes)** — all 6 levels for completeness:
```bash
head -c 2000 alice29.txt > /tmp/bench_2k.txt

for Q in Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 F16; do
    q_lower=$(echo "$Q" | tr '[:upper:]' '[:lower:]')
    model="models/rwkv7-0.1b-g1-${q_lower}.gguf"

    cargo run --release -- compress /tmp/bench_2k.txt /tmp/rwkz_tmp_${Q}.rkz --q "$Q"
    cargo run --release -- decompress /tmp/rwkz_tmp_${Q}.rkz /tmp/restored_${Q}.txt \
        --model "$model"

    cmp -s /tmp/bench_2k.txt /tmp/restored_${Q}.txt && echo "$Q: PASS" || echo "$Q: FAIL"
    ls -lh /tmp/rwkz_tmp_${Q}.rkz | awk '{print $5}'
done
```

**Medium sample (20,000 bytes)** — Q8_0 representative:
```bash
head -c 20000 alice29.txt > /tmp/bench_20k.txt
cargo run --release -- compress /tmp/bench_20k.txt /tmp/rwkz_tmp_20k.rkz --q Q8_0
```

### Step 4: Collect and Format Results

Document in this format:
```
| Compressor | Size | bpb | Compress | Decompress |
|-----------|------|-----|----------|------------|
| bzip2 -6 | 43,202 B | 2.27 | 0.02 s | 0.01 s |
...
```

## Time Expectations

On i7-7700K (4 cores):

| Operation | 2,000 bytes | 20,000 bytes | 152,000 bytes |
|-----------|------------|-------------|---------------|
| Q8_0 compress | ~32 s | ~337 s (5.6 min) | ~51 min (extrapolated) |
| Traditional (any) | <0.1 s | <0.1 s | <0.1 s |

**Plan accordingly**: full-file rwkz benchmarks take hours on CPU. Sample sizes of 2-20KB are practical for iteration.

## Updating docs/BENCHMARK.md

After collecting data, update `docs/BENCHMARK.md` with:
1. Hardware specs at the top
2. Traditional compressor results table (full file)
3. rwkz per-level table (sample size noted)
4. Head-to-head comparison at same sample size
5. Extrapolated estimates for full file
6. Speed-compression trade-off visualization
7. Interpretation notes
