# File Format Specification

This document specifies the **RWKZ v2** compressed file format (`.rkz`).

## Overview

A `.rkz` file contains a header followed by one or more compressed blocks, terminated by an EOF marker.

All multi-byte integers are **little-endian**.

## Header Layout (120 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | `RWKZ\x01` (ASCII: 0x54, 0x53, 0x5A, 0x01) |
| 4 | 2 | `version` | File format version. Currently `2` (u16 LE). |
| 6 | 48 | `model_name` | Null-padded ASCII string identifying the model family (e.g., `"rwkv7-0.1b-g1"`) |
| 54 | 16 | `quantization` | Null-padded ASCII string identifying the quantization type (e.g., `"Q8_0"`, `"F16"`, `"Q4_0"`) |
| 70 | 32 | `fingerprint` | SHA256 model fingerprint (see below) |
| 102 | 4 | `block_size` | Maximum block size in bytes (u32 LE). Default: 52428800 (50 MB) |
| 106 | 14 | `reserved` | Reserved for future use. Must be zero. |

### Magic Number

The magic `RWKZ\x01` identifies the file as a RWKZ compressed file. Version byte `\x01` distinguishes RWKZ from the original rwkz (which uses a different format).

### Model Fingerprint

The fingerprint is a 32-byte SHA256 hash computed as:

```
SHA256(
    first_256KB_of_model_file  ||
    last_256KB_of_model_file   ||
    file_size_as_u64_le
)
```

This uniquely identifies a specific model variant (family + quantization). Decompression MUST fail if the loaded model's fingerprint does not match the header fingerprint.

### Quantization Strings

Standard values:
- `Q4_0`, `Q4_1` — 4-bit quantization
- `Q5_0`, `Q5_1` — 5-bit quantization
- `Q8_0` — 8-bit quantization
- `F16` — half-precision (float16)
- `F32` — full single-precision (legacy safetensors)

## Block Layout (24 bytes + variable data)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | `original_size` | Uncompressed size of this block in bytes (u64 LE) |
| 8 | 4 | `num_tokens` | Number of tokens in this block (u32 LE) |
| 12 | 4 | `crc32` | CRC32 checksum of the original uncompressed data (u32 LE) |
| 16 | 8 | `compressed_len` | Length of compressed data in bytes (u64 LE) |
| 24 | N | `compressed_data` | Arithmetic-coded bit stream |

### Token Count

`num_tokens` tells the decompressor how many tokens to decode. Since arithmetic coding doesn't store an explicit stop symbol, the decoder must know the exact token count in advance.

### CRC32

The CRC32 covers the **original uncompressed text bytes** for this block. It is computed as:

```rust
let mut hasher = crc32fast::Hasher::new();
hasher.update(original_text.as_bytes());
let crc32 = hasher.finalize();
```

After decompression, the output bytes are CRC32-checked. A mismatch indicates data corruption.

### Compressed Data

The compressed data is an arithmetic-coded bit stream. It is **not** self-delimiting — the decoder relies on `num_tokens` to know when to stop.

## EOF Marker (8 bytes)

The file ends with 8 zero bytes. After reading the last block, the reader should expect:

```
00 00 00 00 00 00 00 00
```

If the reader encounters end-of-file before this marker, it is treated as the end of blocks (graceful handling).

## Complete File Layout

```
┌──────────────────────────────────────────────┐
│  Offset 0: Header (120 bytes)                │
├──────────────────────────────────────────────┤
│  Offset 120: Block 1                         │
│    original_size[8]                          │
│    num_tokens[4]                             │
│    crc32[4]                                  │
│    compressed_len[8]                         │
│    compressed_data[compressed_len]           │
├──────────────────────────────────────────────┤
│  Block 2...                                  │
├──────────────────────────────────────────────┤
│  EOF: 8 zero bytes                           │
└──────────────────────────────────────────────┘
```

## Version History

| Version | Changes |
|---------|---------|
| 2 | Added `model_name`, `quantization`, `fingerprint` fields; `model_id` (64 bytes) replaced; header size 65→120 bytes |
| 1 | Initial format: `magic[4] + version[2] + model_id[64] + block_size[4]` (not used in modern RWKZ) |

## Pseudo-code for Reading

```rust
fn read_file(reader: &mut impl Read) -> Result<()> {
    let header = FileHeader::read_from(reader)?;
    assert_eq!(header.version, 2);

    // Verify model matches
    let fp = compute_model_fingerprint(&loaded_model);
    assert_eq!(fp, header.fingerprint);

    loop {
        let block = Block::read_from(reader)?;
        if block.original_size == 0 { break; }  // EOF

        let text = decompress_block(&block.compressed_data, block.num_tokens);
        block.verify_crc32(text.as_bytes())?;
        write(text)?;
    }
}
```

## Security Considerations

- **CRC32 is not cryptographic.** It protects against accidental corruption (bit flips, truncated files), not malicious tampering.
- **Fingerprint enforcement** prevents accidental model mismatch but does not authenticate the model file itself.
- **Block size limits.** The default 50 MB block size bounds memory allocation during decompression.
- **Compressed data length is bounded.** Decompressors should validate `compressed_len` against available input data before allocating.
