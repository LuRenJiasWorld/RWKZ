use std::io::{self, Read, Write};

use crc32fast::Hasher;

use crate::error::{Error, Result};

/// File format magic number: "RWKZ\x01"
pub const MAGIC: [u8; 4] = [b'R', b'W', b'K', b'Z'];

/// Current format version (v2 adds model_name, quantization, fingerprint).
pub const VERSION: u16 = 2;

/// Default block size for compression (50 MB, matching original ts_zip concept).
pub const DEFAULT_BLOCK_SIZE: usize = 50 * 1024 * 1024;

/// Field sizes for v2 header.
pub const MODEL_NAME_LEN: usize = 48;
pub const QUANTIZATION_LEN: usize = 16;
pub const FINGERPRINT_LEN: usize = 32;

/// Compressed file header (v2 format).
/// Total size: 4 + 2 + 48 + 16 + 32 + 4 + 14 = 120 bytes
#[derive(Debug, Clone)]
pub struct FileHeader {
    pub version: u16,
    pub model_name: String,
    pub quantization: String,
    pub fingerprint: [u8; FINGERPRINT_LEN],
    pub block_size: u32,
}

impl FileHeader {
    pub fn new(
        model_name: &str,
        quantization: &str,
        fingerprint: [u8; FINGERPRINT_LEN],
        block_size: usize,
    ) -> Self {
        Self {
            version: VERSION,
            model_name: model_name.to_string(),
            quantization: quantization.to_string(),
            fingerprint,
            block_size: block_size as u32,
        }
    }

    fn write_fixed<W: Write>(w: &mut W, s: &str, len: usize) -> Result<()> {
        let bytes = s.as_bytes();
        let write_len = bytes.len().min(len);
        w.write_all(&bytes[..write_len])?;
        // Zero pad remaining
        if write_len < len {
            w.write_all(&vec![0u8; len - write_len])?;
        }
        Ok(())
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&MAGIC)?;
        w.write_all(&self.version.to_le_bytes())?;
        Self::write_fixed(w, &self.model_name, MODEL_NAME_LEN)?;
        Self::write_fixed(w, &self.quantization, QUANTIZATION_LEN)?;
        w.write_all(&self.fingerprint)?;
        w.write_all(&self.block_size.to_le_bytes())?;
        // Reserved: 14 bytes of zeros
        w.write_all(&[0u8; 14])?;
        Ok(())
    }

    fn read_fixed(r: &mut impl Read, len: usize) -> Result<String> {
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(len);
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(Error::Format(format!(
                "invalid magic: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        let mut version_buf = [0u8; 2];
        r.read_exact(&mut version_buf)?;
        let version = u16::from_le_bytes(version_buf);

        if version != VERSION {
            return Err(Error::Format(format!(
                "unsupported version: {version}, expected {VERSION}"
            )));
        }
        let model_name = Self::read_fixed(r, MODEL_NAME_LEN)?;
        let quantization = Self::read_fixed(r, QUANTIZATION_LEN)?;

        let mut fingerprint = [0u8; FINGERPRINT_LEN];
        r.read_exact(&mut fingerprint)?;

        let mut block_size_buf = [0u8; 4];
        r.read_exact(&mut block_size_buf)?;
        let block_size = u32::from_le_bytes(block_size_buf);

        // Skip reserved bytes
        let mut reserved = [0u8; 14];
        r.read_exact(&mut reserved)?;

        Ok(Self {
            version,
            model_name,
            quantization,
            fingerprint,
            block_size,
        })
    }
}

/// A single compressed block.
#[derive(Debug)]
pub struct Block {
    pub original_size: u64,
    pub num_tokens: u32,
    pub crc32: u32,
    pub compressed_data: Vec<u8>,
}

impl Block {
    pub fn new(original_data: &[u8], compressed_data: Vec<u8>, num_tokens: usize) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(original_data);
        let crc32 = hasher.finalize();
        Self {
            original_size: original_data.len() as u64,
            num_tokens: num_tokens as u32,
            crc32,
            compressed_data,
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.original_size.to_le_bytes())?;
        w.write_all(&self.num_tokens.to_le_bytes())?;
        w.write_all(&self.crc32.to_le_bytes())?;
        w.write_all(&(self.compressed_data.len() as u64).to_le_bytes())?;
        w.write_all(&self.compressed_data)?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf8)?;
        let original_size = u64::from_le_bytes(buf8);

        r.read_exact(&mut buf4)?;
        let num_tokens = u32::from_le_bytes(buf4);

        r.read_exact(&mut buf4)?;
        let crc32 = u32::from_le_bytes(buf4);

        r.read_exact(&mut buf8)?;
        let compressed_len = u64::from_le_bytes(buf8) as usize;

        let mut compressed_data = vec![0u8; compressed_len];
        r.read_exact(&mut compressed_data)?;

        Ok(Self {
            original_size,
            num_tokens,
            crc32,
            compressed_data,
        })
    }

    pub fn verify_crc32(&self, data: &[u8]) -> Result<()> {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let actual = hasher.finalize();
        if actual != self.crc32 {
            return Err(Error::Format(format!(
                "CRC32 mismatch: expected {:08x}, got {:08x}",
                self.crc32, actual
            )));
        }
        Ok(())
    }
}

/// EOF marker (8 zero bytes after last block).
pub const EOF_MARKER: [u8; 8] = [0u8; 8];

/// Write EOF marker.
pub fn write_eof<W: Write>(w: &mut W) -> Result<()> {
    w.write_all(&EOF_MARKER)?;
    Ok(())
}

/// Check for EOF marker.
pub fn is_eof<R: Read>(r: &mut R) -> Result<bool> {
    let mut buf = [0u8; 8];
    match r.read_exact(&mut buf) {
        Ok(()) => {
            if buf == EOF_MARKER {
                Ok(true)
            } else {
                Err(Error::Format("expected EOF marker".into()))
            }
        }
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(true),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let fp = [0xABu8; 32];
        let header = FileHeader::new("rwkv7-0.1b-g1", "Q8_0", fp, DEFAULT_BLOCK_SIZE);
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let read_header = FileHeader::read_from(&mut cursor).unwrap();

        assert_eq!(read_header.version, VERSION);
        assert_eq!(read_header.model_name, "rwkv7-0.1b-g1");
        assert_eq!(read_header.quantization, "Q8_0");
        assert_eq!(read_header.fingerprint, fp);
        assert_eq!(read_header.block_size, DEFAULT_BLOCK_SIZE as u32);
    }

    #[test]
    fn test_block_roundtrip() {
        let original = b"Hello, world! This is a test.";
        let compressed = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let block = Block::new(original, compressed, 7);

        let mut buf = Vec::new();
        block.write_to(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let read_block = Block::read_from(&mut cursor).unwrap();

        assert_eq!(read_block.original_size, original.len() as u64);
        assert_eq!(read_block.num_tokens, 7);
        assert_eq!(read_block.compressed_data, block.compressed_data);
        assert_eq!(read_block.crc32, block.crc32);
    }

    #[test]
    fn test_crc32_verification() {
        let original = b"test data";
        let block = Block::new(original, vec![], 3);
        assert!(block.verify_crc32(original).is_ok());
        assert!(block.verify_crc32(b"wrong data").is_err());
    }

    #[test]
    fn test_invalid_magic() {
        let buf = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0];
        let mut cursor = std::io::Cursor::new(&buf);
        assert!(FileHeader::read_from(&mut cursor).is_err());
    }
}
