use crate::arithmetic::Decoder;
use crate::error::{Error, Result};
use crate::format::{Block, FileHeader};
use crate::model::LMPredictor;
use crate::tokenizer::TextTokenizer;

use std::io::{Read, Write};

/// Decompress a file compressed by the compressor.
pub struct Decompressor {
    predictor: LMPredictor,
    tokenizer: TextTokenizer,
}

impl Decompressor {
    pub fn new(predictor: LMPredictor, tokenizer: TextTokenizer) -> Self {
        Self {
            predictor,
            tokenizer,
        }
    }

    /// Verify that the loaded model matches the file header.
    /// Returns an error if fingerprints don't match.
    pub fn verify_model_match(&self, header: &FileHeader) -> Result<()> {
        let actual_fp = self.predictor.fingerprint();
        if actual_fp != [0u8; 32] && header.fingerprint != [0u8; 32] {
            if actual_fp != header.fingerprint {
                return Err(Error::Format(format!(
                    "Model fingerprint mismatch:\n  file expects: {} ({})\n  loaded model: {} ({})",
                    hex_fmt(&header.fingerprint),
                    header.quantization,
                    hex_fmt(&actual_fp),
                    self.predictor.quantization(),
                )));
            }
        }
        Ok(())
    }

    /// Decompress from reader and write original text to writer.
    pub fn decompress<R: Read, W: Write>(&mut self, reader: &mut R, writer: &mut W) -> Result<()> {
        // Read and verify header
        let header = FileHeader::read_from(reader)?;
        self.verify_model_match(&header)?;

        loop {
            // Try to read next block; if EOF marker or no more data, stop
            let block = match Block::read_from(reader) {
                Ok(b) => b,
                Err(_) => break,
            };

            if block.original_size == 0 {
                break;
            }

            // Decompress block
            let original_text =
                self.decompress_block(&block.compressed_data, block.num_tokens as usize)?;

            // Verify CRC32
            block.verify_crc32(original_text.as_bytes())?;

            writer.write_all(original_text.as_bytes())?;
        }

        Ok(())
    }

    fn decompress_block(
        &mut self,
        compressed_data: &[u8],
        num_tokens: usize,
    ) -> Result<String> {
        let mut decoder = Decoder::new(compressed_data);
        let mut tokens = Vec::with_capacity(num_tokens);

        // First token: use same initial CDF as encoder
        let initial_cdf = self.predictor.predict_next_cdf(0)?;
        let first_token = decoder.decode_symbol(&initial_cdf) as u32;
        tokens.push(first_token);

        // Subsequent tokens
        for _ in 1..num_tokens {
            let last_token = tokens.last().copied().unwrap_or(0);
            let cdf = self.predictor.predict_next_cdf(last_token)?;
            let token = decoder.decode_symbol(&cdf) as u32;
            tokens.push(token);
        }

        let text = self.tokenizer.decode(&tokens)?;
        Ok(text)
    }
}

fn hex_fmt(bytes: &[u8; 32]) -> String {
    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>() + "..."
}
