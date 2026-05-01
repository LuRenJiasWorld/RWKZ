use std::io::{BufReader, Read, Write};

use crate::arithmetic::Encoder;
use crate::error::Result;
use crate::format::{self, Block, FileHeader};
use crate::model::{LMPredictor, ModelInfo};
use crate::tokenizer::TextTokenizer;

/// Compress a text file using LLM-based arithmetic coding.
pub struct Compressor {
    predictor: LMPredictor,
    tokenizer: TextTokenizer,
    model_info: ModelInfo,
    block_size: usize,
}

impl Compressor {
    pub fn new(predictor: LMPredictor, tokenizer: TextTokenizer, model_info: ModelInfo) -> Self {
        Self {
            predictor,
            tokenizer,
            model_info,
            block_size: format::DEFAULT_BLOCK_SIZE,
        }
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Compress text from reader and write compressed output to writer.
    pub fn compress<R: Read, W: Write>(&mut self, reader: R, writer: &mut W) -> Result<Stats> {
        let mut reader = BufReader::new(reader);
        let mut total_original = 0u64;
        let mut total_compressed = 0u64;
        let mut blocks = 0u64;

        // Write header with model info
        let header = FileHeader::new(
            &self.model_info.name,
            &self.model_info.quantization,
            self.model_info.fingerprint,
            self.block_size,
        );
        header.write_to(writer)?;

        loop {
            // Read a block of text
            let mut block_text = String::new();
            let mut buf = [0u8; 8192];
            let mut block_bytes = 0;

            while block_bytes < self.block_size {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        block_text.push_str(&String::from_utf8_lossy(&buf[..n]));
                        block_bytes += n;
                    }
                    Err(e) => return Err(e.into()),
                };
            }

            if block_text.is_empty() {
                break;
            }

            // Tokenize
            let tokens = self.tokenizer.encode(&block_text)?;
            let original_bytes = block_text.len();

            // Compress tokens using LLM prediction + arithmetic coding
            let mut encoder = Encoder::new();

            // First token: use uniform-ish initial CDF (model hasn't seen anything yet)
            let initial_cdf = self.predictor.predict_next_cdf(0)?;
            encoder.encode_symbol(tokens[0] as usize, &initial_cdf);

            // Subsequent tokens: predict based on previous token
            for i in 1..tokens.len() {
                let cdf = self.predictor.predict_next_cdf(tokens[i - 1])?;
                encoder.encode_symbol(tokens[i] as usize, &cdf);
            }
            let compressed_data = encoder.finish();

            // Write block
            let block = Block::new(block_text.as_bytes(), compressed_data, tokens.len());
            block.write_to(writer)?;

            total_original += original_bytes as u64;
            total_compressed += block.compressed_data.len() as u64;
            blocks += 1;
        }

        format::write_eof(writer)?;

        Ok(Stats {
            original_size: total_original,
            compressed_size: total_compressed + format::MAGIC.len() as u64
                + 2 // version
                + format::MODEL_NAME_LEN as u64
                + format::QUANTIZATION_LEN as u64
                + format::FINGERPRINT_LEN as u64
                + 4  // block_size
                + 14 // reserved
                + 8, // eof
            blocks,
        })
    }
}

/// Compression statistics.
#[derive(Debug)]
pub struct Stats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub blocks: u64,
}

impl Stats {
    pub fn bits_per_byte(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (self.compressed_size as f64 * 8.0) / self.original_size as f64
    }
}
