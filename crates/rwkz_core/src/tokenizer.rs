use candle_transformers::models::rwkv_v5::Tokenizer as RwkvTokenizer;

use crate::error::{Error, Result};

/// Text tokenizer wrapping RWKV's custom tokenizer.
pub struct TextTokenizer {
    inner: RwkvTokenizer,
}

impl TextTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner =
            RwkvTokenizer::new(path).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
}
