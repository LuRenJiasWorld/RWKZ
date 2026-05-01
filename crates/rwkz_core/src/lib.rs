pub mod arithmetic;
pub mod compressor;
pub mod decompressor;
pub mod error;
pub mod format;
pub mod model;
pub mod tokenizer;

// Quantized RWKV v7 model — only compiled when candle has GGUF support
pub mod quantized_rwkv_v7;
