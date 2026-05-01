use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Model(String),
    Tokenizer(String),
    Format(String),
    Compression(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Model(msg) => write!(f, "model error: {msg}"),
            Self::Tokenizer(msg) => write!(f, "tokenizer error: {msg}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Compression(msg) => write!(f, "compression error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
