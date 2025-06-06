//! Error types for the rust_mem_vid library

use thiserror::Error;

/// Result type alias for the library
pub type Result<T> = std::result::Result<T, MemvidError>;

/// Main error type for the rust_mem_vid library
#[derive(Error, Debug)]
pub enum MemvidError {
    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// QR code encoding/decoding errors
    #[error("QR code error: {0}")]
    QrCode(String),

    /// Video processing errors
    #[error("Video error: {0}")]
    Video(String),

    /// FFmpeg errors (disabled in simplified version)
    #[error("FFmpeg error: {0}")]
    FfmpegSimplified(String),

    /// OpenCV errors (disabled in simplified version)
    #[error("OpenCV error: {0}")]
    OpenCvSimplified(String),

    /// Embedding model errors
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Candle (ML framework) errors (disabled in simplified version)
    #[error("Candle error: {0}")]
    CandleSimplified(String),

    /// Text processing errors
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    /// PDF processing errors
    #[error("PDF error: {0}")]
    Pdf(String),

    /// EPUB processing errors
    #[error("EPUB error: {0}")]
    Epub(String),

    /// Index management errors
    #[error("Index error: {0}")]
    Index(String),

    /// Search errors
    #[error("Search error: {0}")]
    Search(String),

    /// HTTP/API errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// LLM API errors
    #[error("LLM API error: {0}")]
    LlmApi(String),

    /// Chat errors
    #[error("Chat error: {0}")]
    Chat(String),

    /// File not found errors
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid input errors
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Unsupported format errors
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Codec errors
    #[error("Codec error: {0}")]
    Codec(String),

    /// Compression errors
    #[error("Compression error: {0}")]
    Compression(String),

    /// Decompression errors
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Cache errors
    #[error("Cache error: {0}")]
    Cache(String),

    /// Threading/concurrency errors
    #[error("Concurrency error: {0}")]
    Concurrency(String),

    /// Memory allocation errors
    #[error("Memory error: {0}")]
    Memory(String),

    /// Generic errors with context
    #[error("Error: {message}")]
    Generic { message: String },

    /// Multiple errors combined
    #[error("Multiple errors: {errors:?}")]
    Multiple { errors: Vec<MemvidError> },
}

impl MemvidError {
    /// Create a new configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config(message.into())
    }

    /// Create a new QR code error
    pub fn qr_code<S: Into<String>>(message: S) -> Self {
        Self::QrCode(message.into())
    }

    /// Create a new video error
    pub fn video<S: Into<String>>(message: S) -> Self {
        Self::Video(message.into())
    }

    /// Create a new embedding error
    pub fn embedding<S: Into<String>>(message: S) -> Self {
        Self::Embedding(message.into())
    }

    /// Create a new text processing error
    pub fn text_processing<S: Into<String>>(message: S) -> Self {
        Self::TextProcessing(message.into())
    }

    /// Create a new PDF error
    pub fn pdf<S: Into<String>>(message: S) -> Self {
        Self::Pdf(message.into())
    }

    /// Create a new EPUB error
    pub fn epub<S: Into<String>>(message: S) -> Self {
        Self::Epub(message.into())
    }

    /// Create a new data processing error
    pub fn data_processing<S: Into<String>>(message: S) -> Self {
        Self::Generic { message: message.into() }
    }

    /// Create a new index error
    pub fn index<S: Into<String>>(message: S) -> Self {
        Self::Index(message.into())
    }

    /// Create a new search error
    pub fn search<S: Into<String>>(message: S) -> Self {
        Self::Search(message.into())
    }

    /// Create a new LLM API error
    pub fn llm_api<S: Into<String>>(message: S) -> Self {
        Self::LlmApi(message.into())
    }

    /// Create a new chat error
    pub fn chat<S: Into<String>>(message: S) -> Self {
        Self::Chat(message.into())
    }

    /// Create a new file not found error
    pub fn file_not_found<S: Into<String>>(path: S) -> Self {
        Self::FileNotFound(path.into())
    }

    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Create a new unsupported format error
    pub fn unsupported_format<S: Into<String>>(format: S) -> Self {
        Self::UnsupportedFormat(format.into())
    }

    /// Create a new codec error
    pub fn codec<S: Into<String>>(message: S) -> Self {
        Self::Codec(message.into())
    }

    /// Create a new generic error
    pub fn generic<S: Into<String>>(message: S) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Combine multiple errors
    pub fn multiple(errors: Vec<MemvidError>) -> Self {
        Self::Multiple { errors }
    }
}

/// Convert from qrcode::types::QrError
impl From<qrcode::types::QrError> for MemvidError {
    fn from(err: qrcode::types::QrError) -> Self {
        Self::QrCode(err.to_string())
    }
}

/// Convert from image::ImageError
impl From<image::ImageError> for MemvidError {
    fn from(err: image::ImageError) -> Self {
        Self::QrCode(format!("Image error: {}", err))
    }
}

// PDF extract support disabled in simplified version

/// Convert from toml::de::Error
impl From<toml::de::Error> for MemvidError {
    fn from(err: toml::de::Error) -> Self {
        Self::Config(format!("TOML parsing error: {}", err))
    }
}

/// Convert from config::ConfigError
impl From<config::ConfigError> for MemvidError {
    fn from(err: config::ConfigError) -> Self {
        Self::Config(err.to_string())
    }
}

/// Convert from toml::ser::Error
impl From<toml::ser::Error> for MemvidError {
    fn from(err: toml::ser::Error) -> Self {
        Self::Config(format!("TOML serialization error: {}", err))
    }
}

/// Convert from rqrr::DeQRError
impl From<rqrr::DeQRError> for MemvidError {
    fn from(err: rqrr::DeQRError) -> Self {
        Self::QrCode(format!("QR decode error: {:?}", err))
    }
}
