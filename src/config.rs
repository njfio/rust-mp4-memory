//! Configuration management for rust_mem_vid

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{MemvidError, Result};

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Logging configuration
    pub log_level: String,
    
    /// QR code configuration
    pub qr: QrConfig,
    
    /// Video encoding configuration
    pub video: VideoConfig,
    
    /// Text processing configuration
    pub text: TextConfig,

    /// Folder processing configuration
    pub folder: FolderConfig,

    /// Embedding model configuration
    pub embeddings: EmbeddingConfig,
    
    /// Search configuration
    pub search: SearchConfig,
    
    /// Chat configuration
    pub chat: ChatConfig,
    
    /// Codec-specific parameters
    pub codec_parameters: HashMap<String, CodecConfig>,
}

/// QR code configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QrConfig {
    /// QR code version (1-40)
    pub version: i16,
    
    /// Error correction level
    pub error_correction: String,
    
    /// Box size for QR code
    pub box_size: u32,
    
    /// Border size
    pub border: u32,
    
    /// Fill color
    pub fill_color: String,
    
    /// Background color
    pub back_color: String,
}

/// Video encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Default codec
    pub default_codec: String,
    
    /// Default FPS
    pub fps: f64,
    
    /// Default frame width
    pub frame_width: u32,
    
    /// Default frame height
    pub frame_height: u32,
    
    /// Use hardware acceleration if available
    pub use_hardware_acceleration: bool,
    
    /// Number of encoding threads
    pub encoding_threads: usize,
}

/// Text processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Default chunk size
    pub chunk_size: usize,
    
    /// Default overlap between chunks
    pub overlap: usize,
    
    /// Maximum chunk size
    pub max_chunk_size: usize,
    
    /// Minimum chunk size
    pub min_chunk_size: usize,
}

/// Folder processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderConfig {
    /// Maximum recursion depth (0 = current directory only, None = unlimited)
    pub max_depth: Option<usize>,
    /// File extensions to include (None = all supported types)
    pub include_extensions: Option<Vec<String>>,
    /// File extensions to exclude
    pub exclude_extensions: Vec<String>,
    /// Patterns to exclude (glob patterns)
    pub exclude_patterns: Vec<String>,
    /// Minimum file size in bytes
    pub min_file_size: usize,
    /// Maximum file size in bytes (100MB default)
    pub max_file_size: usize,
    /// Follow symbolic links
    pub follow_symlinks: bool,
    /// Include hidden files and directories
    pub include_hidden: bool,
    /// Skip binary files
    pub skip_binary: bool,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name or path
    pub model_name: String,
    
    /// Model revision
    pub model_revision: Option<String>,
    
    /// Use GPU if available
    pub use_gpu: bool,
    
    /// Batch size for embedding generation
    pub batch_size: usize,
    
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// Cache directory for models
    pub cache_dir: Option<PathBuf>,
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_top_k: usize,

    /// Maximum number of results
    pub max_top_k: usize,

    /// Similarity threshold
    pub similarity_threshold: f32,

    /// Use FAISS for vector search
    pub use_faiss: bool,

    /// Number of search threads
    pub search_threads: usize,

    /// Cache size for decoded frames
    pub cache_size: usize,

    /// Enable index building (disable for faster processing when search isn't needed)
    pub enable_index_building: bool,

    /// Enable background indexing (build index after video creation)
    pub enable_background_indexing: bool,
}

/// Chat configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Default LLM provider
    pub default_provider: String,
    
    /// API configurations for different providers
    pub providers: HashMap<String, LlmProviderConfig>,
    
    /// Maximum context length
    pub max_context_length: usize,
    
    /// Temperature for generation
    pub temperature: f32,
    
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    /// API endpoint
    pub endpoint: String,
    
    /// Model name
    pub model: String,
    
    /// API key (optional, can be set via environment)
    pub api_key: Option<String>,
    
    /// Additional headers
    pub headers: HashMap<String, String>,
    
    /// Request timeout in seconds
    pub timeout: u64,
}

/// Codec-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Video file extension
    pub file_extension: String,
    
    /// FFmpeg codec name
    pub ffmpeg_codec: String,
    
    /// OpenCV fourcc code
    pub opencv_fourcc: Option<String>,
    
    /// Encoding preset
    pub preset: String,
    
    /// Constant Rate Factor (quality)
    pub crf: u32,
    
    /// Pixel format
    pub pixel_format: String,
    
    /// Frame rate
    pub fps: f64,
    
    /// Frame width
    pub width: u32,
    
    /// Frame height
    pub height: u32,
    
    /// Additional FFmpeg parameters
    pub extra_params: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        let mut codec_parameters = HashMap::new();
        
        // MP4V codec (OpenCV compatible)
        codec_parameters.insert("mp4v".to_string(), CodecConfig {
            file_extension: ".mp4".to_string(),
            ffmpeg_codec: "libx264".to_string(),
            opencv_fourcc: Some("mp4v".to_string()),
            preset: "medium".to_string(),
            crf: 23,
            pixel_format: "yuv420p".to_string(),
            fps: 30.0,
            width: 512,
            height: 512,
            extra_params: vec![],
        });
        
        // H.264 codec
        codec_parameters.insert("h264".to_string(), CodecConfig {
            file_extension: ".mp4".to_string(),
            ffmpeg_codec: "libx264".to_string(),
            opencv_fourcc: None,
            preset: "medium".to_string(),
            crf: 23,
            pixel_format: "yuv420p".to_string(),
            fps: 30.0,
            width: 512,
            height: 512,
            extra_params: vec![],
        });
        
        // H.265 codec
        codec_parameters.insert("h265".to_string(), CodecConfig {
            file_extension: ".mp4".to_string(),
            ffmpeg_codec: "libx265".to_string(),
            opencv_fourcc: None,
            preset: "medium".to_string(),
            crf: 28,
            pixel_format: "yuv420p".to_string(),
            fps: 30.0,
            width: 512,
            height: 512,
            extra_params: vec![],
        });
        
        // AV1 codec
        codec_parameters.insert("av1".to_string(), CodecConfig {
            file_extension: ".mp4".to_string(),
            ffmpeg_codec: "libaom-av1".to_string(),
            opencv_fourcc: None,
            preset: "6".to_string(),
            crf: 30,
            pixel_format: "yuv420p".to_string(),
            fps: 30.0,
            width: 512,
            height: 512,
            extra_params: vec!["-cpu-used".to_string(), "6".to_string()],
        });
        
        let mut providers = HashMap::new();
        providers.insert("openai".to_string(), LlmProviderConfig {
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            api_key: None,
            headers: HashMap::new(),
            timeout: 30,
        });
        
        providers.insert("anthropic".to_string(), LlmProviderConfig {
            endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            model: "claude-3-sonnet-20240229".to_string(),
            api_key: None,
            headers: HashMap::new(),
            timeout: 30,
        });
        
        Self {
            log_level: "info".to_string(),
            qr: QrConfig {
                version: 1,
                error_correction: "M".to_string(),
                box_size: 10,
                border: 4,
                fill_color: "black".to_string(),
                back_color: "white".to_string(),
            },
            video: VideoConfig {
                default_codec: "mp4v".to_string(),
                fps: 30.0,
                frame_width: 512,
                frame_height: 512,
                use_hardware_acceleration: true,
                encoding_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            },
            text: TextConfig {
                chunk_size: 1000,  // Safe size for QR codes with compression
                overlap: 50,
                max_chunk_size: 1500, // Conservative max to ensure QR compatibility
                min_chunk_size: 100,
            },
            folder: FolderConfig {
                max_depth: Some(10), // Reasonable default depth
                include_extensions: None, // Include all supported types
                exclude_extensions: vec![
                    "exe".to_string(), "dll".to_string(), "so".to_string(), "dylib".to_string(),
                    "bin".to_string(), "obj".to_string(), "o".to_string(), "a".to_string(),
                    "lib".to_string(), "zip".to_string(), "tar".to_string(), "gz".to_string(),
                    "rar".to_string(), "7z".to_string(), "iso".to_string(), "img".to_string(),
                    "dmg".to_string(), "pkg".to_string(), "deb".to_string(), "rpm".to_string(),
                ],
                exclude_patterns: vec![
                    "*/target/*".to_string(), // Rust build artifacts
                    "*/node_modules/*".to_string(), // Node.js dependencies
                    "*/.git/*".to_string(), // Git repository
                    "*/.svn/*".to_string(), // SVN repository
                    "*/.hg/*".to_string(), // Mercurial repository
                    "*/build/*".to_string(), // Build directories
                    "*/dist/*".to_string(), // Distribution directories
                    "*/__pycache__/*".to_string(), // Python cache
                    "*.tmp".to_string(), // Temporary files
                    "*.temp".to_string(), // Temporary files
                    "*.cache".to_string(), // Cache files
                ],
                min_file_size: 1, // At least 1 byte
                max_file_size: 100 * 1024 * 1024, // 100MB max
                follow_symlinks: false, // Don't follow symlinks by default
                include_hidden: false, // Skip hidden files by default
                skip_binary: true, // Skip binary files by default
            },
            embeddings: EmbeddingConfig {
                model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                model_revision: None,
                use_gpu: false,
                batch_size: 32,
                max_sequence_length: 512,
                cache_dir: None,
            },
            search: SearchConfig {
                default_top_k: 5,
                max_top_k: 100,
                similarity_threshold: 0.0,
                use_faiss: true,
                search_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
                cache_size: 1000,
                enable_index_building: true,
                enable_background_indexing: false,
            },
            chat: ChatConfig {
                default_provider: "openai".to_string(),
                providers,
                max_context_length: 4000,
                temperature: 0.7,
                max_tokens: 1000,
            },
            codec_parameters,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Get codec configuration
    pub fn get_codec_config(&self, codec: &str) -> Result<&CodecConfig> {
        self.codec_parameters
            .get(codec)
            .ok_or_else(|| MemvidError::codec(format!("Unsupported codec: {}", codec)))
    }
    
    /// Get LLM provider configuration
    pub fn get_provider_config(&self, provider: &str) -> Result<&LlmProviderConfig> {
        self.chat
            .providers
            .get(provider)
            .ok_or_else(|| MemvidError::config(format!("Unknown provider: {}", provider)))
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate QR config
        if self.qr.version < 1 || self.qr.version > 40 {
            return Err(MemvidError::config("QR version must be between 1 and 40"));
        }
        
        // Validate text config
        if self.text.chunk_size < self.text.min_chunk_size {
            return Err(MemvidError::config("Chunk size cannot be less than minimum"));
        }
        
        if self.text.chunk_size > self.text.max_chunk_size {
            return Err(MemvidError::config("Chunk size cannot be greater than maximum"));
        }
        
        // Validate video config
        if self.video.fps <= 0.0 {
            return Err(MemvidError::config("FPS must be positive"));
        }
        
        // Validate search config
        if self.search.default_top_k > self.search.max_top_k {
            return Err(MemvidError::config("Default top_k cannot be greater than max_top_k"));
        }
        
        Ok(())
    }
}

/// Get the default configuration
pub fn get_default_config() -> Config {
    Config::default()
}

/// Load configuration from environment and files
pub fn load_config() -> Result<Config> {
    let mut config = Config::default();
    
    // Try to load from config file
    if let Ok(file_config) = Config::from_file("memvid.toml") {
        config = file_config;
    }
    
    // Override with environment variables
    if let Ok(log_level) = std::env::var("MEMVID_LOG_LEVEL") {
        config.log_level = log_level;
    }
    
    if let Ok(model_name) = std::env::var("MEMVID_EMBEDDING_MODEL") {
        config.embeddings.model_name = model_name;
    }
    
    // Validate the final configuration
    config.validate()?;
    
    Ok(config)
}
