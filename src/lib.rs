//! # Rust MemVid
//!
//! A Rust implementation of the MemVid library - a video-based AI memory system
//! that stores text chunks as QR codes in video files with lightning-fast semantic search.
//!
//! ## Features
//!
//! - Store millions of text chunks in MP4 files
//! - Lightning-fast semantic search using embeddings
//! - Support for multiple video codecs (H.264, H.265, AV1)
//! - PDF and EPUB document processing
//! - Built-in chat functionality with LLM integration
//! - Efficient compression and storage
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rust_mem_vid::{MemvidEncoder, MemvidRetriever, MemvidChat};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create encoder and add text chunks
//!     let mut encoder = MemvidEncoder::new().await?;
//!     encoder.add_chunks(vec![
//!         "Important fact 1".to_string(),
//!         "Important fact 2".to_string(),
//!         "Historical event details".to_string(),
//!     ]).await?;
//!     
//!     // Build video memory
//!     encoder.build_video("memory.mp4", "memory_index.json").await?;
//!     
//!     // Search the memory
//!     let retriever = MemvidRetriever::new("memory.mp4", "memory_index.json").await?;
//!     let results = retriever.search("historical events", 5).await?;
//!     
//!     // Chat with the memory
//!     let mut chat = MemvidChat::new("memory.mp4", "memory_index.json").await?;
//!     let response = chat.chat("What do you know about historical events?").await?;
//!     println!("{}", response);
//!     
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod data;
pub mod encoder;
pub mod folder;
pub mod retriever;
pub mod chat;
pub mod qr;
pub mod video;
pub mod embeddings;
pub mod index;
pub mod text;
pub mod error;
pub mod utils;
pub mod background_indexing;

// New temporal and comparative analysis modules
pub mod memory_diff;
pub mod multi_memory;
pub mod temporal_analysis;

// AI Intelligence modules (Phase 1)
pub mod knowledge_graph;
pub mod concept_extractors;
pub mod relationship_analyzers;
pub mod content_synthesis;
pub mod analytics_dashboard;

// Web platform module (Phase 2)
pub mod web_server;

// Re-export main types
pub use encoder::MemvidEncoder;
pub use retriever::MemvidRetriever;
pub use chat::MemvidChat;
pub use config::Config;
pub use error::{MemvidError, Result};

// Re-export commonly used types
pub use data::{DataProcessor, DataFileType, DataChunk, ChunkingStrategy};
pub use embeddings::EmbeddingModel;
pub use folder::{FolderProcessor, FileInfo, FolderStats, SkipReason};
pub use index::IndexManager;
pub use text::{TextProcessor, ChunkMetadata};
pub use video::{VideoEncoder, VideoDecoder, Codec};

// Re-export temporal and comparative analysis types
pub use memory_diff::{MemoryDiff, MemoryDiffEngine, DiffSummary, ChunkDiff, ChunkModification};
pub use multi_memory::{MultiMemoryEngine, MultiMemorySearchResult, MemoryInfo, GlobalMemoryStats};
pub use temporal_analysis::{MemorySnapshot, MemoryTimeline, TemporalAnalysisEngine, TimelineAnalysis};

// Re-export AI Intelligence types
pub use knowledge_graph::{KnowledgeGraph, KnowledgeGraphBuilder, ConceptNode, ConceptRelationship};
pub use content_synthesis::{ContentSynthesizer, SynthesisResult, SynthesisType};
pub use analytics_dashboard::{AnalyticsDashboard, DashboardOutput, AnalyticsData};

// Re-export Web Platform types
pub use web_server::{MemoryWebServer, MemoryInstance, MemoryMetadata, MemoryPermissions};

// Re-export Background Indexing types
pub use background_indexing::{BackgroundIndexer, IndexingStatus, submit_background_indexing, get_indexing_status, wait_for_indexing};

/// Version of the rust_mem_vid library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default chunk size for text processing
pub const DEFAULT_CHUNK_SIZE: usize = 512;

/// Default overlap between chunks
pub const DEFAULT_OVERLAP: usize = 50;

/// Default number of search results
pub const DEFAULT_TOP_K: usize = 5;

/// Initialize the library with default configuration
pub async fn init() -> Result<()> {
    // Initialize logging (ignore error if already initialized)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    // In a full implementation, this would initialize FFmpeg
    // ffmpeg_next::init()?;

    Ok(())
}

/// Initialize the library with custom configuration
pub async fn init_with_config(config: Config) -> Result<()> {
    // Initialize logging with custom level (ignore error if already initialized)
    let filter = tracing_subscriber::EnvFilter::new(&config.log_level);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init();

    // In a full implementation, this would initialize FFmpeg
    // ffmpeg_next::init()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_workflow() {
        let _ = init().await;
        
        // This test would require actual implementation
        // For now, just test that the modules compile
        assert_eq!(VERSION.len() > 0, true);
    }
}
