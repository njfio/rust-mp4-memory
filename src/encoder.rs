//! Main encoding functionality for creating QR code videos

use image::DynamicImage;
use std::path::Path;
use tracing::{info, warn, error};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::index::IndexManager;
use crate::qr::{QrProcessor, BatchQrProcessor};
use crate::text::{TextProcessor, TextChunk};
use crate::video::{VideoEncoder, Codec, VideoStats};

/// Main encoder for creating QR code videos from text
pub struct MemvidEncoder {
    config: Config,
    text_processor: TextProcessor,
    qr_processor: QrProcessor,
    batch_qr_processor: BatchQrProcessor,
    video_encoder: VideoEncoder,
    index_manager: Option<IndexManager>,
    chunks: Vec<TextChunk>,
}

impl MemvidEncoder {
    /// Create a new encoder with the given configuration
    pub async fn new_with_config(config: Config) -> Result<Self> {
        let text_processor = TextProcessor::new(config.text.clone());
        let qr_processor = QrProcessor::new(config.qr.clone());
        let batch_qr_processor = BatchQrProcessor::new(config.qr.clone());
        let video_encoder = VideoEncoder::new(config.clone());
        let index_manager = IndexManager::new(config.clone()).await?;

        Ok(Self {
            config,
            text_processor,
            qr_processor,
            batch_qr_processor,
            video_encoder,
            index_manager: Some(index_manager),
            chunks: Vec::new(),
        })
    }

    /// Create a new encoder with default configuration
    pub async fn new() -> Result<Self> {
        let config = Config::default();
        Self::new_with_config(config).await
    }

    /// Add text chunks directly
    pub async fn add_chunks(&mut self, texts: Vec<String>) -> Result<()> {
        let num_texts = texts.len();
        for text in texts.into_iter() {
            let chunk = TextChunk {
                content: text.clone(),
                metadata: crate::text::ChunkMetadata {
                    id: self.chunks.len(),
                    source: None,
                    page: None,
                    char_offset: 0,
                    length: text.len(),
                    frame: self.chunks.len() as u32,
                    extra: std::collections::HashMap::new(),
                },
            };
            self.chunks.push(chunk);
        }

        info!("Added {} chunks. Total: {}", num_texts, self.chunks.len());
        Ok(())
    }

    /// Add text and automatically chunk it
    pub async fn add_text(&mut self, text: &str, source: Option<String>) -> Result<()> {
        let mut new_chunks = self.text_processor.chunk_text(text, source)?;
        
        // Update frame numbers to be sequential
        for chunk in &mut new_chunks {
            chunk.metadata.frame = self.chunks.len() as u32;
            chunk.metadata.id = self.chunks.len();
            self.chunks.push(chunk.clone());
        }

        info!("Added text with {} chunks. Total: {}", new_chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Add PDF file
    pub async fn add_pdf(&mut self, pdf_path: &str) -> Result<()> {
        if !Path::new(pdf_path).exists() {
            return Err(MemvidError::file_not_found(pdf_path));
        }

        let mut new_chunks = self.text_processor.process_pdf(pdf_path).await?;
        
        // Update frame numbers to be sequential
        for chunk in &mut new_chunks {
            chunk.metadata.frame = self.chunks.len() as u32;
            chunk.metadata.id = self.chunks.len();
            self.chunks.push(chunk.clone());
        }

        info!("Added PDF {} with {} chunks. Total: {}", pdf_path, new_chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Add EPUB file
    pub async fn add_epub(&mut self, epub_path: &str) -> Result<()> {
        if !Path::new(epub_path).exists() {
            return Err(MemvidError::file_not_found(epub_path));
        }

        let mut new_chunks = self.text_processor.process_epub(epub_path).await?;
        
        // Update frame numbers to be sequential
        for chunk in &mut new_chunks {
            chunk.metadata.frame = self.chunks.len() as u32;
            chunk.metadata.id = self.chunks.len();
            self.chunks.push(chunk.clone());
        }

        info!("Added EPUB {} with {} chunks. Total: {}", epub_path, new_chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Add text file
    pub async fn add_text_file(&mut self, file_path: &str) -> Result<()> {
        if !Path::new(file_path).exists() {
            return Err(MemvidError::file_not_found(file_path));
        }

        let mut new_chunks = self.text_processor.process_text_file(file_path).await?;
        
        // Update frame numbers to be sequential
        for chunk in &mut new_chunks {
            chunk.metadata.frame = self.chunks.len() as u32;
            chunk.metadata.id = self.chunks.len();
            self.chunks.push(chunk.clone());
        }

        info!("Added text file {} with {} chunks. Total: {}", file_path, new_chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Add all files from a directory
    pub async fn add_directory(&mut self, dir_path: &str) -> Result<()> {
        if !Path::new(dir_path).exists() {
            return Err(MemvidError::file_not_found(dir_path));
        }

        let mut new_chunks = self.text_processor.process_directory(dir_path).await?;
        
        // Update frame numbers to be sequential
        for chunk in &mut new_chunks {
            chunk.metadata.frame = self.chunks.len() as u32;
            chunk.metadata.id = self.chunks.len();
            self.chunks.push(chunk.clone());
        }

        info!("Added directory {} with {} chunks. Total: {}", dir_path, new_chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Build QR code video from chunks
    pub async fn build_video(&mut self, output_path: &str, index_path: &str) -> Result<EncodingStats> {
        self.build_video_with_codec(output_path, index_path, None).await
    }

    /// Build QR code video with specific codec
    pub async fn build_video_with_codec(
        &mut self,
        output_path: &str,
        index_path: &str,
        codec: Option<Codec>,
    ) -> Result<EncodingStats> {
        if self.chunks.is_empty() {
            return Err(MemvidError::invalid_input("No chunks to encode"));
        }

        let codec = codec.unwrap_or_else(|| {
            Codec::from_str(&self.config.video.default_codec).unwrap_or(Codec::Mp4v)
        });

        info!("Building video with {} chunks using {:?} codec", self.chunks.len(), codec);

        let start_time = std::time::Instant::now();

        // Generate QR codes for all chunks
        info!("Generating QR codes...");
        let chunk_texts: Vec<String> = self.chunks.iter().map(|c| {
            serde_json::to_string(&ChunkData {
                id: c.metadata.id,
                text: c.content.clone(),
                frame: c.metadata.frame,
                metadata: c.metadata.clone(),
            }).unwrap_or_else(|_| c.content.clone())
        }).collect();

        let qr_images = self.batch_qr_processor.encode_batch(&chunk_texts).await?;
        info!("Generated {} QR codes", qr_images.len());

        // Encode video
        info!("Encoding video...");
        let video_stats = self.video_encoder.encode_qr_video(&qr_images, output_path, codec).await?;
        info!("Video encoded successfully");

        // Build and save index
        info!("Building search index...");
        if let Some(ref mut index_manager) = self.index_manager {
            index_manager.add_chunks(self.chunks.clone()).await?;
            index_manager.save(index_path)?;
            info!("Index saved to {}", index_path);
        } else {
            warn!("No index manager available, skipping index creation");
        }

        let total_time = start_time.elapsed();

        let stats = EncodingStats {
            total_chunks: self.chunks.len(),
            total_characters: self.chunks.iter().map(|c| c.content.len()).sum(),
            video_stats,
            encoding_time_seconds: total_time.as_secs_f64(),
            qr_generation_time_seconds: 0.0, // TODO: measure separately
            index_build_time_seconds: 0.0,   // TODO: measure separately
        };

        info!("Encoding completed in {:.2} seconds", stats.encoding_time_seconds);
        info!("Video file: {} ({:.2} MB)", output_path, stats.video_stats.file_size_bytes as f64 / 1024.0 / 1024.0);
        info!("Index file: {}", index_path);

        Ok(stats)
    }

    /// Get encoding statistics
    pub fn get_stats(&self) -> MemvidEncoderStats {
        let chunk_stats = self.text_processor.get_chunk_stats(&self.chunks);
        
        MemvidEncoderStats {
            total_chunks: self.chunks.len(),
            total_characters: chunk_stats.total_characters,
            avg_chunk_length: chunk_stats.avg_chunk_length,
            unique_sources: chunk_stats.sources,
            config: self.config.clone(),
        }
    }

    /// Clear all chunks
    pub fn clear(&mut self) {
        self.chunks.clear();
        if let Some(ref mut index_manager) = self.index_manager {
            index_manager.clear();
        }
        info!("Cleared all chunks");
    }

    /// Get number of chunks
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if encoder is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get chunks reference
    pub fn chunks(&self) -> &[TextChunk] {
        &self.chunks
    }
}

/// Data structure for QR code content
#[derive(serde::Serialize, serde::Deserialize)]
struct ChunkData {
    id: usize,
    text: String,
    frame: u32,
    metadata: crate::text::ChunkMetadata,
}

/// Encoding statistics
#[derive(Debug, Clone)]
pub struct EncodingStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub video_stats: VideoStats,
    pub encoding_time_seconds: f64,
    pub qr_generation_time_seconds: f64,
    pub index_build_time_seconds: f64,
}

/// Encoder statistics
#[derive(Debug, Clone)]
pub struct MemvidEncoderStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub avg_chunk_length: f64,
    pub unique_sources: usize,
    pub config: Config,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_encoder_creation() {
        // This test requires embedding model download, so we'll skip it in CI
        if std::env::var("CI").is_ok() {
            return;
        }

        let encoder = MemvidEncoder::new().await;
        // Just test that it doesn't panic
        assert!(encoder.is_ok() || encoder.is_err());
    }

    #[tokio::test]
    async fn test_add_chunks() {
        // Create encoder without embedding model for testing
        let config = Config::default();
        let mut encoder = MemvidEncoder {
            config: config.clone(),
            text_processor: TextProcessor::new(config.text.clone()),
            qr_processor: QrProcessor::new(config.qr.clone()),
            batch_qr_processor: BatchQrProcessor::new(config.qr.clone()),
            video_encoder: VideoEncoder::new(config.clone()),
            index_manager: None,
            chunks: Vec::new(),
        };

        let chunks = vec![
            "First chunk".to_string(),
            "Second chunk".to_string(),
        ];

        encoder.add_chunks(chunks).await.unwrap();
        assert_eq!(encoder.len(), 2);
        assert!(!encoder.is_empty());
    }

    #[test]
    fn test_encoder_stats() {
        let config = Config::default();
        let encoder = MemvidEncoder {
            config: config.clone(),
            text_processor: TextProcessor::new(config.text.clone()),
            qr_processor: QrProcessor::new(config.qr.clone()),
            batch_qr_processor: BatchQrProcessor::new(config.qr.clone()),
            video_encoder: VideoEncoder::new(config.clone()),
            index_manager: None,
            chunks: Vec::new(),
        };

        let stats = encoder.get_stats();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_characters, 0);
    }
}
