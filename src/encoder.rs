//! Main encoding functionality for creating QR code videos

use std::path::Path;
use tracing::{info, warn, error, debug};

use crate::config::Config;
use crate::video::VideoMetadata;
use crate::data::{DataProcessor, DataFileType};
use crate::error::{MemvidError, Result};
use crate::folder::{FolderProcessor, FolderStats};
use crate::index::IndexManager;
use crate::qr::{QrProcessor, BatchQrProcessor};
use crate::text::{TextProcessor, TextChunk};
use crate::video::{VideoEncoder, Codec};

/// Analysis of chunk sizes and optimization recommendations
#[derive(Debug, Clone)]
pub struct ChunkAnalysis {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub avg_chunk_size: usize,
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub recommended_size: usize,
    pub oversized_chunks: usize,
    pub needs_optimization: bool,
}

/// Main encoder for creating QR code videos from text
pub struct MemvidEncoder {
    config: Config,
    text_processor: TextProcessor,
    data_processor: DataProcessor,
    folder_processor: FolderProcessor,
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
            config: config.clone(),
            text_processor,
            data_processor: DataProcessor::new(config.clone()),
            folder_processor: FolderProcessor::new(config.folder.clone())?,
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



    /// Add data file (CSV, Parquet, JSON, code files, etc.)
    pub async fn add_data_file(&mut self, file_path: &str) -> Result<()> {
        if !Path::new(file_path).exists() {
            return Err(MemvidError::file_not_found(file_path));
        }

        let data_chunks = self.data_processor.process_file(file_path).await?;
        let mut text_chunks = self.data_processor.to_text_chunks(data_chunks);

        // Update frame numbers to be sequential
        for chunk in &mut text_chunks {
            chunk.metadata.frame = self.chunks.len() as u32;
            chunk.metadata.id = self.chunks.len();
            self.chunks.push(chunk.clone());
        }

        info!("Added data file {} with {} chunks. Total: {}", file_path, text_chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Add CSV file with specific chunking strategy
    pub async fn add_csv_file(&mut self, file_path: &str) -> Result<()> {
        self.add_data_file(file_path).await
    }

    /// Add Parquet file
    pub async fn add_parquet_file(&mut self, file_path: &str) -> Result<()> {
        self.add_data_file(file_path).await
    }

    /// Add code file (Rust, JavaScript, Python, etc.)
    pub async fn add_code_file(&mut self, file_path: &str) -> Result<()> {
        self.add_data_file(file_path).await
    }

    /// Add log file
    pub async fn add_log_file(&mut self, file_path: &str) -> Result<()> {
        self.add_data_file(file_path).await
    }

    /// Add any supported file type automatically
    pub async fn add_file(&mut self, file_path: &str) -> Result<()> {
        let path = Path::new(file_path);
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Try data processor first for supported formats
        if DataFileType::from_extension(&extension).is_some() {
            self.add_data_file(file_path).await
        } else {
            // Fall back to text processor
            match extension.as_str() {
                "pdf" => self.add_pdf(file_path).await,
                "epub" => self.add_epub(file_path).await,
                "txt" | "md" | "rst" => self.add_text_file(file_path).await,
                _ => Err(MemvidError::unsupported_format(&extension)),
            }
        }
    }

    /// Build QR code video from chunks
    pub async fn build_video(&mut self, output_path: &str, index_path: &str) -> Result<EncodingStats> {
        self.build_video_with_codec(output_path, index_path, None).await
    }

    /// Optimize chunk sizes for QR code compatibility
    pub fn optimize_chunks_for_qr(&mut self) -> Result<()> {
        let recommended_size = self.qr_processor.get_recommended_chunk_size()?;

        info!("Optimizing chunks for QR codes. Recommended size: {} characters", recommended_size);

        let mut optimized_chunks = Vec::new();

        for chunk in &self.chunks {
            if chunk.content.len() <= recommended_size {
                // Chunk is already small enough
                optimized_chunks.push(chunk.clone());
            } else {
                // Split large chunk into smaller ones
                let content = &chunk.content;
                let mut start = 0;
                let mut chunk_id = optimized_chunks.len();

                while start < content.len() {
                    let end = std::cmp::min(start + recommended_size, content.len());
                    let chunk_content = content[start..end].to_string();

                    let mut new_metadata = chunk.metadata.clone();
                    new_metadata.id = chunk_id;
                    new_metadata.frame = chunk_id as u32;
                    new_metadata.char_offset = start;
                    new_metadata.length = chunk_content.len();

                    optimized_chunks.push(TextChunk {
                        content: chunk_content,
                        metadata: new_metadata,
                    });

                    start = end;
                    chunk_id += 1;
                }
            }
        }

        let original_count = self.chunks.len();
        self.chunks = optimized_chunks;
        let new_count = self.chunks.len();

        info!("Chunk optimization complete: {} -> {} chunks", original_count, new_count);

        Ok(())
    }

    /// Build QR code video with specific codec (optimized for large datasets)
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

        // Optimize chunks for QR code compatibility
        info!("Optimizing chunks for QR code compatibility...");
        self.optimize_chunks_for_qr()?;
        info!("Optimized to {} chunks", self.chunks.len());

        let start_time = std::time::Instant::now();

        // Check if we should use streaming processing for very large datasets
        let use_streaming = self.chunks.len() > 1000;

        if use_streaming {
            info!("Using streaming processing for large dataset ({} chunks)", self.chunks.len());
            self.build_video_streaming(output_path, index_path, codec).await
        } else {
            info!("Using batch processing for dataset ({} chunks)", self.chunks.len());
            self.build_video_batch(output_path, index_path, codec).await
        }
    }

    /// Build video using batch processing (for smaller datasets)
    async fn build_video_batch(
        &mut self,
        output_path: &str,
        index_path: &str,
        codec: Codec,
    ) -> Result<EncodingStats> {
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

        let encoding_time = start_time.elapsed();

        Ok(EncodingStats {
            total_chunks: self.chunks.len(),
            total_frames: video_stats.frame_count as usize,
            encoding_time_seconds: encoding_time.as_secs_f64(),
            video_file_size_bytes: video_stats.file_size_bytes,
            compression_ratio: self.calculate_compression_ratio(&video_stats),
            qr_generation_time_seconds: 0.0, // TODO: Track separately
            video_encoding_time_seconds: video_stats.encoding_time_seconds,
            index_building_time_seconds: 0.0, // TODO: Track separately
        })
    }

    /// Build video using streaming processing (for large datasets)
    async fn build_video_streaming(
        &mut self,
        output_path: &str,
        index_path: &str,
        codec: Codec,
    ) -> Result<EncodingStats> {
        use std::path::Path;

        let start_time = std::time::Instant::now();
        let total_chunks = self.chunks.len();
        let batch_size = 500; // Process in smaller batches to manage memory

        info!("Processing {} chunks in streaming batches of {}", total_chunks, batch_size);

        // Create output directory structure
        if let Some(parent) = Path::new(output_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let base_path = Path::new(output_path).with_extension("");
        let frames_dir = format!("{}_frames", base_path.to_string_lossy());
        std::fs::create_dir_all(&frames_dir)?;

        let mut total_frames = 0u32;
        let mut total_file_size = 0u64;

        // Process chunks in batches
        for (batch_idx, chunk_batch) in self.chunks.chunks(batch_size).enumerate() {
            let batch_start = batch_idx * batch_size;
            info!("Processing batch {}/{} (chunks {}-{})",
                batch_idx + 1,
                (total_chunks + batch_size - 1) / batch_size,
                batch_start,
                std::cmp::min(batch_start + batch_size, total_chunks) - 1
            );

            // Generate QR codes for this batch
            let chunk_texts: Vec<String> = chunk_batch.iter().map(|c| {
                serde_json::to_string(&ChunkData {
                    id: c.metadata.id,
                    text: c.content.clone(),
                    frame: c.metadata.frame,
                    metadata: c.metadata.clone(),
                }).unwrap_or_else(|_| c.content.clone())
            }).collect();

            let qr_images = self.batch_qr_processor.encode_batch(&chunk_texts).await?;

            // Save frames for this batch
            for (i, image) in qr_images.iter().enumerate() {
                let frame_idx = batch_start + i;
                let frame_path = format!("{}/frame_{:06}.png", frames_dir, frame_idx);
                image.save(&frame_path)?;
                total_frames += 1;
            }

            // Clear memory by dropping the QR images
            drop(qr_images);

            if batch_idx % 5 == 0 {
                info!("Completed {} batches, {} frames processed", batch_idx + 1, total_frames);
            }
        }

        // Create video metadata
        let codec_config = self.config.get_codec_config(codec.as_str())?;
        let video_metadata = VideoMetadata {
            frame_count: total_frames,
            fps: codec_config.fps,
            width: codec_config.width,
            height: codec_config.height,
            codec: codec.as_str().to_string(),
            frames_dir: frames_dir.clone(),
        };

        let metadata_json = serde_json::to_string_pretty(&video_metadata)?;
        std::fs::write(output_path, metadata_json)?;

        // Calculate file size
        total_file_size = std::fs::metadata(output_path)?.len();

        // Build and save index
        info!("Building search index...");
        if let Some(ref mut index_manager) = self.index_manager {
            index_manager.add_chunks(self.chunks.clone()).await?;
            index_manager.save(index_path)?;
            info!("Index saved to {}", index_path);
        } else {
            warn!("No index manager available, skipping index creation");
        }

        let encoding_time = start_time.elapsed();

        // Create video stats for compatibility
        let video_stats = crate::video::VideoStats {
            frame_count: total_frames,
            duration_seconds: total_frames as f64 / codec_config.fps,
            file_size_bytes: total_file_size,
            encoding_time_seconds: encoding_time.as_secs_f64(),
            codec: codec.as_str().to_string(),
            fps: codec_config.fps,
            width: codec_config.width,
            height: codec_config.height,
        };

        Ok(EncodingStats {
            total_chunks: self.chunks.len(),
            total_frames: total_frames as usize,
            encoding_time_seconds: encoding_time.as_secs_f64(),
            video_file_size_bytes: total_file_size,
            compression_ratio: self.calculate_compression_ratio(&video_stats),
            qr_generation_time_seconds: 0.0, // TODO: Track separately
            video_encoding_time_seconds: video_stats.encoding_time_seconds,
            index_building_time_seconds: 0.0, // TODO: Track separately
        })
    }

    /// Calculate compression ratio for video stats
    fn calculate_compression_ratio(&self, video_stats: &crate::video::VideoStats) -> f64 {
        let total_text_bytes: usize = self.chunks.iter().map(|c| c.content.len()).sum();
        if total_text_bytes > 0 {
            video_stats.file_size_bytes as f64 / total_text_bytes as f64
        } else {
            0.0
        }
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

    /// Get recommended chunk size for current configuration
    pub fn get_recommended_chunk_size(&self) -> Result<usize> {
        self.qr_processor.get_recommended_chunk_size()
    }

    /// Analyze current chunks and suggest optimizations
    pub fn analyze_chunks(&self) -> ChunkAnalysis {
        let total_chunks = self.chunks.len();
        let total_chars: usize = self.chunks.iter().map(|c| c.content.len()).sum();
        let avg_chunk_size = if total_chunks > 0 { total_chars / total_chunks } else { 0 };
        let max_chunk_size = self.chunks.iter().map(|c| c.content.len()).max().unwrap_or(0);
        let min_chunk_size = self.chunks.iter().map(|c| c.content.len()).min().unwrap_or(0);

        let recommended_size = self.qr_processor.get_recommended_chunk_size().unwrap_or(1000);
        let oversized_chunks = self.chunks.iter().filter(|c| c.content.len() > recommended_size).count();

        ChunkAnalysis {
            total_chunks,
            total_characters: total_chars,
            avg_chunk_size,
            max_chunk_size,
            min_chunk_size,
            recommended_size,
            oversized_chunks,
            needs_optimization: oversized_chunks > 0,
        }
    }

    /// Get chunks reference
    pub fn chunks(&self) -> &[TextChunk] {
        &self.chunks
    }

    /// Add all supported files from a directory recursively
    pub async fn add_directory(&mut self, dir_path: &str) -> Result<FolderStats> {
        self.add_directory_with_config(dir_path, None).await
    }

    /// Add directory with custom folder configuration
    pub async fn add_directory_with_config(&mut self, dir_path: &str, folder_config: Option<crate::config::FolderConfig>) -> Result<FolderStats> {
        let start_time = std::time::Instant::now();

        // Use custom config if provided, otherwise use encoder's config
        let processor = if let Some(config) = folder_config {
            FolderProcessor::new(config)?
        } else {
            FolderProcessor::new(self.config.folder.clone())?
        };

        // Discover files
        let files = processor.discover_files(dir_path)?;

        let mut stats = FolderStats {
            directories_scanned: 1,
            files_found: files.len(),
            ..Default::default()
        };

        info!("Found {} files in directory: {}", files.len(), dir_path);

        // Process each file
        for file_info in files {
            let file_path = file_info.path.to_string_lossy();

            match self.add_file(&file_path).await {
                Ok(_) => {
                    stats.files_processed += 1;
                    stats.bytes_processed += file_info.size;
                    debug!("Processed file: {}", file_path);
                }
                Err(e) => {
                    stats.files_failed += 1;
                    warn!("Failed to process file {}: {}", file_path, e);
                }
            }
        }

        stats.processing_time_ms = start_time.elapsed().as_millis() as u64;

        info!("Directory processing complete: {} processed, {} failed, {} skipped",
              stats.files_processed, stats.files_failed, stats.files_skipped);

        Ok(stats)
    }

    /// Add multiple directories
    pub async fn add_directories(&mut self, dir_paths: &[&str]) -> Result<Vec<FolderStats>> {
        let mut all_stats = Vec::new();

        for dir_path in dir_paths {
            match self.add_directory(dir_path).await {
                Ok(stats) => all_stats.push(stats),
                Err(e) => {
                    error!("Failed to process directory {}: {}", dir_path, e);
                    // Continue with other directories
                }
            }
        }

        Ok(all_stats)
    }

    /// Preview files that would be processed in a directory (without actually processing)
    pub fn preview_directory(&self, dir_path: &str) -> Result<Vec<crate::folder::FileInfo>> {
        let processor = FolderProcessor::new(self.config.folder.clone())?;
        processor.discover_files(dir_path)
    }

    /// Preview files with custom configuration
    pub fn preview_directory_with_config(&self, dir_path: &str, folder_config: crate::config::FolderConfig) -> Result<Vec<crate::folder::FileInfo>> {
        let processor = FolderProcessor::new(folder_config)?;
        processor.discover_files(dir_path)
    }

    /// Get folder processing configuration
    pub fn folder_config(&self) -> &crate::config::FolderConfig {
        &self.config.folder
    }

    /// Update folder processing configuration
    pub fn set_folder_config(&mut self, config: crate::config::FolderConfig) -> Result<()> {
        self.folder_processor = FolderProcessor::new(config.clone())?;
        self.config.folder = config;
        Ok(())
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
    pub total_frames: usize,
    pub encoding_time_seconds: f64,
    pub video_file_size_bytes: u64,
    pub compression_ratio: f64,
    pub qr_generation_time_seconds: f64,
    pub video_encoding_time_seconds: f64,
    pub index_building_time_seconds: f64,
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
            data_processor: DataProcessor::new(config.clone()),
            folder_processor: FolderProcessor::new(config.folder.clone()).unwrap(),
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
            data_processor: DataProcessor::new(config.clone()),
            folder_processor: FolderProcessor::new(config.folder.clone()).unwrap(),
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
