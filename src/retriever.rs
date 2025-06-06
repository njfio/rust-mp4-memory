//! Retrieval functionality for searching QR code videos

use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn, error};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::index::{IndexManager, ChunkSearchResult};
use crate::qr::{QrProcessor, BatchQrProcessor};
use crate::video::{VideoDecoder, VideoInfo};

/// Main retriever for searching QR code videos
pub struct MemvidRetriever {
    config: Config,
    video_path: String,
    index_path: String,
    qr_processor: QrProcessor,
    batch_qr_processor: BatchQrProcessor,
    video_decoder: VideoDecoder,
    index_manager: IndexManager,
    video_info: VideoInfo,
    frame_cache: HashMap<u32, String>,
}

impl MemvidRetriever {
    /// Create a new retriever
    pub async fn new(video_path: &str, index_path: &str) -> Result<Self> {
        let config = Config::default();
        Self::new_with_config(video_path, index_path, config).await
    }

    /// Create a new retriever with custom configuration
    pub async fn new_with_config(video_path: &str, index_path: &str, config: Config) -> Result<Self> {
        if !Path::new(video_path).exists() {
            return Err(MemvidError::file_not_found(video_path));
        }

        if !Path::new(index_path).exists() {
            return Err(MemvidError::file_not_found(index_path));
        }

        let qr_processor = QrProcessor::new(config.qr.clone());
        let batch_qr_processor = BatchQrProcessor::new(config.qr.clone());
        let video_decoder = VideoDecoder::new(config.clone());

        // Load index
        let index_manager = IndexManager::load(index_path).await?;

        // Get video information
        let video_info = video_decoder.get_video_info(video_path).await?;

        info!("Loaded retriever for video: {} ({} frames, {:.1}s)", 
              video_path, video_info.total_frames, video_info.duration_seconds);
        info!("Index contains {} chunks", index_manager.len());

        Ok(Self {
            config,
            video_path: video_path.to_string(),
            index_path: index_path.to_string(),
            qr_processor,
            batch_qr_processor,
            video_decoder,
            index_manager,
            video_info,
            frame_cache: HashMap::new(),
        })
    }

    /// Search for relevant chunks using semantic search
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let search_results = self.search_with_metadata(query, top_k).await?;
        Ok(search_results.into_iter().map(|r| r.text).collect())
    }

    /// Search with full metadata
    pub async fn search_with_metadata(&self, query: &str, top_k: usize) -> Result<Vec<RetrievalResult>> {
        info!("Searching for: '{}' (top_k={})", query, top_k);

        // Search the index
        let search_results = self.index_manager.search(query, top_k).await?;

        // Convert to retrieval results
        let mut results = Vec::new();
        for result in search_results {
            let frame_number = result.metadata.frame;
            results.push(RetrievalResult {
                chunk_id: result.chunk_id,
                similarity: result.similarity,
                text: result.text,
                metadata: result.metadata,
                frame_number,
            });
        }

        info!("Found {} results", results.len());
        Ok(results)
    }

    /// Get chunk by ID
    pub fn get_chunk(&self, chunk_id: usize) -> Option<RetrievalResult> {
        if let Some((metadata, text)) = self.index_manager.get_chunk(chunk_id) {
            Some(RetrievalResult {
                chunk_id,
                similarity: 1.0,
                text: text.clone(),
                metadata: metadata.clone(),
                frame_number: metadata.frame,
            })
        } else {
            None
        }
    }

    /// Get context window around a chunk
    pub fn get_context_window(&self, chunk_id: usize, window_size: usize) -> Vec<RetrievalResult> {
        let context_chunks = self.index_manager.get_context_window(chunk_id, window_size);
        
        context_chunks
            .into_iter()
            .map(|(id, metadata, text)| RetrievalResult {
                chunk_id: id,
                similarity: if id == chunk_id { 1.0 } else { 0.5 },
                text: text.clone(),
                metadata: metadata.clone(),
                frame_number: metadata.frame,
            })
            .collect()
    }

    /// Extract and decode frame from video
    pub async fn extract_frame(&mut self, frame_number: u32) -> Result<String> {
        // Check cache first
        if let Some(cached) = self.frame_cache.get(&frame_number) {
            return Ok(cached.clone());
        }

        // Extract frame from video
        let image = self.video_decoder.extract_frame(&self.video_path, frame_number).await?;
        
        // Decode QR code
        let decoded = self.qr_processor.decode_qr(&image)?;

        // Cache the result
        if self.frame_cache.len() < self.config.search.cache_size {
            self.frame_cache.insert(frame_number, decoded.clone());
        }

        Ok(decoded)
    }

    /// Extract and decode multiple frames in parallel
    pub async fn extract_frames(&mut self, frame_numbers: &[u32]) -> Result<HashMap<u32, String>> {
        let mut results = HashMap::new();
        let mut uncached_frames = Vec::new();

        // Check cache first
        for &frame_number in frame_numbers {
            if let Some(cached) = self.frame_cache.get(&frame_number) {
                results.insert(frame_number, cached.clone());
            } else {
                uncached_frames.push(frame_number);
            }
        }

        if uncached_frames.is_empty() {
            return Ok(results);
        }

        // Extract uncached frames
        let extracted_results = self.batch_qr_processor
            .extract_and_decode_frames(&self.video_path, &uncached_frames)
            .await?;

        // Process results and update cache
        for (frame_number, decoded_opt) in extracted_results {
            if let Some(decoded) = decoded_opt {
                results.insert(frame_number, decoded.clone());
                
                // Update cache
                if self.frame_cache.len() < self.config.search.cache_size {
                    self.frame_cache.insert(frame_number, decoded);
                }
            }
        }

        Ok(results)
    }

    /// Get chunks by frame numbers
    pub async fn get_chunks_by_frames(&mut self, frame_numbers: &[u32]) -> Result<Vec<RetrievalResult>> {
        let chunks = self.index_manager.get_chunks_by_frames(frame_numbers);
        
        let results = chunks
            .into_iter()
            .map(|(chunk_id, metadata, text)| RetrievalResult {
                chunk_id,
                similarity: 1.0,
                text: text.clone(),
                metadata: metadata.clone(),
                frame_number: metadata.frame,
            })
            .collect();

        Ok(results)
    }

    /// Get retriever statistics
    pub fn get_stats(&self) -> RetrieverStats {
        let index_stats = self.index_manager.get_stats();
        
        RetrieverStats {
            video_path: self.video_path.clone(),
            index_path: self.index_path.clone(),
            video_info: self.video_info.clone(),
            index_stats,
            cache_size: self.frame_cache.len(),
            max_cache_size: self.config.search.cache_size,
        }
    }

    /// Clear frame cache
    pub fn clear_cache(&mut self) {
        self.frame_cache.clear();
        info!("Cleared frame cache");
    }

    /// Prefetch frames into cache
    pub async fn prefetch_frames(&mut self, frame_numbers: &[u32]) -> Result<()> {
        let uncached: Vec<u32> = frame_numbers
            .iter()
            .filter(|&&frame| !self.frame_cache.contains_key(&frame))
            .cloned()
            .collect();

        if !uncached.is_empty() {
            info!("Prefetching {} frames", uncached.len());
            self.extract_frames(&uncached).await?;
        }

        Ok(())
    }

    /// Search and get full context for results
    pub async fn search_with_context(
        &self,
        query: &str,
        top_k: usize,
        context_window: usize,
    ) -> Result<Vec<ContextualResult>> {
        let search_results = self.search_with_metadata(query, top_k).await?;
        
        let mut contextual_results = Vec::new();
        
        for result in search_results {
            let context = self.get_context_window(result.chunk_id, context_window);
            
            contextual_results.push(ContextualResult {
                main_result: result,
                context,
            });
        }

        Ok(contextual_results)
    }

    /// Get video information
    pub fn get_video_info(&self) -> &VideoInfo {
        &self.video_info
    }

    /// Get index manager reference
    pub fn get_index_manager(&self) -> &IndexManager {
        &self.index_manager
    }
}

/// Retrieval result with metadata
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk_id: usize,
    pub similarity: f32,
    pub text: String,
    pub metadata: crate::text::ChunkMetadata,
    pub frame_number: u32,
}

/// Contextual result with surrounding chunks
#[derive(Debug, Clone)]
pub struct ContextualResult {
    pub main_result: RetrievalResult,
    pub context: Vec<RetrievalResult>,
}

/// Retriever statistics
#[derive(Debug, Clone)]
pub struct RetrieverStats {
    pub video_path: String,
    pub index_path: String,
    pub video_info: VideoInfo,
    pub index_stats: crate::index::IndexStats,
    pub cache_size: usize,
    pub max_cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_retriever_creation() {
        // This test would require actual video and index files
        // For now, just test that the structure compiles
        let result = MemvidRetriever::new("nonexistent.mp4", "nonexistent.json").await;
        assert!(result.is_err()); // Should fail because files don't exist
    }

    #[test]
    fn test_retrieval_result() {
        let result = RetrievalResult {
            chunk_id: 0,
            similarity: 0.95,
            text: "Test text".to_string(),
            metadata: crate::text::ChunkMetadata {
                id: 0,
                source: None,
                page: None,
                char_offset: 0,
                length: 9,
                frame: 0,
                extra: std::collections::HashMap::new(),
            },
            frame_number: 0,
        };

        assert_eq!(result.chunk_id, 0);
        assert_eq!(result.similarity, 0.95);
        assert_eq!(result.text, "Test text");
    }
}
