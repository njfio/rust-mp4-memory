//! Index management for storing and retrieving chunk metadata and embeddings

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::embeddings::{Embedding, EmbeddingModel, VectorIndex, SearchResult};
use crate::text::{TextChunk, ChunkMetadata};
use crate::config::Config;
use crate::error::{MemvidError, Result};

/// Index manager for handling chunk storage and retrieval
pub struct IndexManager {
    /// Vector index for semantic search
    vector_index: VectorIndex,
    
    /// Chunk metadata storage
    chunks: Vec<ChunkMetadata>,
    
    /// Mapping from chunk ID to text content
    chunk_texts: HashMap<usize, String>,
    
    /// Embedding model for generating embeddings
    embedding_model: Option<EmbeddingModel>,
    
    /// Configuration
    config: Config,
}

impl IndexManager {
    /// Create a new index manager
    pub async fn new(config: Config) -> Result<Self> {
        let embedding_model = EmbeddingModel::load(config.embeddings.clone()).await?;
        let embedding_dim = embedding_model.embedding_dim();
        
        Ok(Self {
            vector_index: VectorIndex::new(embedding_dim),
            chunks: Vec::new(),
            chunk_texts: HashMap::new(),
            embedding_model: Some(embedding_model),
            config,
        })
    }

    /// Create index manager without embedding model (for loading existing index)
    pub fn new_without_model(config: Config) -> Self {
        Self {
            vector_index: VectorIndex::new(384), // Default dimension
            chunks: Vec::new(),
            chunk_texts: HashMap::new(),
            embedding_model: None,
            config,
        }
    }

    /// Add text chunks to the index
    pub async fn add_chunks(&mut self, chunks: Vec<TextChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Extract texts for embedding
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        
        // Generate embeddings
        let embeddings = if let Some(ref model) = self.embedding_model {
            model.embed_batch(&texts).await?
        } else {
            return Err(MemvidError::embedding("No embedding model available"));
        };

        // Add to vector index and store metadata
        for (chunk, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            let chunk_id = self.chunks.len();
            
            // Create metadata for vector index
            let mut vector_metadata = HashMap::new();
            vector_metadata.insert("id".to_string(), chunk_id.to_string());
            vector_metadata.insert("frame".to_string(), chunk.metadata.frame.to_string());
            vector_metadata.insert("text".to_string(), chunk.content.clone());
            
            if let Some(ref source) = chunk.metadata.source {
                vector_metadata.insert("source".to_string(), source.clone());
            }

            // Add to vector index
            self.vector_index.add_embedding(embedding, vector_metadata)?;
            
            // Store chunk metadata and text
            self.chunks.push(chunk.metadata);
            self.chunk_texts.insert(chunk_id, chunk.content);
        }

        Ok(())
    }

    /// Search for similar chunks
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ChunkSearchResult>> {
        if let Some(ref model) = self.embedding_model {
            // Generate query embedding
            let query_embedding = model.embed_text(query).await?;
            
            // Search vector index
            let search_results = self.vector_index.search(&query_embedding, top_k)?;
            
            // Convert to chunk search results
            let mut results = Vec::new();
            for result in search_results {
                if let Ok(chunk_id) = result.metadata.get("id").unwrap_or(&"0".to_string()).parse::<usize>() {
                    if let (Some(metadata), Some(text)) = (self.chunks.get(chunk_id), self.chunk_texts.get(&chunk_id)) {
                        results.push(ChunkSearchResult {
                            chunk_id,
                            similarity: result.similarity,
                            metadata: metadata.clone(),
                            text: text.clone(),
                        });
                    }
                }
            }
            
            Ok(results)
        } else {
            Err(MemvidError::search("No embedding model available for search"))
        }
    }

    /// Get chunk by ID
    pub fn get_chunk(&self, chunk_id: usize) -> Option<(&ChunkMetadata, &String)> {
        if let (Some(metadata), Some(text)) = (self.chunks.get(chunk_id), self.chunk_texts.get(&chunk_id)) {
            Some((metadata, text))
        } else {
            None
        }
    }

    /// Get chunk metadata by ID
    pub fn get_chunk_metadata(&self, chunk_id: usize) -> Option<&ChunkMetadata> {
        self.chunks.get(chunk_id)
    }

    /// Get chunk text by ID
    pub fn get_chunk_text(&self, chunk_id: usize) -> Option<&String> {
        self.chunk_texts.get(&chunk_id)
    }

    /// Get chunks by frame numbers
    pub fn get_chunks_by_frames(&self, frame_numbers: &[u32]) -> Vec<(usize, &ChunkMetadata, &String)> {
        let mut results = Vec::new();
        
        for (chunk_id, metadata) in self.chunks.iter().enumerate() {
            if frame_numbers.contains(&metadata.frame) {
                if let Some(text) = self.chunk_texts.get(&chunk_id) {
                    results.push((chunk_id, metadata, text));
                }
            }
        }
        
        results
    }

    /// Get context window around a chunk
    pub fn get_context_window(&self, chunk_id: usize, window_size: usize) -> Vec<(usize, &ChunkMetadata, &String)> {
        let mut results = Vec::new();
        
        let start = chunk_id.saturating_sub(window_size);
        let end = std::cmp::min(chunk_id + window_size + 1, self.chunks.len());
        
        for id in start..end {
            if let (Some(metadata), Some(text)) = (self.chunks.get(id), self.chunk_texts.get(&id)) {
                results.push((id, metadata, text));
            }
        }
        
        results
    }

    /// Save index to files
    pub fn save(&self, base_path: &str) -> Result<()> {
        let base_path = Path::new(base_path);
        
        // Save vector index
        let vector_index_path = base_path.with_extension("vector");
        self.vector_index.save(vector_index_path.to_str().unwrap())?;
        
        // Save metadata and texts
        let metadata_path = base_path.with_extension("metadata");
        let index_data = IndexData {
            chunks: self.chunks.clone(),
            chunk_texts: self.chunk_texts.clone(),
            config: self.config.clone(),
        };
        
        let serialized = serde_json::to_vec_pretty(&index_data)?;
        std::fs::write(metadata_path, serialized)?;
        
        Ok(())
    }

    /// Load index from files
    pub async fn load(base_path: &str) -> Result<Self> {
        let base_path = Path::new(base_path);
        
        // Load metadata and texts
        let metadata_path = base_path.with_extension("metadata");
        let data = std::fs::read(metadata_path)?;
        let index_data: IndexData = serde_json::from_slice(&data)?;
        
        // Load vector index
        let vector_index_path = base_path.with_extension("vector");
        let vector_index = VectorIndex::load(vector_index_path.to_str().unwrap())?;
        
        // Load embedding model
        let embedding_model = EmbeddingModel::load(index_data.config.embeddings.clone()).await?;
        
        Ok(Self {
            vector_index,
            chunks: index_data.chunks,
            chunk_texts: index_data.chunk_texts,
            embedding_model: Some(embedding_model),
            config: index_data.config,
        })
    }

    /// Load index without embedding model (for read-only operations)
    pub fn load_readonly(base_path: &str) -> Result<Self> {
        let base_path = Path::new(base_path);
        
        // Load metadata and texts
        let metadata_path = base_path.with_extension("metadata");
        let data = std::fs::read(metadata_path)?;
        let index_data: IndexData = serde_json::from_slice(&data)?;
        
        // Load vector index
        let vector_index_path = base_path.with_extension("vector");
        let vector_index = VectorIndex::load(vector_index_path.to_str().unwrap())?;
        
        Ok(Self {
            vector_index,
            chunks: index_data.chunks,
            chunk_texts: index_data.chunk_texts,
            embedding_model: None,
            config: index_data.config,
        })
    }

    /// Get index statistics
    pub fn get_stats(&self) -> IndexStats {
        let total_chunks = self.chunks.len();
        let total_characters: usize = self.chunk_texts.values().map(|text| text.len()).sum();
        let avg_chunk_length = if total_chunks > 0 {
            total_characters as f64 / total_chunks as f64
        } else {
            0.0
        };

        let sources: std::collections::HashSet<_> = self.chunks
            .iter()
            .filter_map(|chunk| chunk.source.as_ref())
            .collect();

        IndexStats {
            total_chunks,
            total_characters,
            avg_chunk_length,
            unique_sources: sources.len(),
            embedding_dimension: if self.vector_index.is_empty() { 0 } else { 
                // This is a simplification - in practice we'd store the dimension
                384 // Default BERT dimension
            },
            has_embedding_model: self.embedding_model.is_some(),
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.vector_index = VectorIndex::new(if let Some(ref model) = self.embedding_model {
            model.embedding_dim()
        } else {
            384
        });
        self.chunks.clear();
        self.chunk_texts.clear();
    }

    /// Get total number of chunks
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Get all chunks as TextChunk objects
    pub fn get_all_chunks(&self) -> Vec<crate::text::TextChunk> {
        let mut chunks = Vec::new();
        for (chunk_id, metadata) in self.chunks.iter().enumerate() {
            if let Some(text) = self.chunk_texts.get(&chunk_id) {
                chunks.push(crate::text::TextChunk {
                    content: text.clone(),
                    metadata: metadata.clone(),
                });
            }
        }
        chunks
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Search result for chunk queries
#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    pub chunk_id: usize,
    pub similarity: f32,
    pub metadata: ChunkMetadata,
    pub text: String,
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub avg_chunk_length: f64,
    pub unique_sources: usize,
    pub embedding_dimension: usize,
    pub has_embedding_model: bool,
}

/// Serializable index data
#[derive(Serialize, Deserialize)]
struct IndexData {
    chunks: Vec<ChunkMetadata>,
    chunk_texts: HashMap<usize, String>,
    config: Config,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::TextProcessor;

    #[tokio::test]
    async fn test_index_manager() {
        let config = Config::default();
        
        // This test would require a real embedding model
        // For now, just test the structure
        let manager = IndexManager::new_without_model(config);
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_index_stats() {
        let config = Config::default();
        let manager = IndexManager::new_without_model(config);
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_characters, 0);
        assert_eq!(stats.unique_sources, 0);
    }
}
