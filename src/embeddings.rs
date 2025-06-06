//! Text embeddings and semantic search functionality
//!
//! This is a simplified implementation for the initial version.
//! In a full implementation, this would use transformer models like BERT.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::EmbeddingConfig;
use crate::error::{MemvidError, Result};

/// Text embedding vector
pub type Embedding = Vec<f32>;

/// Embedding model for generating text embeddings
/// This is a simplified implementation using basic text features
pub struct EmbeddingModel {
    config: EmbeddingConfig,
    dimension: usize,
}

impl EmbeddingModel {
    /// Load embedding model (simplified implementation)
    pub async fn load(config: EmbeddingConfig) -> Result<Self> {
        // In a real implementation, this would load a transformer model
        // For now, we'll use a simple TF-IDF-like approach
        Ok(Self {
            config,
            dimension: 384, // Standard BERT-base dimension
        })
    }

    /// Generate embedding for a single text
    pub async fn embed_text(&self, text: &str) -> Result<Embedding> {
        // Simplified embedding using basic text features
        self.simple_text_embedding(text)
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.simple_text_embedding(text)?);
        }
        Ok(embeddings)
    }

    /// Simple text embedding using basic features
    fn simple_text_embedding(&self, text: &str) -> Result<Embedding> {
        // This is a very simplified embedding approach
        // In a real implementation, you would use a proper transformer model

        let lowercase_text = text.to_lowercase();
        let words: Vec<&str> = lowercase_text
            .split_whitespace()
            .collect();

        let mut embedding = vec![0.0f32; self.dimension];

        // Simple bag-of-words with position encoding
        for (i, word) in words.iter().enumerate() {
            let word_hash = self.simple_hash(word) as usize;
            let pos_weight = 1.0 / (1.0 + i as f32 * 0.1); // Position decay

            // Distribute word influence across multiple dimensions
            for j in 0..10 {
                let dim_idx = (word_hash + j * 37) % self.dimension;
                embedding[dim_idx] += pos_weight * 0.1;
            }
        }

        // Add text length feature
        let length_feature = (text.len() as f32).ln() / 10.0;
        embedding[0] += length_feature;

        // Normalize to unit vector
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        Ok(embedding)
    }

    /// Simple hash function for words
    fn simple_hash(&self, word: &str) -> u32 {
        let mut hash = 5381u32;
        for byte in word.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u32);
        }
        hash
    }

    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, a: &Embedding, b: &Embedding) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.dimension
    }
}

/// Vector search index for fast similarity search
pub struct VectorIndex {
    embeddings: Vec<Embedding>,
    metadata: Vec<HashMap<String, String>>,
    dimension: usize,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(dimension: usize) -> Self {
        Self {
            embeddings: Vec::new(),
            metadata: Vec::new(),
            dimension,
        }
    }

    /// Add embedding with metadata to the index
    pub fn add_embedding(&mut self, embedding: Embedding, metadata: HashMap<String, String>) -> Result<usize> {
        if embedding.len() != self.dimension {
            return Err(MemvidError::embedding(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        let id = self.embeddings.len();
        self.embeddings.push(embedding);
        self.metadata.push(metadata);
        Ok(id)
    }

    /// Search for similar embeddings
    pub fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != self.dimension {
            return Err(MemvidError::search(format!(
                "Query embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                query_embedding.len()
            )));
        }

        let mut results = Vec::new();

        // Calculate similarities
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let similarity = self.cosine_similarity(query_embedding, embedding);
            results.push(SearchResult {
                id: i,
                similarity,
                metadata: self.metadata[i].clone(),
            });
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        // Return top k results
        results.truncate(top_k);
        Ok(results)
    }

    /// Calculate cosine similarity between two embeddings
    fn cosine_similarity(&self, a: &Embedding, b: &Embedding) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Get number of embeddings in index
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Save index to file
    pub fn save(&self, path: &str) -> Result<()> {
        let data = IndexData {
            embeddings: self.embeddings.clone(),
            metadata: self.metadata.clone(),
            dimension: self.dimension,
        };

        let serialized = serde_json::to_vec(&data)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load index from file
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let index_data: IndexData = serde_json::from_slice(&data)?;

        Ok(Self {
            embeddings: index_data.embeddings,
            metadata: index_data.metadata,
            dimension: index_data.dimension,
        })
    }
}

/// Search result from vector index
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub similarity: f32,
    pub metadata: HashMap<String, String>,
}

/// Serializable index data
#[derive(Serialize, Deserialize)]
struct IndexData {
    embeddings: Vec<Embedding>,
    metadata: Vec<HashMap<String, String>>,
    dimension: usize,
}

// FAISS support is disabled in this simplified version
// In a full implementation, you would uncomment and implement FAISS integration

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index() {
        let mut index = VectorIndex::new(3);
        
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0];
        let embedding3 = vec![0.5, 0.5, 0.0];
        
        let mut metadata = HashMap::new();
        metadata.insert("text".to_string(), "test".to_string());
        
        index.add_embedding(embedding1.clone(), metadata.clone()).unwrap();
        index.add_embedding(embedding2.clone(), metadata.clone()).unwrap();
        index.add_embedding(embedding3.clone(), metadata.clone()).unwrap();
        
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // Should be most similar to embedding1
    }

    #[test]
    fn test_cosine_similarity() {
        let index = VectorIndex::new(3);
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((index.cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((index.cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
}
