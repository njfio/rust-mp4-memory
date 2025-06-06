//! Memory comparison and diff functionality for analyzing changes between memory videos

use std::collections::{HashMap, HashSet};
use std::path::Path;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::retriever::MemvidRetriever;
use crate::index::IndexManager;
use crate::text::TextChunk;

/// Represents the difference between two memory videos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDiff {
    pub old_memory: String,
    pub new_memory: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub summary: DiffSummary,
    pub added_chunks: Vec<ChunkDiff>,
    pub removed_chunks: Vec<ChunkDiff>,
    pub modified_chunks: Vec<ChunkModification>,
    pub unchanged_chunks: Vec<ChunkDiff>,
    pub semantic_changes: Vec<SemanticChange>,
}

/// Summary statistics of the diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub total_old_chunks: usize,
    pub total_new_chunks: usize,
    pub added_count: usize,
    pub removed_count: usize,
    pub modified_count: usize,
    pub unchanged_count: usize,
    pub similarity_score: f64, // 0.0 to 1.0
    pub content_growth_ratio: f64, // new_size / old_size
}

/// Represents a chunk in a diff context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDiff {
    pub chunk_id: usize,
    pub content: String,
    pub source: Option<String>,
    pub frame_number: u32,
    pub similarity_to_counterpart: Option<f64>,
}

/// Represents a modification between two similar chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkModification {
    pub old_chunk: ChunkDiff,
    pub new_chunk: ChunkDiff,
    pub similarity_score: f64,
    pub change_type: ChangeType,
    pub text_diff: TextDiff,
}

/// Types of changes detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    ContentUpdate,
    SourceChange,
    MinorEdit,
    MajorRewrite,
    Expansion,
    Reduction,
}

/// Detailed text differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDiff {
    pub added_text: Vec<String>,
    pub removed_text: Vec<String>,
    pub common_text: Vec<String>,
    pub edit_distance: usize,
}

/// Semantic changes detected through embedding analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChange {
    pub topic: String,
    pub old_representation: Vec<String>, // Representative chunks from old memory
    pub new_representation: Vec<String>, // Representative chunks from new memory
    pub change_magnitude: f64,
    pub change_direction: String, // "expansion", "reduction", "shift", "new", "removed"
}

/// Memory diff engine for comparing memory videos
pub struct MemoryDiffEngine {
    config: Config,
    similarity_threshold: f64,
    semantic_analysis_enabled: bool,
}

impl MemoryDiffEngine {
    /// Create a new memory diff engine
    pub fn new(config: Config) -> Self {
        Self {
            config,
            similarity_threshold: 0.8, // Chunks with >80% similarity are considered "modified"
            semantic_analysis_enabled: true,
        }
    }

    /// Create with custom similarity threshold
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Enable or disable semantic analysis
    pub fn with_semantic_analysis(mut self, enabled: bool) -> Self {
        self.semantic_analysis_enabled = enabled;
        self
    }

    /// Compare two memory videos and generate a comprehensive diff
    pub async fn compare_memories(
        &self,
        old_video: &str,
        old_index: &str,
        new_video: &str,
        new_index: &str,
    ) -> Result<MemoryDiff> {
        info!("Comparing memories: {} vs {}", old_video, new_video);

        // Load both memories
        let old_retriever = MemvidRetriever::new_with_config(old_video, old_index, self.config.clone()).await?;
        let new_retriever = MemvidRetriever::new_with_config(new_video, new_index, self.config.clone()).await?;

        // Load index data
        let old_index_manager = IndexManager::load_readonly(old_index)?;
        let new_index_manager = IndexManager::load_readonly(new_index)?;

        let old_chunks = old_index_manager.get_all_chunks();
        let new_chunks = new_index_manager.get_all_chunks();

        info!("Loaded {} old chunks and {} new chunks", old_chunks.len(), new_chunks.len());

        // Perform chunk-level comparison
        let (added, removed, modified, unchanged) = self.compare_chunks(&old_chunks, &new_chunks).await?;

        // Calculate summary statistics
        let summary = self.calculate_summary(&old_chunks, &new_chunks, &added, &removed, &modified, &unchanged);

        // Perform semantic analysis if enabled
        let semantic_changes = if self.semantic_analysis_enabled {
            self.analyze_semantic_changes(&old_chunks, &new_chunks).await?
        } else {
            Vec::new()
        };

        Ok(MemoryDiff {
            old_memory: old_video.to_string(),
            new_memory: new_video.to_string(),
            timestamp: chrono::Utc::now(),
            summary,
            added_chunks: added,
            removed_chunks: removed,
            modified_chunks: modified,
            unchanged_chunks: unchanged,
            semantic_changes,
        })
    }

    /// Compare chunks between two memories
    async fn compare_chunks(
        &self,
        old_chunks: &[TextChunk],
        new_chunks: &[TextChunk],
    ) -> Result<(Vec<ChunkDiff>, Vec<ChunkDiff>, Vec<ChunkModification>, Vec<ChunkDiff>)> {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut unchanged = Vec::new();

        // Create maps for efficient lookup
        let old_content_map: HashMap<String, &TextChunk> = old_chunks.iter()
            .map(|chunk| (chunk.content.clone(), chunk))
            .collect();

        let new_content_map: HashMap<String, &TextChunk> = new_chunks.iter()
            .map(|chunk| (chunk.content.clone(), chunk))
            .collect();

        let mut processed_new: HashSet<String> = HashSet::new();

        // Find removed and modified chunks
        for old_chunk in old_chunks {
            if let Some(new_chunk) = new_content_map.get(&old_chunk.content) {
                // Exact match - unchanged
                unchanged.push(ChunkDiff {
                    chunk_id: old_chunk.metadata.id,
                    content: old_chunk.content.clone(),
                    source: old_chunk.metadata.source.clone(),
                    frame_number: old_chunk.metadata.frame,
                    similarity_to_counterpart: Some(1.0),
                });
                processed_new.insert(old_chunk.content.clone());
            } else {
                // Look for similar chunks
                let mut best_match: Option<(&TextChunk, f64)> = None;
                
                for new_chunk in new_chunks {
                    if processed_new.contains(&new_chunk.content) {
                        continue;
                    }
                    
                    let similarity = self.calculate_text_similarity(&old_chunk.content, &new_chunk.content);
                    if similarity > self.similarity_threshold {
                        if let Some((_, best_sim)) = best_match {
                            if similarity > best_sim {
                                best_match = Some((new_chunk, similarity));
                            }
                        } else {
                            best_match = Some((new_chunk, similarity));
                        }
                    }
                }

                if let Some((matched_chunk, similarity)) = best_match {
                    // Modified chunk
                    let text_diff = self.calculate_text_diff(&old_chunk.content, &matched_chunk.content);
                    let change_type = self.classify_change(&old_chunk.content, &matched_chunk.content, similarity);

                    modified.push(ChunkModification {
                        old_chunk: ChunkDiff {
                            chunk_id: old_chunk.metadata.id,
                            content: old_chunk.content.clone(),
                            source: old_chunk.metadata.source.clone(),
                            frame_number: old_chunk.metadata.frame,
                            similarity_to_counterpart: Some(similarity),
                        },
                        new_chunk: ChunkDiff {
                            chunk_id: matched_chunk.metadata.id,
                            content: matched_chunk.content.clone(),
                            source: matched_chunk.metadata.source.clone(),
                            frame_number: matched_chunk.metadata.frame,
                            similarity_to_counterpart: Some(similarity),
                        },
                        similarity_score: similarity,
                        change_type,
                        text_diff,
                    });

                    processed_new.insert(matched_chunk.content.clone());
                } else {
                    // Removed chunk
                    removed.push(ChunkDiff {
                        chunk_id: old_chunk.metadata.id,
                        content: old_chunk.content.clone(),
                        source: old_chunk.metadata.source.clone(),
                        frame_number: old_chunk.metadata.frame,
                        similarity_to_counterpart: None,
                    });
                }
            }
        }

        // Find added chunks
        for new_chunk in new_chunks {
            if !processed_new.contains(&new_chunk.content) && !old_content_map.contains_key(&new_chunk.content) {
                added.push(ChunkDiff {
                    chunk_id: new_chunk.metadata.id,
                    content: new_chunk.content.clone(),
                    source: new_chunk.metadata.source.clone(),
                    frame_number: new_chunk.metadata.frame,
                    similarity_to_counterpart: None,
                });
            }
        }

        debug!("Comparison complete: {} added, {} removed, {} modified, {} unchanged", 
               added.len(), removed.len(), modified.len(), unchanged.len());

        Ok((added, removed, modified, unchanged))
    }

    /// Calculate text similarity between two strings
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        // Simple Jaccard similarity based on words
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Calculate detailed text diff
    fn calculate_text_diff(&self, old_text: &str, new_text: &str) -> TextDiff {
        let old_words: HashSet<&str> = old_text.split_whitespace().collect();
        let new_words: HashSet<&str> = new_text.split_whitespace().collect();
        
        let added: Vec<String> = new_words.difference(&old_words).map(|s| s.to_string()).collect();
        let removed: Vec<String> = old_words.difference(&new_words).map(|s| s.to_string()).collect();
        let common: Vec<String> = old_words.intersection(&new_words).map(|s| s.to_string()).collect();
        
        // Simple edit distance (Levenshtein distance approximation)
        let edit_distance = (added.len() + removed.len()).max(
            (old_text.len() as i32 - new_text.len() as i32).abs() as usize
        );

        TextDiff {
            added_text: added,
            removed_text: removed,
            common_text: common,
            edit_distance,
        }
    }

    /// Classify the type of change
    fn classify_change(&self, old_text: &str, new_text: &str, similarity: f64) -> ChangeType {
        let old_len = old_text.len();
        let new_len = new_text.len();
        let size_ratio = new_len as f64 / old_len as f64;

        if similarity > 0.95 {
            ChangeType::MinorEdit
        } else if similarity < 0.5 {
            ChangeType::MajorRewrite
        } else if size_ratio > 1.5 {
            ChangeType::Expansion
        } else if size_ratio < 0.5 {
            ChangeType::Reduction
        } else {
            ChangeType::ContentUpdate
        }
    }

    /// Calculate summary statistics
    fn calculate_summary(
        &self,
        old_chunks: &[TextChunk],
        new_chunks: &[TextChunk],
        added: &[ChunkDiff],
        removed: &[ChunkDiff],
        modified: &[ChunkModification],
        unchanged: &[ChunkDiff],
    ) -> DiffSummary {
        let total_old = old_chunks.len();
        let total_new = new_chunks.len();
        
        let similarity_score = if total_old == 0 && total_new == 0 {
            1.0
        } else if total_old == 0 || total_new == 0 {
            0.0
        } else {
            unchanged.len() as f64 / total_old.max(total_new) as f64
        };

        let content_growth_ratio = if total_old == 0 {
            if total_new == 0 { 1.0 } else { f64::INFINITY }
        } else {
            total_new as f64 / total_old as f64
        };

        DiffSummary {
            total_old_chunks: total_old,
            total_new_chunks: total_new,
            added_count: added.len(),
            removed_count: removed.len(),
            modified_count: modified.len(),
            unchanged_count: unchanged.len(),
            similarity_score,
            content_growth_ratio,
        }
    }

    /// Analyze semantic changes (placeholder for now)
    async fn analyze_semantic_changes(
        &self,
        _old_chunks: &[TextChunk],
        _new_chunks: &[TextChunk],
    ) -> Result<Vec<SemanticChange>> {
        // TODO: Implement semantic analysis using embeddings
        // This would involve:
        // 1. Clustering chunks by topic
        // 2. Comparing topic distributions between old and new
        // 3. Identifying semantic shifts and new topics
        Ok(Vec::new())
    }
}
