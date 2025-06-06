//! Multi-memory search and analysis functionality

use std::collections::{HashMap, HashSet};
use std::path::Path;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::retriever::{MemvidRetriever, RetrievalResult};
use crate::temporal_analysis::MemorySnapshot;

/// Multi-memory search engine for querying across multiple memory videos
pub struct MultiMemoryEngine {
    config: Config,
    memories: HashMap<String, MemvidRetriever>,
    memory_metadata: HashMap<String, MemoryInfo>,
}

/// Information about a loaded memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub name: String,
    pub video_path: String,
    pub index_path: String,
    pub total_chunks: usize,
    pub total_characters: usize,
    pub creation_time: Option<chrono::DateTime<chrono::Utc>>,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// Search result from multiple memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiMemorySearchResult {
    pub query: String,
    pub total_results: usize,
    pub results_by_memory: HashMap<String, Vec<RetrievalResult>>,
    pub aggregated_results: Vec<AggregatedSearchResult>,
    pub cross_memory_correlations: Vec<CrossMemoryCorrelation>,
    pub search_metadata: SearchMetadata,
}

/// Aggregated search result combining information from multiple memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSearchResult {
    pub text: String,
    pub similarity: f64,
    pub source_memory: String,
    pub chunk_id: usize,
    pub frame_number: u32,
    pub related_results: Vec<RelatedResult>, // Similar results from other memories
    pub temporal_context: Option<TemporalContext>,
}

/// Related result from another memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedResult {
    pub memory_name: String,
    pub text: String,
    pub similarity_to_main: f64,
    pub chunk_id: usize,
}

/// Temporal context for a search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub memory_age_days: f64,
    pub is_latest_version: bool,
    pub has_newer_versions: bool,
    pub evolution_summary: Option<String>,
}

/// Correlation between results from different memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossMemoryCorrelation {
    pub memory1: String,
    pub memory2: String,
    pub correlation_type: CorrelationType,
    pub correlation_strength: f64,
    pub description: String,
    pub related_chunks: Vec<(usize, usize)>, // (chunk_id_memory1, chunk_id_memory2)
}

/// Types of correlations between memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationType {
    Complementary,  // Information that complements each other
    Contradictory,  // Conflicting information
    Evolutionary,   // Same topic evolved over time
    Redundant,      // Duplicate information
    Contextual,     // Related context
}

/// Metadata about the search operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetadata {
    pub search_time_ms: u64,
    pub memories_searched: usize,
    pub total_chunks_searched: usize,
    pub correlation_analysis_enabled: bool,
    pub temporal_analysis_enabled: bool,
}

impl MultiMemoryEngine {
    /// Create a new multi-memory engine
    pub fn new(config: Config) -> Self {
        Self {
            config,
            memories: HashMap::new(),
            memory_metadata: HashMap::new(),
        }
    }

    /// Add a memory to the engine
    pub async fn add_memory(
        &mut self,
        name: &str,
        video_path: &str,
        index_path: &str,
        tags: Vec<String>,
        description: Option<String>,
    ) -> Result<()> {
        info!("Adding memory '{}' from {}", name, video_path);

        if !Path::new(video_path).exists() {
            return Err(MemvidError::file_not_found(video_path));
        }

        if !Path::new(index_path).exists() {
            return Err(MemvidError::file_not_found(index_path));
        }

        // Load the retriever
        let retriever = MemvidRetriever::new_with_config(video_path, index_path, self.config.clone()).await?;
        let stats = retriever.get_stats();

        // Create memory info
        let memory_info = MemoryInfo {
            name: name.to_string(),
            video_path: video_path.to_string(),
            index_path: index_path.to_string(),
            total_chunks: stats.index_stats.total_chunks,
            total_characters: stats.index_stats.total_characters,
            creation_time: None, // TODO: Extract from file metadata
            tags,
            description,
        };

        self.memories.insert(name.to_string(), retriever);
        self.memory_metadata.insert(name.to_string(), memory_info);

        info!("Added memory '{}' with {} chunks", name, stats.index_stats.total_chunks);
        Ok(())
    }

    /// Remove a memory from the engine
    pub fn remove_memory(&mut self, name: &str) -> Result<()> {
        if self.memories.remove(name).is_some() {
            self.memory_metadata.remove(name);
            info!("Removed memory '{}'", name);
            Ok(())
        } else {
            Err(MemvidError::invalid_input(&format!("Memory '{}' not found", name)))
        }
    }

    /// List all loaded memories
    pub fn list_memories(&self) -> Vec<&MemoryInfo> {
        self.memory_metadata.values().collect()
    }

    /// Search across all loaded memories
    pub async fn search_all(
        &self,
        query: &str,
        top_k: usize,
        enable_correlations: bool,
        enable_temporal: bool,
    ) -> Result<MultiMemorySearchResult> {
        let start_time = std::time::Instant::now();
        info!("Searching across {} memories for: '{}'", self.memories.len(), query);

        let mut results_by_memory = HashMap::new();
        let mut total_chunks_searched = 0;

        // Search each memory
        for (name, retriever) in &self.memories {
            match retriever.search_with_metadata(query, top_k).await {
                Ok(results) => {
                    let stats = retriever.get_stats();
                    total_chunks_searched += stats.index_stats.total_chunks;
                    results_by_memory.insert(name.clone(), results);
                    debug!("Found {} results in memory '{}'", results_by_memory[name].len(), name);
                }
                Err(e) => {
                    warn!("Failed to search memory '{}': {}", name, e);
                    results_by_memory.insert(name.clone(), Vec::new());
                }
            }
        }

        // Aggregate and rank results
        let aggregated_results = self.aggregate_results(&results_by_memory, enable_temporal).await?;

        // Find cross-memory correlations
        let correlations = if enable_correlations {
            self.find_correlations(&results_by_memory).await?
        } else {
            Vec::new()
        };

        let search_time = start_time.elapsed().as_millis() as u64;
        let total_results = results_by_memory.values().map(|v| v.len()).sum();

        Ok(MultiMemorySearchResult {
            query: query.to_string(),
            total_results,
            results_by_memory,
            aggregated_results,
            cross_memory_correlations: correlations,
            search_metadata: SearchMetadata {
                search_time_ms: search_time,
                memories_searched: self.memories.len(),
                total_chunks_searched,
                correlation_analysis_enabled: enable_correlations,
                temporal_analysis_enabled: enable_temporal,
            },
        })
    }

    /// Search specific memories by name or tag
    pub async fn search_filtered(
        &self,
        query: &str,
        top_k: usize,
        memory_filter: MemoryFilter,
    ) -> Result<MultiMemorySearchResult> {
        let filtered_memories = self.filter_memories(memory_filter);
        
        let mut results_by_memory = HashMap::new();
        let mut total_chunks_searched = 0;

        for name in filtered_memories {
            if let Some(retriever) = self.memories.get(&name) {
                match retriever.search_with_metadata(query, top_k).await {
                    Ok(results) => {
                        let stats = retriever.get_stats();
                        total_chunks_searched += stats.index_stats.total_chunks;
                        results_by_memory.insert(name.clone(), results);
                    }
                    Err(e) => {
                        warn!("Failed to search memory '{}': {}", name, e);
                    }
                }
            }
        }

        let aggregated_results = self.aggregate_results(&results_by_memory, false).await?;

        Ok(MultiMemorySearchResult {
            query: query.to_string(),
            total_results: results_by_memory.values().map(|v| v.len()).sum(),
            results_by_memory,
            aggregated_results,
            cross_memory_correlations: Vec::new(),
            search_metadata: SearchMetadata {
                search_time_ms: 0,
                memories_searched: self.memories.len(),
                total_chunks_searched,
                correlation_analysis_enabled: false,
                temporal_analysis_enabled: false,
            },
        })
    }

    /// Aggregate results from multiple memories
    async fn aggregate_results(
        &self,
        results_by_memory: &HashMap<String, Vec<RetrievalResult>>,
        enable_temporal: bool,
    ) -> Result<Vec<AggregatedSearchResult>> {
        let mut aggregated = Vec::new();

        for (memory_name, results) in results_by_memory {
            for result in results {
                let related_results = self.find_related_results(result, results_by_memory, memory_name).await?;
                
                let temporal_context = if enable_temporal {
                    self.build_temporal_context(memory_name, result).await?
                } else {
                    None
                };

                aggregated.push(AggregatedSearchResult {
                    text: result.text.clone(),
                    similarity: result.similarity as f64,
                    source_memory: memory_name.clone(),
                    chunk_id: result.chunk_id,
                    frame_number: result.frame_number,
                    related_results,
                    temporal_context,
                });
            }
        }

        // Sort by similarity score
        aggregated.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        Ok(aggregated)
    }

    /// Find related results from other memories
    async fn find_related_results(
        &self,
        main_result: &RetrievalResult,
        all_results: &HashMap<String, Vec<RetrievalResult>>,
        exclude_memory: &str,
    ) -> Result<Vec<RelatedResult>> {
        let mut related = Vec::new();
        let similarity_threshold = 0.7;

        for (memory_name, results) in all_results {
            if memory_name == exclude_memory {
                continue;
            }

            for result in results {
                let similarity = self.calculate_text_similarity(&main_result.text, &result.text);
                if similarity > similarity_threshold {
                    related.push(RelatedResult {
                        memory_name: memory_name.clone(),
                        text: result.text.clone(),
                        similarity_to_main: similarity,
                        chunk_id: result.chunk_id,
                    });
                }
            }
        }

        // Sort by similarity and take top 3
        related.sort_by(|a, b| b.similarity_to_main.partial_cmp(&a.similarity_to_main).unwrap_or(std::cmp::Ordering::Equal));
        related.truncate(3);

        Ok(related)
    }

    /// Build temporal context for a result
    async fn build_temporal_context(&self, _memory_name: &str, _result: &RetrievalResult) -> Result<Option<TemporalContext>> {
        // TODO: Implement temporal context analysis
        // This would require tracking memory creation times and versions
        Ok(None)
    }

    /// Find correlations between memories
    async fn find_correlations(&self, _results_by_memory: &HashMap<String, Vec<RetrievalResult>>) -> Result<Vec<CrossMemoryCorrelation>> {
        // TODO: Implement correlation analysis
        // This would involve comparing results across memories to find:
        // - Complementary information
        // - Contradictory information
        // - Evolutionary changes
        // - Redundant content
        Ok(Vec::new())
    }

    /// Filter memories based on criteria
    fn filter_memories(&self, filter: MemoryFilter) -> Vec<String> {
        match filter {
            MemoryFilter::All => self.memory_metadata.keys().cloned().collect(),
            MemoryFilter::Names(names) => names.into_iter().filter(|name| self.memory_metadata.contains_key(name)).collect(),
            MemoryFilter::Tags(tags) => {
                self.memory_metadata.iter()
                    .filter(|(_, info)| tags.iter().any(|tag| info.tags.contains(tag)))
                    .map(|(name, _)| name.clone())
                    .collect()
            }
            MemoryFilter::CreatedAfter(date) => {
                self.memory_metadata.iter()
                    .filter(|(_, info)| {
                        info.creation_time.map_or(false, |time| time > date)
                    })
                    .map(|(name, _)| name.clone())
                    .collect()
            }
        }
    }

    /// Calculate text similarity between two strings
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        use std::collections::HashSet;
        
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

    /// Get statistics about all loaded memories
    pub fn get_global_stats(&self) -> GlobalMemoryStats {
        let total_memories = self.memories.len();
        let total_chunks = self.memory_metadata.values().map(|info| info.total_chunks).sum();
        let total_characters = self.memory_metadata.values().map(|info| info.total_characters).sum();
        
        let all_tags: HashSet<String> = self.memory_metadata.values()
            .flat_map(|info| info.tags.iter().cloned())
            .collect();

        GlobalMemoryStats {
            total_memories,
            total_chunks,
            total_characters,
            unique_tags: all_tags.len(),
            memory_names: self.memory_metadata.keys().cloned().collect(),
        }
    }
}

/// Filter criteria for memory selection
#[derive(Debug, Clone)]
pub enum MemoryFilter {
    All,
    Names(Vec<String>),
    Tags(Vec<String>),
    CreatedAfter(chrono::DateTime<chrono::Utc>),
}

/// Global statistics across all memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMemoryStats {
    pub total_memories: usize,
    pub total_chunks: usize,
    pub total_characters: usize,
    pub unique_tags: usize,
    pub memory_names: Vec<String>,
}
