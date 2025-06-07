//! Knowledge graph generation and analysis for intelligent content relationships

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::retriever::MemvidRetriever;
use crate::text::TextChunk;
use crate::embeddings::EmbeddingModel;

/// Represents a concept node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: String,
    pub name: String,
    pub concept_type: ConceptType,
    pub importance_score: f64,
    pub frequency: usize,
    pub related_chunks: Vec<String>, // Chunk IDs that mention this concept
    pub embedding: Option<Vec<f32>>,
    pub metadata: ConceptMetadata,
}

/// Types of concepts in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptType {
    Entity,        // People, places, organizations
    Topic,         // Subject areas, themes
    Keyword,       // Important terms
    Relationship,  // Connections between entities
    Event,         // Time-based occurrences
    Process,       // Procedures, methodologies
    Concept,       // Abstract ideas
}

/// Metadata about a concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMetadata {
    pub first_seen: chrono::DateTime<chrono::Utc>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub source_memories: Vec<String>,
    pub confidence_score: f64,
    pub aliases: Vec<String>,
    pub description: Option<String>,
}

/// Represents a relationship between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelationship {
    pub id: String,
    pub source_concept: String,
    pub target_concept: String,
    pub relationship_type: RelationshipType,
    pub strength: f64,
    pub evidence_chunks: Vec<String>,
    pub confidence: f64,
    pub temporal_pattern: Option<TemporalPattern>,
}

/// Types of relationships between concepts
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub enum RelationshipType {
    IsA,           // Hierarchical relationship
    PartOf,        // Component relationship
    RelatedTo,     // General association
    Causes,        // Causal relationship
    Enables,       // Enabling relationship
    Conflicts,     // Contradictory relationship
    Temporal,      // Time-based relationship
    Spatial,       // Location-based relationship
    Functional,    // Functional dependency
}

/// Temporal pattern in relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_type: String, // "increasing", "decreasing", "cyclical", "stable"
    pub strength_over_time: Vec<(chrono::DateTime<chrono::Utc>, f64)>,
    pub trend_direction: f64, // -1.0 to 1.0
}

/// Complete knowledge graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub nodes: HashMap<String, ConceptNode>,
    pub relationships: HashMap<String, ConceptRelationship>,
    pub metadata: GraphMetadata,
    pub communities: Vec<ConceptCommunity>,
    pub temporal_evolution: Vec<GraphSnapshot>,
}

/// Metadata about the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub total_concepts: usize,
    pub total_relationships: usize,
    pub source_memories: Vec<String>,
    pub generation_algorithm: String,
    pub confidence_threshold: f64,
}

/// Community of related concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCommunity {
    pub id: String,
    pub name: String,
    pub concepts: Vec<String>,
    pub central_concepts: Vec<String>,
    pub cohesion_score: f64,
    pub topic_summary: Option<String>,
}

/// Snapshot of graph at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub concept_count: usize,
    pub relationship_count: usize,
    pub top_concepts: Vec<String>,
    pub emerging_concepts: Vec<String>,
    pub declining_concepts: Vec<String>,
}

/// Knowledge graph builder and analyzer
pub struct KnowledgeGraphBuilder {
    config: Config,
    embedding_model: Option<EmbeddingModel>,
    concept_extractors: Vec<Box<dyn ConceptExtractor>>,
    relationship_analyzers: Vec<Box<dyn RelationshipAnalyzer>>,
}

/// Trait for extracting concepts from text
pub trait ConceptExtractor: Send + Sync {
    fn extract_concepts(&self, text: &str, chunk_id: &str) -> Result<Vec<ConceptCandidate>>;
    fn get_extractor_name(&self) -> &str;
}

/// Trait for analyzing relationships between concepts
pub trait RelationshipAnalyzer: Send + Sync {
    fn analyze_relationships(&self, concepts: &[ConceptNode], chunks: &[TextChunk]) -> Result<Vec<RelationshipCandidate>>;
    fn get_analyzer_name(&self) -> &str;
}

/// Candidate concept before validation
#[derive(Debug, Clone)]
pub struct ConceptCandidate {
    pub name: String,
    pub concept_type: ConceptType,
    pub confidence: f64,
    pub context: String,
    pub chunk_id: String,
}

/// Candidate relationship before validation
#[derive(Debug, Clone)]
pub struct RelationshipCandidate {
    pub source_concept: String,
    pub target_concept: String,
    pub relationship_type: RelationshipType,
    pub confidence: f64,
    pub evidence: String,
    pub chunk_ids: Vec<String>,
}

impl KnowledgeGraphBuilder {
    /// Create a new knowledge graph builder
    pub fn new(config: Config) -> Self {
        Self {
            config,
            embedding_model: None,
            concept_extractors: Vec::new(),
            relationship_analyzers: Vec::new(),
        }
    }

    /// Initialize with embedding model
    pub async fn with_embeddings(mut self) -> Result<Self> {
        let embedding_model = EmbeddingModel::load(self.config.embeddings.clone()).await?;
        self.embedding_model = Some(embedding_model);
        Ok(self)
    }

    /// Add concept extractors
    pub fn add_concept_extractor(mut self, extractor: Box<dyn ConceptExtractor>) -> Self {
        self.concept_extractors.push(extractor);
        self
    }

    /// Add relationship analyzers
    pub fn add_relationship_analyzer(mut self, analyzer: Box<dyn RelationshipAnalyzer>) -> Self {
        self.relationship_analyzers.push(analyzer);
        self
    }

    /// Build knowledge graph from memory videos
    pub async fn build_from_memories(&self, memory_paths: &[(String, String)]) -> Result<KnowledgeGraph> {
        info!("Building knowledge graph from {} memories", memory_paths.len());

        let mut all_chunks = Vec::new();
        let mut source_memories = Vec::new();

        // Load all chunks from memories
        for (video_path, index_path) in memory_paths {
            let retriever = MemvidRetriever::new_with_config(video_path, index_path, self.config.clone()).await?;
            let chunks = self.load_chunks_from_retriever(&retriever).await?;
            all_chunks.extend(chunks);
            source_memories.push(video_path.clone());
        }

        info!("Loaded {} total chunks for analysis", all_chunks.len());

        // Extract concepts from all chunks
        let concept_candidates = self.extract_all_concepts(&all_chunks).await?;
        info!("Extracted {} concept candidates", concept_candidates.len());

        // Validate and consolidate concepts
        let concepts = self.validate_and_consolidate_concepts(concept_candidates).await?;
        info!("Validated {} unique concepts", concepts.len());

        // Analyze relationships between concepts
        let relationship_candidates = self.analyze_all_relationships(&concepts, &all_chunks).await?;
        info!("Found {} relationship candidates", relationship_candidates.len());

        // Validate relationships
        let relationships = self.validate_relationships(relationship_candidates).await?;
        info!("Validated {} relationships", relationships.len());

        // Detect communities
        let communities = self.detect_communities(&concepts, &relationships).await?;
        info!("Detected {} concept communities", communities.len());

        // Build final graph
        let graph = KnowledgeGraph {
            nodes: concepts.into_iter().map(|c| (c.id.clone(), c)).collect(),
            relationships: relationships.into_iter().map(|r| (r.id.clone(), r)).collect(),
            metadata: GraphMetadata {
                created_at: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
                total_concepts: 0, // Will be updated
                total_relationships: 0, // Will be updated
                source_memories,
                generation_algorithm: "multi_extractor_v1".to_string(),
                confidence_threshold: 0.7,
            },
            communities,
            temporal_evolution: Vec::new(),
        };

        info!("Knowledge graph built successfully");
        Ok(graph)
    }

    /// Load chunks from a retriever
    async fn load_chunks_from_retriever(&self, retriever: &MemvidRetriever) -> Result<Vec<TextChunk>> {
        // This would need to be implemented in the retriever
        // For now, return empty vector
        warn!("load_chunks_from_retriever not yet implemented");
        Ok(Vec::new())
    }

    /// Extract concepts from all chunks
    async fn extract_all_concepts(&self, chunks: &[TextChunk]) -> Result<Vec<ConceptCandidate>> {
        let mut all_candidates = Vec::new();

        for chunk in chunks {
            for extractor in &self.concept_extractors {
                let candidates = extractor.extract_concepts(&chunk.content, &chunk.metadata.id.to_string())?;
                all_candidates.extend(candidates);
            }
        }

        Ok(all_candidates)
    }

    /// Validate and consolidate concept candidates
    async fn validate_and_consolidate_concepts(&self, candidates: Vec<ConceptCandidate>) -> Result<Vec<ConceptNode>> {
        let mut concept_map: HashMap<String, ConceptNode> = HashMap::new();

        for candidate in candidates {
            if candidate.confidence < 0.5 {
                continue; // Skip low-confidence candidates
            }

            let concept_id = self.normalize_concept_name(&candidate.name);
            
            if let Some(existing) = concept_map.get_mut(&concept_id) {
                // Update existing concept
                existing.frequency += 1;
                existing.related_chunks.push(candidate.chunk_id);
                existing.importance_score = (existing.importance_score + candidate.confidence) / 2.0;
            } else {
                // Create new concept
                let concept = ConceptNode {
                    id: concept_id.clone(),
                    name: candidate.name,
                    concept_type: candidate.concept_type,
                    importance_score: candidate.confidence,
                    frequency: 1,
                    related_chunks: vec![candidate.chunk_id],
                    embedding: None, // Will be computed later
                    metadata: ConceptMetadata {
                        first_seen: chrono::Utc::now(),
                        last_seen: chrono::Utc::now(),
                        source_memories: Vec::new(),
                        confidence_score: candidate.confidence,
                        aliases: Vec::new(),
                        description: None,
                    },
                };
                concept_map.insert(concept_id, concept);
            }
        }

        Ok(concept_map.into_values().collect())
    }

    /// Analyze relationships between concepts
    async fn analyze_all_relationships(&self, concepts: &[ConceptNode], chunks: &[TextChunk]) -> Result<Vec<RelationshipCandidate>> {
        let mut all_candidates = Vec::new();

        for analyzer in &self.relationship_analyzers {
            let candidates = analyzer.analyze_relationships(concepts, chunks)?;
            all_candidates.extend(candidates);
        }

        Ok(all_candidates)
    }

    /// Validate relationship candidates
    async fn validate_relationships(&self, candidates: Vec<RelationshipCandidate>) -> Result<Vec<ConceptRelationship>> {
        let mut relationships = Vec::new();

        for candidate in candidates {
            if candidate.confidence < 0.6 {
                continue; // Skip low-confidence relationships
            }

            let relationship = ConceptRelationship {
                id: format!("{}_{}_{}_{:?}", 
                           candidate.source_concept, 
                           candidate.target_concept,
                           candidate.relationship_type as u8,
                           chrono::Utc::now().timestamp()),
                source_concept: candidate.source_concept,
                target_concept: candidate.target_concept,
                relationship_type: candidate.relationship_type,
                strength: candidate.confidence,
                evidence_chunks: candidate.chunk_ids,
                confidence: candidate.confidence,
                temporal_pattern: None, // Will be computed in temporal analysis
            };

            relationships.push(relationship);
        }

        Ok(relationships)
    }

    /// Detect communities of related concepts
    async fn detect_communities(&self, concepts: &[ConceptNode], relationships: &[ConceptRelationship]) -> Result<Vec<ConceptCommunity>> {
        // Implement community detection algorithm (e.g., Louvain algorithm)
        // For now, return empty vector
        warn!("Community detection not yet implemented");
        Ok(Vec::new())
    }

    /// Normalize concept names for deduplication
    fn normalize_concept_name(&self, name: &str) -> String {
        name.to_lowercase()
            .trim()
            .replace(&[' ', '-', '_'][..], "_")
    }
}
