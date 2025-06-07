//! Relationship analysis implementations for knowledge graph generation

use std::collections::{HashMap, HashSet};
use regex::Regex;
use tracing::warn;

use crate::knowledge_graph::{RelationshipAnalyzer, RelationshipCandidate, RelationshipType, ConceptNode};
use crate::text::TextChunk;
use crate::error::Result;

/// Co-occurrence based relationship analyzer
pub struct CooccurrenceAnalyzer {
    window_size: usize,
    min_cooccurrence: usize,
    relationship_patterns: Vec<(Regex, RelationshipType, f64)>,
}

impl CooccurrenceAnalyzer {
    pub fn new() -> Result<Self> {
        let relationship_patterns = vec![
            (Regex::new(r"\b(\w+)\s+is\s+a\s+(\w+)\b")?, RelationshipType::IsA, 0.9),
            (Regex::new(r"\b(\w+)\s+part\s+of\s+(\w+)\b")?, RelationshipType::PartOf, 0.85),
            (Regex::new(r"\b(\w+)\s+causes?\s+(\w+)\b")?, RelationshipType::Causes, 0.8),
            (Regex::new(r"\b(\w+)\s+enables?\s+(\w+)\b")?, RelationshipType::Enables, 0.75),
            (Regex::new(r"\b(\w+)\s+conflicts?\s+with\s+(\w+)\b")?, RelationshipType::Conflicts, 0.8),
            (Regex::new(r"\b(\w+)\s+related\s+to\s+(\w+)\b")?, RelationshipType::RelatedTo, 0.6),
            (Regex::new(r"\b(\w+)\s+and\s+(\w+)\b")?, RelationshipType::RelatedTo, 0.5),
        ];

        Ok(Self {
            window_size: 50, // words
            min_cooccurrence: 2,
            relationship_patterns,
        })
    }
}

impl RelationshipAnalyzer for CooccurrenceAnalyzer {
    fn analyze_relationships(&self, concepts: &[ConceptNode], chunks: &[TextChunk]) -> Result<Vec<RelationshipCandidate>> {
        let mut candidates = Vec::new();
        let mut cooccurrence_matrix: HashMap<(String, String), CooccurrenceData> = HashMap::new();

        // Build concept name to ID mapping
        let concept_names: HashMap<String, String> = concepts.iter()
            .map(|c| (c.name.to_lowercase(), c.id.clone()))
            .collect();

        // Analyze each chunk for concept co-occurrences
        for chunk in chunks {
            let words: Vec<&str> = chunk.content.split_whitespace().collect();
            
            // Find concept mentions in this chunk
            let mut concept_positions: Vec<(usize, String)> = Vec::new();
            
            for (pos, word) in words.iter().enumerate() {
                let normalized = word.to_lowercase();
                let normalized_word = normalized.trim_matches(|c: char| !c.is_alphanumeric());
                if let Some(concept_id) = concept_names.get(normalized_word) {
                    concept_positions.push((pos, concept_id.clone()));
                }
            }

            // Check for multi-word concepts
            for concept in concepts {
                let concept_words: Vec<&str> = concept.name.split_whitespace().collect();
                if concept_words.len() > 1 {
                    for window in words.windows(concept_words.len()) {
                        let window_text = window.join(" ").to_lowercase();
                        if window_text == concept.name.to_lowercase() {
                            let start_pos = words.iter().position(|&w| w == window[0]).unwrap_or(0);
                            concept_positions.push((start_pos, concept.id.clone()));
                        }
                    }
                }
            }

            // Analyze co-occurrences within window
            for i in 0..concept_positions.len() {
                for j in (i + 1)..concept_positions.len() {
                    let (pos1, concept1) = &concept_positions[i];
                    let (pos2, concept2) = &concept_positions[j];
                    
                    if pos1.abs_diff(*pos2) <= self.window_size {
                        let key = if concept1 < concept2 {
                            (concept1.clone(), concept2.clone())
                        } else {
                            (concept2.clone(), concept1.clone())
                        };

                        let entry = cooccurrence_matrix.entry(key).or_insert(CooccurrenceData {
                            count: 0,
                            chunks: Vec::new(),
                            contexts: Vec::new(),
                        });

                        entry.count += 1;
                        entry.chunks.push(chunk.metadata.id.to_string());
                        
                        // Extract context around the co-occurrence
                        let context_start = (*pos1.min(pos2)).saturating_sub(10);
                        let context_end = (*pos1.max(pos2) + 10).min(words.len());
                        let context = words[context_start..context_end].join(" ");
                        entry.contexts.push(context);
                    }
                }
            }

            // Analyze explicit relationship patterns
            for (pattern, rel_type, confidence) in &self.relationship_patterns {
                for captures in pattern.captures_iter(&chunk.content.to_lowercase()) {
                    if let (Some(concept1), Some(concept2)) = (captures.get(1), captures.get(2)) {
                        let concept1_text = concept1.as_str();
                        let concept2_text = concept2.as_str();
                        
                        if let (Some(id1), Some(id2)) = (concept_names.get(concept1_text), concept_names.get(concept2_text)) {
                            candidates.push(RelationshipCandidate {
                                source_concept: id1.clone(),
                                target_concept: id2.clone(),
                                relationship_type: rel_type.clone(),
                                confidence: *confidence,
                                evidence: captures.get(0).unwrap().as_str().to_string(),
                                chunk_ids: vec![chunk.metadata.id.to_string()],
                            });
                        }
                    }
                }
            }
        }

        // Convert significant co-occurrences to relationship candidates
        for ((concept1, concept2), data) in cooccurrence_matrix {
            if data.count >= self.min_cooccurrence {
                let confidence = (data.count as f64 / chunks.len() as f64).min(1.0);
                
                candidates.push(RelationshipCandidate {
                    source_concept: concept1,
                    target_concept: concept2,
                    relationship_type: RelationshipType::RelatedTo,
                    confidence: confidence * 0.7, // Co-occurrence has moderate confidence
                    evidence: format!("Co-occurred {} times", data.count),
                    chunk_ids: data.chunks,
                });
            }
        }

        Ok(candidates)
    }

    fn get_analyzer_name(&self) -> &str {
        "CooccurrenceAnalyzer"
    }
}

#[derive(Debug, Clone)]
struct CooccurrenceData {
    count: usize,
    chunks: Vec<String>,
    contexts: Vec<String>,
}

/// Semantic similarity based relationship analyzer
pub struct SemanticSimilarityAnalyzer {
    similarity_threshold: f64,
    max_relationships_per_concept: usize,
}

impl SemanticSimilarityAnalyzer {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_relationships_per_concept: 10,
        }
    }
}

impl RelationshipAnalyzer for SemanticSimilarityAnalyzer {
    fn analyze_relationships(&self, concepts: &[ConceptNode], _chunks: &[TextChunk]) -> Result<Vec<RelationshipCandidate>> {
        let mut candidates = Vec::new();

        // This would require embeddings to be computed for concepts
        // For now, we'll use a simple text similarity approach
        
        for i in 0..concepts.len() {
            let mut similarities: Vec<(usize, f64)> = Vec::new();
            
            for j in (i + 1)..concepts.len() {
                let similarity = self.calculate_text_similarity(&concepts[i].name, &concepts[j].name);
                if similarity > self.similarity_threshold {
                    similarities.push((j, similarity));
                }
            }

            // Sort by similarity and take top relationships
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            similarities.truncate(self.max_relationships_per_concept);

            for (j, similarity) in similarities {
                candidates.push(RelationshipCandidate {
                    source_concept: concepts[i].id.clone(),
                    target_concept: concepts[j].id.clone(),
                    relationship_type: RelationshipType::RelatedTo,
                    confidence: similarity,
                    evidence: format!("Semantic similarity: {:.2}", similarity),
                    chunk_ids: Vec::new(), // No specific chunks for semantic similarity
                });
            }
        }

        Ok(candidates)
    }

    fn get_analyzer_name(&self) -> &str {
        "SemanticSimilarityAnalyzer"
    }
}

impl SemanticSimilarityAnalyzer {
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        // Simple Jaccard similarity
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
}

/// Temporal relationship analyzer for tracking concept evolution
pub struct TemporalRelationshipAnalyzer {
    time_window_days: i64,
}

impl TemporalRelationshipAnalyzer {
    pub fn new() -> Self {
        Self {
            time_window_days: 30, // Look for relationships within 30 days
        }
    }
}

impl RelationshipAnalyzer for TemporalRelationshipAnalyzer {
    fn analyze_relationships(&self, concepts: &[ConceptNode], chunks: &[TextChunk]) -> Result<Vec<RelationshipCandidate>> {
        let mut candidates = Vec::new();

        // Group chunks by time periods
        let mut time_periods: HashMap<String, Vec<&TextChunk>> = HashMap::new();
        
        for chunk in chunks {
            // For now, we'll use a simple time grouping
            // In a real implementation, we'd extract timestamps from chunk metadata
            let time_key = "2024-01".to_string(); // Placeholder
            time_periods.entry(time_key).or_insert(Vec::new()).push(chunk);
        }

        // Analyze concept evolution across time periods
        for concepts_pair in concepts.windows(2) {
            let concept1 = &concepts_pair[0];
            let concept2 = &concepts_pair[1];

            // Check if concepts appear in sequence across time periods
            let mut temporal_evidence = Vec::new();
            
            for (time_period, period_chunks) in &time_periods {
                let concept1_mentions = period_chunks.iter()
                    .filter(|chunk| chunk.content.to_lowercase().contains(&concept1.name.to_lowercase()))
                    .count();
                
                let concept2_mentions = period_chunks.iter()
                    .filter(|chunk| chunk.content.to_lowercase().contains(&concept2.name.to_lowercase()))
                    .count();

                if concept1_mentions > 0 && concept2_mentions > 0 {
                    temporal_evidence.push(format!("Both concepts mentioned in {}", time_period));
                }
            }

            if !temporal_evidence.is_empty() {
                candidates.push(RelationshipCandidate {
                    source_concept: concept1.id.clone(),
                    target_concept: concept2.id.clone(),
                    relationship_type: RelationshipType::Temporal,
                    confidence: 0.6,
                    evidence: temporal_evidence.join("; "),
                    chunk_ids: Vec::new(),
                });
            }
        }

        Ok(candidates)
    }

    fn get_analyzer_name(&self) -> &str {
        "TemporalRelationshipAnalyzer"
    }
}

/// Hierarchical relationship analyzer for detecting is-a and part-of relationships
pub struct HierarchicalAnalyzer {
    hierarchy_patterns: Vec<(Regex, RelationshipType)>,
}

impl HierarchicalAnalyzer {
    pub fn new() -> Result<Self> {
        let hierarchy_patterns = vec![
            (Regex::new(r"\b(\w+)\s+is\s+a\s+(type|kind|form|example)\s+of\s+(\w+)\b")?, RelationshipType::IsA),
            (Regex::new(r"\b(\w+)\s+inherits?\s+from\s+(\w+)\b")?, RelationshipType::IsA),
            (Regex::new(r"\b(\w+)\s+extends?\s+(\w+)\b")?, RelationshipType::IsA),
            (Regex::new(r"\b(\w+)\s+is\s+part\s+of\s+(\w+)\b")?, RelationshipType::PartOf),
            (Regex::new(r"\b(\w+)\s+belongs\s+to\s+(\w+)\b")?, RelationshipType::PartOf),
            (Regex::new(r"\b(\w+)\s+contains?\s+(\w+)\b")?, RelationshipType::PartOf),
            (Regex::new(r"\b(\w+)\s+includes?\s+(\w+)\b")?, RelationshipType::PartOf),
        ];

        Ok(Self {
            hierarchy_patterns,
        })
    }
}

impl RelationshipAnalyzer for HierarchicalAnalyzer {
    fn analyze_relationships(&self, concepts: &[ConceptNode], chunks: &[TextChunk]) -> Result<Vec<RelationshipCandidate>> {
        let mut candidates = Vec::new();

        // Build concept name to ID mapping
        let concept_names: HashMap<String, String> = concepts.iter()
            .map(|c| (c.name.to_lowercase(), c.id.clone()))
            .collect();

        for chunk in chunks {
            let text = chunk.content.to_lowercase();
            
            for (pattern, relationship_type) in &self.hierarchy_patterns {
                for captures in pattern.captures_iter(&text) {
                    if captures.len() >= 3 {
                        let concept1_text = captures.get(1).unwrap().as_str();
                        let concept2_text = captures.get(captures.len() - 1).unwrap().as_str();
                        
                        if let (Some(id1), Some(id2)) = (concept_names.get(concept1_text), concept_names.get(concept2_text)) {
                            candidates.push(RelationshipCandidate {
                                source_concept: id1.clone(),
                                target_concept: id2.clone(),
                                relationship_type: relationship_type.clone(),
                                confidence: 0.85,
                                evidence: captures.get(0).unwrap().as_str().to_string(),
                                chunk_ids: vec![chunk.metadata.id.to_string()],
                            });
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    fn get_analyzer_name(&self) -> &str {
        "HierarchicalAnalyzer"
    }
}
