//! Concept extraction implementations for knowledge graph generation

use std::collections::{HashMap, HashSet};
use regex::Regex;
use tracing::warn;

use crate::knowledge_graph::{ConceptExtractor, ConceptCandidate, ConceptType};
use crate::error::Result;

/// Named Entity Recognition extractor
pub struct NamedEntityExtractor {
    person_patterns: Vec<Regex>,
    organization_patterns: Vec<Regex>,
    location_patterns: Vec<Regex>,
    technical_patterns: Vec<Regex>,
}

impl NamedEntityExtractor {
    pub fn new() -> Result<Self> {
        let person_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")?, // First Last
            Regex::new(r"\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b")?, // Dr. First Last
            Regex::new(r"\bProf\. [A-Z][a-z]+ [A-Z][a-z]+\b")?, // Prof. First Last
        ];

        let organization_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Corporation)\b")?,
            Regex::new(r"\b(University of|MIT|Stanford|Harvard|Google|Microsoft|Apple|Amazon)\b")?,
        ];

        let location_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+, [A-Z]{2}\b")?, // City, State
            Regex::new(r"\b(United States|USA|UK|Canada|Germany|France|Japan|China)\b")?,
        ];

        let technical_patterns = vec![
            Regex::new(r"\b[A-Z]{2,}[a-z]*\b")?, // Acronyms
            Regex::new(r"\b\w+\.(js|py|rs|cpp|java|go)\b")?, // File extensions
            Regex::new(r"\b(API|SDK|HTTP|JSON|XML|SQL|NoSQL|REST|GraphQL)\b")?,
        ];

        Ok(Self {
            person_patterns,
            organization_patterns,
            location_patterns,
            technical_patterns,
        })
    }
}

impl ConceptExtractor for NamedEntityExtractor {
    fn extract_concepts(&self, text: &str, chunk_id: &str) -> Result<Vec<ConceptCandidate>> {
        let mut candidates = Vec::new();

        // Extract persons
        for pattern in &self.person_patterns {
            for mat in pattern.find_iter(text) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Entity,
                    confidence: 0.8,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Extract organizations
        for pattern in &self.organization_patterns {
            for mat in pattern.find_iter(text) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Entity,
                    confidence: 0.9,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Extract locations
        for pattern in &self.location_patterns {
            for mat in pattern.find_iter(text) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Entity,
                    confidence: 0.85,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Extract technical terms
        for pattern in &self.technical_patterns {
            for mat in pattern.find_iter(text) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Keyword,
                    confidence: 0.7,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        Ok(candidates)
    }

    fn get_extractor_name(&self) -> &str {
        "NamedEntityExtractor"
    }
}

impl NamedEntityExtractor {
    fn extract_context(&self, text: &str, start: usize, end: usize) -> String {
        let context_size = 50;
        let context_start = start.saturating_sub(context_size);
        let context_end = (end + context_size).min(text.len());
        text[context_start..context_end].to_string()
    }
}

/// Keyword and topic extractor using TF-IDF and frequency analysis
pub struct KeywordExtractor {
    stop_words: HashSet<String>,
    min_frequency: usize,
    min_length: usize,
}

impl KeywordExtractor {
    pub fn new() -> Self {
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "this", "that", "these", "those", "i", "me", "my", "myself",
            "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
            "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "will", "would", "should",
            "could", "can", "may", "might", "must", "shall", "should", "ought", "need", "dare"
        ].iter().map(|s| s.to_string()).collect();

        Self {
            stop_words,
            min_frequency: 2,
            min_length: 3,
        }
    }
}

impl ConceptExtractor for KeywordExtractor {
    fn extract_concepts(&self, text: &str, chunk_id: &str) -> Result<Vec<ConceptCandidate>> {
        let mut candidates = Vec::new();
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        // Tokenize and count words
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() >= self.min_length)
            .filter(|w| !self.stop_words.contains(&w.to_lowercase()))
            .collect();

        for word in &words {
            let normalized = word.to_lowercase();
            *word_freq.entry(normalized).or_insert(0) += 1;
        }

        // Extract n-grams (2-grams and 3-grams)
        let mut ngram_freq: HashMap<String, usize> = HashMap::new();
        
        // 2-grams
        for window in words.windows(2) {
            let ngram = format!("{} {}", window[0].to_lowercase(), window[1].to_lowercase());
            if !self.contains_stop_word(&ngram) {
                *ngram_freq.entry(ngram).or_insert(0) += 1;
            }
        }

        // 3-grams
        for window in words.windows(3) {
            let ngram = format!("{} {} {}", 
                               window[0].to_lowercase(), 
                               window[1].to_lowercase(), 
                               window[2].to_lowercase());
            if !self.contains_stop_word(&ngram) {
                *ngram_freq.entry(ngram).or_insert(0) += 1;
            }
        }

        // Convert frequent words to candidates
        for (word, freq) in word_freq {
            if freq >= self.min_frequency {
                let confidence = (freq as f64 / words.len() as f64).min(1.0);
                candidates.push(ConceptCandidate {
                    name: word,
                    concept_type: ConceptType::Keyword,
                    confidence: confidence * 0.6, // Lower confidence for single words
                    context: text[..100.min(text.len())].to_string(),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Convert frequent n-grams to candidates
        for (ngram, freq) in ngram_freq {
            if freq >= self.min_frequency {
                let confidence = (freq as f64 / (words.len() - 1) as f64).min(1.0);
                candidates.push(ConceptCandidate {
                    name: ngram,
                    concept_type: ConceptType::Topic,
                    confidence: confidence * 0.8, // Higher confidence for phrases
                    context: text[..100.min(text.len())].to_string(),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        Ok(candidates)
    }

    fn get_extractor_name(&self) -> &str {
        "KeywordExtractor"
    }
}

impl KeywordExtractor {
    fn contains_stop_word(&self, text: &str) -> bool {
        text.split_whitespace()
            .any(|word| self.stop_words.contains(&word.to_lowercase()))
    }
}

/// Domain-specific concept extractor for technical content
pub struct TechnicalConceptExtractor {
    programming_patterns: Vec<Regex>,
    algorithm_patterns: Vec<Regex>,
    data_structure_patterns: Vec<Regex>,
    methodology_patterns: Vec<Regex>,
}

impl TechnicalConceptExtractor {
    pub fn new() -> Result<Self> {
        let programming_patterns = vec![
            Regex::new(r"\b(function|class|method|variable|array|object|string|integer|boolean)\b")?,
            Regex::new(r"\b(async|await|promise|callback|closure|lambda|iterator)\b")?,
            Regex::new(r"\b(database|query|index|schema|table|column|row)\b")?,
        ];

        let algorithm_patterns = vec![
            Regex::new(r"\b(algorithm|sorting|searching|optimization|recursion|iteration)\b")?,
            Regex::new(r"\b(binary search|quick sort|merge sort|depth first|breadth first)\b")?,
            Regex::new(r"\b(machine learning|neural network|deep learning|artificial intelligence)\b")?,
        ];

        let data_structure_patterns = vec![
            Regex::new(r"\b(array|list|stack|queue|tree|graph|hash table|linked list)\b")?,
            Regex::new(r"\b(binary tree|heap|trie|graph|matrix|vector)\b")?,
        ];

        let methodology_patterns = vec![
            Regex::new(r"\b(agile|scrum|kanban|waterfall|devops|ci/cd)\b")?,
            Regex::new(r"\b(test driven|behavior driven|domain driven|event driven)\b")?,
            Regex::new(r"\b(microservices|monolith|serverless|containerization)\b")?,
        ];

        Ok(Self {
            programming_patterns,
            algorithm_patterns,
            data_structure_patterns,
            methodology_patterns,
        })
    }
}

impl ConceptExtractor for TechnicalConceptExtractor {
    fn extract_concepts(&self, text: &str, chunk_id: &str) -> Result<Vec<ConceptCandidate>> {
        let mut candidates = Vec::new();
        let text_lower = text.to_lowercase();

        // Extract programming concepts
        for pattern in &self.programming_patterns {
            for mat in pattern.find_iter(&text_lower) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Concept,
                    confidence: 0.85,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Extract algorithms
        for pattern in &self.algorithm_patterns {
            for mat in pattern.find_iter(&text_lower) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Process,
                    confidence: 0.9,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Extract data structures
        for pattern in &self.data_structure_patterns {
            for mat in pattern.find_iter(&text_lower) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Concept,
                    confidence: 0.88,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        // Extract methodologies
        for pattern in &self.methodology_patterns {
            for mat in pattern.find_iter(&text_lower) {
                candidates.push(ConceptCandidate {
                    name: mat.as_str().to_string(),
                    concept_type: ConceptType::Process,
                    confidence: 0.87,
                    context: self.extract_context(text, mat.start(), mat.end()),
                    chunk_id: chunk_id.to_string(),
                });
            }
        }

        Ok(candidates)
    }

    fn get_extractor_name(&self) -> &str {
        "TechnicalConceptExtractor"
    }
}

impl TechnicalConceptExtractor {
    fn extract_context(&self, text: &str, start: usize, end: usize) -> String {
        let context_size = 100;
        let context_start = start.saturating_sub(context_size);
        let context_end = (end + context_size).min(text.len());
        text[context_start..context_end].to_string()
    }
}
