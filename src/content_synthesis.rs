//! Intelligent content synthesis for generating insights and summaries from memory content

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use tracing::warn;

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::knowledge_graph::{KnowledgeGraph, ConceptNode, ConceptRelationship};
use crate::retriever::MemvidRetriever;
use crate::text::TextChunk;

/// Intelligent content synthesis engine
pub struct ContentSynthesizer {
    config: Config,
    knowledge_graph: Option<KnowledgeGraph>,
    synthesis_strategies: Vec<Box<dyn SynthesisStrategy>>,
}

/// Trait for different synthesis strategies
pub trait SynthesisStrategy: Send + Sync {
    fn synthesize(&self, context: &SynthesisContext) -> Result<SynthesisResult>;
    fn get_strategy_name(&self) -> &str;
    fn get_confidence_threshold(&self) -> f64;
}

/// Context for synthesis operations
#[derive(Debug, Clone)]
pub struct SynthesisContext {
    pub query: String,
    pub relevant_chunks: Vec<TextChunk>,
    pub relevant_concepts: Vec<ConceptNode>,
    pub relevant_relationships: Vec<ConceptRelationship>,
    pub temporal_context: Option<TemporalContext>,
    pub synthesis_type: SynthesisType,
}

/// Types of synthesis operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SynthesisType {
    Summary,           // Generate comprehensive summary
    Insights,          // Extract key insights and patterns
    Contradictions,    // Identify conflicting information
    KnowledgeGaps,     // Identify missing information
    Trends,            // Analyze temporal trends
    Recommendations,   // Generate actionable recommendations
    Connections,       // Find unexpected connections
    Evolution,         // Track concept evolution
}

/// Temporal context for synthesis
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub time_range: (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>),
    pub evolution_patterns: Vec<EvolutionPattern>,
    pub activity_periods: Vec<ActivityPeriod>,
}

/// Pattern of evolution over time
#[derive(Debug, Clone)]
pub struct EvolutionPattern {
    pub concept: String,
    pub pattern_type: String, // "growing", "declining", "cyclical", "stable"
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Period of high activity
#[derive(Debug, Clone)]
pub struct ActivityPeriod {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
    pub activity_type: String,
    pub intensity: f64,
    pub key_concepts: Vec<String>,
}

/// Result of synthesis operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub synthesis_type: SynthesisType,
    pub content: String,
    pub confidence: f64,
    pub key_points: Vec<KeyPoint>,
    pub supporting_evidence: Vec<Evidence>,
    pub metadata: SynthesisMetadata,
}

/// Key point extracted from synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPoint {
    pub point: String,
    pub importance: f64,
    pub supporting_chunks: Vec<String>,
    pub related_concepts: Vec<String>,
}

/// Evidence supporting synthesis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub text: String,
    pub source: String,
    pub confidence: f64,
    pub chunk_id: String,
}

/// Metadata about synthesis operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMetadata {
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub strategy_used: String,
    pub chunks_analyzed: usize,
    pub concepts_involved: usize,
    pub processing_time_ms: u64,
}

impl ContentSynthesizer {
    /// Create a new content synthesizer
    pub fn new(config: Config) -> Self {
        Self {
            config,
            knowledge_graph: None,
            synthesis_strategies: Vec::new(),
        }
    }

    /// Set knowledge graph for enhanced synthesis
    pub fn with_knowledge_graph(mut self, graph: KnowledgeGraph) -> Self {
        self.knowledge_graph = Some(graph);
        self
    }

    /// Add synthesis strategies
    pub fn add_strategy(mut self, strategy: Box<dyn SynthesisStrategy>) -> Self {
        self.synthesis_strategies.push(strategy);
        self
    }

    /// Generate comprehensive summary from query
    pub async fn generate_summary(&self, query: &str, memories: &[(String, String)]) -> Result<SynthesisResult> {
        let context = self.build_synthesis_context(query, memories, SynthesisType::Summary).await?;
        self.synthesize_with_best_strategy(&context).await
    }

    /// Extract insights and patterns
    pub async fn extract_insights(&self, query: &str, memories: &[(String, String)]) -> Result<SynthesisResult> {
        let context = self.build_synthesis_context(query, memories, SynthesisType::Insights).await?;
        self.synthesize_with_best_strategy(&context).await
    }

    /// Identify contradictions in content
    pub async fn find_contradictions(&self, query: &str, memories: &[(String, String)]) -> Result<SynthesisResult> {
        let context = self.build_synthesis_context(query, memories, SynthesisType::Contradictions).await?;
        self.synthesize_with_best_strategy(&context).await
    }

    /// Identify knowledge gaps
    pub async fn identify_knowledge_gaps(&self, query: &str, memories: &[(String, String)]) -> Result<SynthesisResult> {
        let context = self.build_synthesis_context(query, memories, SynthesisType::KnowledgeGaps).await?;
        self.synthesize_with_best_strategy(&context).await
    }

    /// Generate recommendations
    pub async fn generate_recommendations(&self, query: &str, memories: &[(String, String)]) -> Result<SynthesisResult> {
        let context = self.build_synthesis_context(query, memories, SynthesisType::Recommendations).await?;
        self.synthesize_with_best_strategy(&context).await
    }

    /// Build synthesis context from query and memories
    async fn build_synthesis_context(
        &self,
        query: &str,
        memories: &[(String, String)],
        synthesis_type: SynthesisType,
    ) -> Result<SynthesisContext> {
        let mut all_chunks = Vec::new();
        let mut relevant_concepts = Vec::new();
        let mut relevant_relationships = Vec::new();

        // Search all memories for relevant content
        for (video_path, index_path) in memories {
            let retriever = MemvidRetriever::new_with_config(video_path, index_path, self.config.clone()).await?;
            let search_results = retriever.search_with_metadata(query, 20).await?;
            
            // Convert search results to chunks (this would need to be implemented)
            // For now, we'll create placeholder chunks
            for result in search_results {
                let chunk = TextChunk {
                    content: result.text.clone(),
                    metadata: crate::text::ChunkMetadata {
                        id: result.chunk_id,
                        source: Some("memory".to_string()),
                        page: None,
                        char_offset: 0,
                        length: result.text.len(),
                        frame: result.frame_number,
                        extra: std::collections::HashMap::new(),
                    },
                };
                all_chunks.push(chunk);
            }
        }

        // Extract relevant concepts and relationships from knowledge graph
        if let Some(ref graph) = self.knowledge_graph {
            // Find concepts related to the query
            let query_words: HashSet<&str> = query.split_whitespace().collect();
            
            for concept in graph.nodes.values() {
                let concept_words: HashSet<&str> = concept.name.split_whitespace().collect();
                let overlap = query_words.intersection(&concept_words).count();
                
                if overlap > 0 || concept.name.to_lowercase().contains(&query.to_lowercase()) {
                    relevant_concepts.push(concept.clone());
                }
            }

            // Find relationships involving relevant concepts
            let relevant_concept_ids: HashSet<String> = relevant_concepts.iter()
                .map(|c| c.id.clone())
                .collect();

            for relationship in graph.relationships.values() {
                if relevant_concept_ids.contains(&relationship.source_concept) ||
                   relevant_concept_ids.contains(&relationship.target_concept) {
                    relevant_relationships.push(relationship.clone());
                }
            }
        }

        Ok(SynthesisContext {
            query: query.to_string(),
            relevant_chunks: all_chunks,
            relevant_concepts,
            relevant_relationships,
            temporal_context: None, // TODO: Build temporal context
            synthesis_type,
        })
    }

    /// Synthesize content using the best available strategy
    async fn synthesize_with_best_strategy(&self, context: &SynthesisContext) -> Result<SynthesisResult> {
        let start_time = std::time::Instant::now();

        // Try each strategy and pick the best result
        let mut best_result: Option<SynthesisResult> = None;
        let mut best_confidence = 0.0;

        for strategy in &self.synthesis_strategies {
            match strategy.synthesize(context) {
                Ok(result) => {
                    if result.confidence > best_confidence && result.confidence >= strategy.get_confidence_threshold() {
                        best_confidence = result.confidence;
                        best_result = Some(result);
                    }
                }
                Err(e) => {
                    warn!("Strategy {} failed: {}", strategy.get_strategy_name(), e);
                }
            }
        }

        let processing_time = start_time.elapsed().as_millis() as u64;

        match best_result {
            Some(mut result) => {
                result.metadata.processing_time_ms = processing_time;
                Ok(result)
            }
            None => {
                // Fallback to basic synthesis
                Ok(self.basic_synthesis(context, processing_time))
            }
        }
    }

    /// Basic synthesis fallback
    fn basic_synthesis(&self, context: &SynthesisContext, processing_time: u64) -> SynthesisResult {
        let content = if context.relevant_chunks.is_empty() {
            "No relevant content found for the given query.".to_string()
        } else {
            format!(
                "Found {} relevant pieces of information related to '{}'. The content covers various aspects including {}.",
                context.relevant_chunks.len(),
                context.query,
                context.relevant_concepts.iter()
                    .take(5)
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let key_points = context.relevant_concepts.iter()
            .take(5)
            .map(|concept| KeyPoint {
                point: format!("Key concept: {}", concept.name),
                importance: concept.importance_score,
                supporting_chunks: concept.related_chunks.clone(),
                related_concepts: vec![concept.id.clone()],
            })
            .collect();

        let supporting_evidence = context.relevant_chunks.iter()
            .take(3)
            .map(|chunk| Evidence {
                text: chunk.content[..100.min(chunk.content.len())].to_string(),
                source: chunk.metadata.source.clone().unwrap_or_default(),
                confidence: 0.7,
                chunk_id: chunk.metadata.id.to_string(),
            })
            .collect();

        SynthesisResult {
            synthesis_type: context.synthesis_type.clone(),
            content,
            confidence: 0.6,
            key_points,
            supporting_evidence,
            metadata: SynthesisMetadata {
                generated_at: chrono::Utc::now(),
                strategy_used: "BasicSynthesis".to_string(),
                chunks_analyzed: context.relevant_chunks.len(),
                concepts_involved: context.relevant_concepts.len(),
                processing_time_ms: processing_time,
            },
        }
    }
}

/// Template-based synthesis strategy
pub struct TemplateSynthesisStrategy {
    templates: HashMap<SynthesisType, String>,
}

impl TemplateSynthesisStrategy {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        
        templates.insert(
            SynthesisType::Summary,
            "Based on the analysis of {chunk_count} pieces of content, here's a comprehensive summary of '{query}':\n\n{key_points}\n\nKey concepts involved: {concepts}\n\nThis analysis reveals {insights}".to_string()
        );

        templates.insert(
            SynthesisType::Insights,
            "Key insights from the analysis of '{query}':\n\n{insights}\n\nThese patterns emerge from {chunk_count} sources and involve {concept_count} key concepts.".to_string()
        );

        templates.insert(
            SynthesisType::Contradictions,
            "Analysis of '{query}' reveals the following contradictions:\n\n{contradictions}\n\nThese conflicts require further investigation and resolution.".to_string()
        );

        Self { templates }
    }
}

impl SynthesisStrategy for TemplateSynthesisStrategy {
    fn synthesize(&self, context: &SynthesisContext) -> Result<SynthesisResult> {
        let template = self.templates.get(&context.synthesis_type)
            .ok_or_else(|| MemvidError::invalid_input("Unsupported synthesis type"))?;

        // Extract key information
        let key_points = context.relevant_concepts.iter()
            .take(5)
            .map(|c| format!("• {}", c.name))
            .collect::<Vec<_>>()
            .join("\n");

        let concepts = context.relevant_concepts.iter()
            .take(10)
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        let insights = match context.synthesis_type {
            SynthesisType::Insights => {
                context.relevant_relationships.iter()
                    .take(3)
                    .map(|r| format!("• {} is related to {} (confidence: {:.1}%)", 
                                   r.source_concept, r.target_concept, r.confidence * 100.0))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => "Multiple patterns and connections identified.".to_string()
        };

        // Fill template
        let content = template
            .replace("{query}", &context.query)
            .replace("{chunk_count}", &context.relevant_chunks.len().to_string())
            .replace("{concept_count}", &context.relevant_concepts.len().to_string())
            .replace("{key_points}", &key_points)
            .replace("{concepts}", &concepts)
            .replace("{insights}", &insights)
            .replace("{contradictions}", "No significant contradictions detected.");

        let key_points_structured = context.relevant_concepts.iter()
            .take(5)
            .map(|concept| KeyPoint {
                point: concept.name.clone(),
                importance: concept.importance_score,
                supporting_chunks: concept.related_chunks.clone(),
                related_concepts: vec![concept.id.clone()],
            })
            .collect();

        let supporting_evidence = context.relevant_chunks.iter()
            .take(3)
            .map(|chunk| Evidence {
                text: chunk.content[..200.min(chunk.content.len())].to_string(),
                source: chunk.metadata.source.clone().unwrap_or_default(),
                confidence: 0.8,
                chunk_id: chunk.metadata.id.to_string(),
            })
            .collect();

        Ok(SynthesisResult {
            synthesis_type: context.synthesis_type.clone(),
            content,
            confidence: 0.75,
            key_points: key_points_structured,
            supporting_evidence,
            metadata: SynthesisMetadata {
                generated_at: chrono::Utc::now(),
                strategy_used: "TemplateSynthesis".to_string(),
                chunks_analyzed: context.relevant_chunks.len(),
                concepts_involved: context.relevant_concepts.len(),
                processing_time_ms: 0, // Will be set by caller
            },
        })
    }

    fn get_strategy_name(&self) -> &str {
        "TemplateSynthesisStrategy"
    }

    fn get_confidence_threshold(&self) -> f64 {
        0.6
    }
}

/// AI-powered synthesis strategy using LLM APIs
pub struct AiSynthesisStrategy {
    api_key: Option<String>,
    model: String,
    base_url: String,
    max_tokens: usize,
    temperature: f64,
}

impl AiSynthesisStrategy {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").ok()
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .or_else(|| std::env::var("OLLAMA_API_KEY").ok()),
            model: std::env::var("AI_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string()),
            base_url: std::env::var("AI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            max_tokens: 2000,
            temperature: 0.7,
        }
    }

    pub fn with_config(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key: Some(api_key),
            model,
            base_url,
            max_tokens: 2000,
            temperature: 0.7,
        }
    }

    /// Generate AI-powered synthesis using LLM API
    async fn generate_ai_synthesis(&self, context: &SynthesisContext) -> Result<String> {
        if self.api_key.is_none() {
            return Err(MemvidError::invalid_input("No AI API key configured"));
        }

        let prompt = self.build_synthesis_prompt(context);

        // Try different API endpoints based on base URL
        if self.base_url.contains("openai.com") {
            self.call_openai_api(&prompt).await
        } else if self.base_url.contains("anthropic.com") {
            self.call_anthropic_api(&prompt).await
        } else {
            // Assume Ollama or OpenAI-compatible API
            self.call_openai_compatible_api(&prompt).await
        }
    }

    fn build_synthesis_prompt(&self, context: &SynthesisContext) -> String {
        let synthesis_instruction = match context.synthesis_type {
            SynthesisType::Summary => "Generate a comprehensive summary that captures the key themes, main points, and important details.",
            SynthesisType::Insights => "Extract deep insights, patterns, and non-obvious connections. Focus on what the data reveals beyond surface-level information.",
            SynthesisType::Contradictions => "Identify contradictions, conflicts, or inconsistencies in the information. Highlight areas where sources disagree.",
            SynthesisType::KnowledgeGaps => "Identify gaps in knowledge, missing information, or areas that need further investigation.",
            SynthesisType::Trends => "Analyze trends, patterns over time, and evolutionary changes in the content.",
            SynthesisType::Recommendations => "Generate specific, actionable recommendations based on the analysis.",
            SynthesisType::Connections => "Find unexpected connections, relationships, and interdependencies between concepts.",
            SynthesisType::Evolution => "Analyze how concepts, ideas, or topics have evolved or changed over time.",
        };

        let relevant_content = context.relevant_chunks.iter()
            .take(10) // Limit to avoid token limits
            .map(|chunk| format!("- {}", chunk.content.chars().take(500).collect::<String>()))
            .collect::<Vec<_>>()
            .join("\n");

        let concepts = context.relevant_concepts.iter()
            .take(20)
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"You are an expert knowledge analyst. Your task is to {instruction}

Query: "{query}"

Key Concepts Identified: {concepts}

Relevant Content:
{content}

Instructions:
1. {instruction}
2. Be specific and evidence-based
3. Cite relevant information from the content
4. Provide confidence levels for your conclusions
5. Structure your response clearly with key points
6. Aim for depth and insight, not just summarization

Generate a comprehensive analysis:"#,
            instruction = synthesis_instruction,
            query = context.query,
            concepts = concepts,
            content = relevant_content
        )
    }

    async fn call_openai_api(&self, prompt: &str) -> Result<String> {
        let client = reqwest::Client::new();

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        });

        let response = client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key.as_ref().unwrap()))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| MemvidError::synthesis(format!("OpenAI API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemvidError::synthesis(format!("OpenAI API error: {}", error_text)));
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| MemvidError::synthesis(format!("Failed to parse OpenAI response: {}", e)))?;

        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| MemvidError::synthesis("Invalid OpenAI response format"))?;

        Ok(content.to_string())
    }

    async fn call_anthropic_api(&self, prompt: &str) -> Result<String> {
        let client = reqwest::Client::new();

        let request_body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        let response = client
            .post(&format!("{}/messages", self.base_url))
            .header("x-api-key", self.api_key.as_ref().unwrap())
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| MemvidError::synthesis(format!("Anthropic API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemvidError::synthesis(format!("Anthropic API error: {}", error_text)));
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| MemvidError::synthesis(format!("Failed to parse Anthropic response: {}", e)))?;

        let content = response_json["content"][0]["text"]
            .as_str()
            .ok_or_else(|| MemvidError::synthesis("Invalid Anthropic response format"))?;

        Ok(content.to_string())
    }

    async fn call_openai_compatible_api(&self, prompt: &str) -> Result<String> {
        // Use OpenAI format for Ollama and other compatible APIs
        self.call_openai_api(prompt).await
    }
}

impl SynthesisStrategy for AiSynthesisStrategy {
    fn synthesize(&self, context: &SynthesisContext) -> Result<SynthesisResult> {
        // Since we need async but the trait is sync, we'll use block_in_place
        let content = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.generate_ai_synthesis(context).await
            })
        })?;

        // Parse the AI response to extract structured information
        let key_points = self.extract_key_points(&content, context);
        let supporting_evidence = self.extract_evidence(&content, context);

        Ok(SynthesisResult {
            synthesis_type: context.synthesis_type.clone(),
            content,
            confidence: 0.85, // AI synthesis generally has high confidence
            key_points,
            supporting_evidence,
            metadata: SynthesisMetadata {
                generated_at: chrono::Utc::now(),
                strategy_used: "AiSynthesisStrategy".to_string(),
                chunks_analyzed: context.relevant_chunks.len(),
                concepts_involved: context.relevant_concepts.len(),
                processing_time_ms: 0, // Will be set by caller
            },
        })
    }

    fn get_strategy_name(&self) -> &str {
        "AiSynthesisStrategy"
    }

    fn get_confidence_threshold(&self) -> f64 {
        0.8 // Higher threshold for AI synthesis
    }
}

impl AiSynthesisStrategy {
    fn extract_key_points(&self, content: &str, context: &SynthesisContext) -> Vec<KeyPoint> {
        // Simple extraction - look for numbered points or bullet points
        let mut key_points = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("- ") ||
               trimmed.starts_with("• ") ||
               (trimmed.len() > 3 && trimmed.chars().nth(0).unwrap().is_ascii_digit() && trimmed.chars().nth(1) == Some('.')) {

                let point_text = if trimmed.starts_with("- ") || trimmed.starts_with("• ") {
                    trimmed[2..].trim().to_string()
                } else {
                    trimmed[3..].trim().to_string()
                };

                if !point_text.is_empty() && point_text.len() > 10 {
                    key_points.push(KeyPoint {
                        point: point_text,
                        importance: 0.8, // Default importance for AI-extracted points
                        supporting_chunks: context.relevant_chunks.iter()
                            .take(3)
                            .map(|c| c.metadata.id.to_string())
                            .collect(),
                        related_concepts: context.relevant_concepts.iter()
                            .take(3)
                            .map(|c| c.id.clone())
                            .collect(),
                    });
                }
            }
        }

        // If no structured points found, create from concepts
        if key_points.is_empty() {
            key_points = context.relevant_concepts.iter()
                .take(5)
                .map(|concept| KeyPoint {
                    point: format!("Key concept: {}", concept.name),
                    importance: concept.importance_score,
                    supporting_chunks: concept.related_chunks.clone(),
                    related_concepts: vec![concept.id.clone()],
                })
                .collect();
        }

        key_points
    }

    fn extract_evidence(&self, _content: &str, context: &SynthesisContext) -> Vec<Evidence> {
        // Extract evidence from the most relevant chunks
        context.relevant_chunks.iter()
            .take(5)
            .map(|chunk| Evidence {
                text: if chunk.content.len() > 300 {
                    format!("{}...", &chunk.content[..300])
                } else {
                    chunk.content.clone()
                },
                source: chunk.metadata.source.clone().unwrap_or_default(),
                confidence: 0.8,
                chunk_id: chunk.metadata.id.to_string(),
            })
            .collect()
    }
}
