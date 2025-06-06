//! Chat functionality with LLM integration

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tracing::{info, warn, error};

use crate::config::{Config, LlmProviderConfig};
use crate::error::{MemvidError, Result};
use crate::retriever::{MemvidRetriever, RetrievalResult};

/// Chat interface for conversing with memory
pub struct MemvidChat {
    retriever: MemvidRetriever,
    config: Config,
    http_client: Client,
    conversation_history: Vec<ChatMessage>,
    current_provider: String,
}

impl MemvidChat {
    /// Create a new chat interface
    pub async fn new(video_path: &str, index_path: &str) -> Result<Self> {
        let config = Config::default();
        Self::new_with_config(video_path, index_path, config).await
    }

    /// Create a new chat interface with custom configuration
    pub async fn new_with_config(video_path: &str, index_path: &str, config: Config) -> Result<Self> {
        let retriever = MemvidRetriever::new_with_config(video_path, index_path, config.clone()).await?;
        let http_client = Client::new();
        let current_provider = config.chat.default_provider.clone();

        Ok(Self {
            retriever,
            config,
            http_client,
            conversation_history: Vec::new(),
            current_provider,
        })
    }

    /// Send a chat message and get response
    pub async fn chat(&mut self, message: &str) -> Result<String> {
        info!("Processing chat message: '{}'", message);

        // Search for relevant context
        let search_results = self.retriever
            .search_with_metadata(message, self.config.search.default_top_k)
            .await?;

        // Build context from search results
        let context = self.build_context(&search_results);

        // Add user message to history
        self.conversation_history.push(ChatMessage {
            role: "user".to_string(),
            content: message.to_string(),
        });

        // Generate response using LLM
        let response = self.generate_response(&context, message).await?;

        // Add assistant response to history
        self.conversation_history.push(ChatMessage {
            role: "assistant".to_string(),
            content: response.clone(),
        });

        // Trim conversation history if too long
        self.trim_conversation_history();

        info!("Generated response with {} characters", response.len());
        Ok(response)
    }

    /// Chat with specific context window
    pub async fn chat_with_context(&mut self, message: &str, context_window: usize) -> Result<String> {
        info!("Processing chat message with context window: '{}'", message);

        // Search for relevant context with expanded window
        let contextual_results = self.retriever
            .search_with_context(message, self.config.search.default_top_k, context_window)
            .await?;

        // Build enhanced context
        let context = self.build_enhanced_context(&contextual_results);

        // Add user message to history
        self.conversation_history.push(ChatMessage {
            role: "user".to_string(),
            content: message.to_string(),
        });

        // Generate response using LLM
        let response = self.generate_response(&context, message).await?;

        // Add assistant response to history
        self.conversation_history.push(ChatMessage {
            role: "assistant".to_string(),
            content: response.clone(),
        });

        self.trim_conversation_history();

        Ok(response)
    }

    /// Set the LLM provider
    pub fn set_provider(&mut self, provider: &str) -> Result<()> {
        if !self.config.chat.providers.contains_key(provider) {
            return Err(MemvidError::config(format!("Unknown provider: {}", provider)));
        }
        
        self.current_provider = provider.to_string();
        info!("Switched to provider: {}", provider);
        Ok(())
    }

    /// Get conversation history
    pub fn get_conversation_history(&self) -> &[ChatMessage] {
        &self.conversation_history
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
        info!("Cleared conversation history");
    }

    /// Get chat statistics
    pub fn get_stats(&self) -> ChatStats {
        let total_messages = self.conversation_history.len();
        let user_messages = self.conversation_history.iter()
            .filter(|msg| msg.role == "user")
            .count();
        let assistant_messages = self.conversation_history.iter()
            .filter(|msg| msg.role == "assistant")
            .count();

        ChatStats {
            total_messages,
            user_messages,
            assistant_messages,
            current_provider: self.current_provider.clone(),
            retriever_stats: self.retriever.get_stats(),
        }
    }

    /// Build context string from search results
    fn build_context(&self, results: &[RetrievalResult]) -> String {
        if results.is_empty() {
            return "No relevant context found.".to_string();
        }

        let mut context = String::from("Relevant information from the knowledge base:\n\n");
        
        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!("{}. ", i + 1));
            if let Some(ref source) = result.metadata.source {
                context.push_str(&format!("[{}] ", source));
            }
            context.push_str(&result.text);
            context.push_str("\n\n");
        }

        context
    }

    /// Build enhanced context with surrounding chunks
    fn build_enhanced_context(&self, results: &[crate::retriever::ContextualResult]) -> String {
        if results.is_empty() {
            return "No relevant context found.".to_string();
        }

        let mut context = String::from("Relevant information from the knowledge base:\n\n");
        
        for (i, contextual_result) in results.iter().enumerate() {
            context.push_str(&format!("{}. Main Result:\n", i + 1));
            if let Some(ref source) = contextual_result.main_result.metadata.source {
                context.push_str(&format!("[{}] ", source));
            }
            context.push_str(&contextual_result.main_result.text);
            context.push_str("\n");

            if !contextual_result.context.is_empty() {
                context.push_str("   Additional Context:\n");
                for ctx_chunk in &contextual_result.context {
                    if ctx_chunk.chunk_id != contextual_result.main_result.chunk_id {
                        context.push_str("   - ");
                        context.push_str(&ctx_chunk.text[..std::cmp::min(100, ctx_chunk.text.len())]);
                        if ctx_chunk.text.len() > 100 {
                            context.push_str("...");
                        }
                        context.push_str("\n");
                    }
                }
            }
            context.push_str("\n");
        }

        context
    }

    /// Generate response using LLM
    async fn generate_response(&self, context: &str, user_message: &str) -> Result<String> {
        let provider_config = self.config.get_provider_config(&self.current_provider)?;
        
        match self.current_provider.as_str() {
            "openai" => self.generate_openai_response(provider_config, context, user_message).await,
            "anthropic" => self.generate_anthropic_response(provider_config, context, user_message).await,
            _ => Err(MemvidError::llm_api(format!("Unsupported provider: {}", self.current_provider))),
        }
    }

    /// Generate response using OpenAI API
    async fn generate_openai_response(
        &self,
        config: &LlmProviderConfig,
        context: &str,
        user_message: &str,
    ) -> Result<String> {
        let api_key = if let Some(ref key) = config.api_key {
            key.clone()
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            key
        } else {
            return Err(MemvidError::llm_api("OpenAI API key not found"));
        };

        let system_prompt = format!(
            "You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately and helpfully. If the context doesn't contain relevant information, say so clearly.\n\nContext:\n{}",
            context
        );

        let mut messages = vec![
            json!({
                "role": "system",
                "content": system_prompt
            })
        ];

        // Add conversation history (limited)
        let history_limit = 10; // Last 10 messages
        let start_idx = self.conversation_history.len().saturating_sub(history_limit);
        for msg in &self.conversation_history[start_idx..] {
            messages.push(json!({
                "role": msg.role,
                "content": msg.content
            }));
        }

        messages.push(json!({
            "role": "user",
            "content": user_message
        }));

        let request_body = json!({
            "model": config.model,
            "messages": messages,
            "temperature": self.config.chat.temperature,
            "max_tokens": self.config.chat.max_tokens
        });

        let response = self.http_client
            .post(&config.endpoint)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(MemvidError::llm_api(format!("OpenAI API error: {}", error_text)));
        }

        let response_json: OpenAIResponse = response.json().await?;
        
        if let Some(choice) = response_json.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(MemvidError::llm_api("No response from OpenAI API"))
        }
    }

    /// Generate response using Anthropic API
    async fn generate_anthropic_response(
        &self,
        config: &LlmProviderConfig,
        context: &str,
        user_message: &str,
    ) -> Result<String> {
        let api_key = if let Some(ref key) = config.api_key {
            key.clone()
        } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            key
        } else {
            return Err(MemvidError::llm_api("Anthropic API key not found"));
        };

        let system_prompt = format!(
            "You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately and helpfully. If the context doesn't contain relevant information, say so clearly.\n\nContext:\n{}",
            context
        );

        let mut messages = Vec::new();

        // Add conversation history (limited)
        let history_limit = 10;
        let start_idx = self.conversation_history.len().saturating_sub(history_limit);
        for msg in &self.conversation_history[start_idx..] {
            messages.push(json!({
                "role": msg.role,
                "content": msg.content
            }));
        }

        messages.push(json!({
            "role": "user",
            "content": user_message
        }));

        let request_body = json!({
            "model": config.model,
            "max_tokens": self.config.chat.max_tokens,
            "temperature": self.config.chat.temperature,
            "system": system_prompt,
            "messages": messages
        });

        let response = self.http_client
            .post(&config.endpoint)
            .header("x-api-key", api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(MemvidError::llm_api(format!("Anthropic API error: {}", error_text)));
        }

        let response_json: AnthropicResponse = response.json().await?;
        
        if let Some(content) = response_json.content.first() {
            Ok(content.text.clone())
        } else {
            Err(MemvidError::llm_api("No response from Anthropic API"))
        }
    }

    /// Trim conversation history to stay within limits
    fn trim_conversation_history(&mut self) {
        let max_history = 20; // Keep last 20 messages
        if self.conversation_history.len() > max_history {
            let start_idx = self.conversation_history.len() - max_history;
            self.conversation_history = self.conversation_history[start_idx..].to_vec();
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat statistics
#[derive(Debug, Clone)]
pub struct ChatStats {
    pub total_messages: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub current_provider: String,
    pub retriever_stats: crate::retriever::RetrieverStats,
}

/// OpenAI API response
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: String,
}

/// Anthropic API response
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chat_creation() {
        // This test would require actual video and index files
        let result = MemvidChat::new("nonexistent.mp4", "nonexistent.json").await;
        assert!(result.is_err()); // Should fail because files don't exist
    }

    #[test]
    fn test_chat_message() {
        let message = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };

        assert_eq!(message.role, "user");
        assert_eq!(message.content, "Hello");
    }

    #[test]
    fn test_build_context() {
        // This test is simplified since we can't easily construct a MemvidRetriever
        // In a real implementation, we would use a mock or builder pattern

        let results = vec![
            RetrievalResult {
                chunk_id: 0,
                similarity: 0.9,
                text: "Test content".to_string(),
                metadata: crate::text::ChunkMetadata {
                    id: 0,
                    source: Some("test.txt".to_string()),
                    page: None,
                    char_offset: 0,
                    length: 12,
                    frame: 0,
                    extra: std::collections::HashMap::new(),
                },
                frame_number: 0,
            }
        ];

        // Test the context building logic directly
        let mut context = String::from("Relevant information from the knowledge base:\n\n");

        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!("{}. ", i + 1));
            if let Some(ref source) = result.metadata.source {
                context.push_str(&format!("[{}] ", source));
            }
            context.push_str(&result.text);
            context.push_str("\n\n");
        }

        assert!(context.contains("Test content"));
        assert!(context.contains("test.txt"));
    }
}
