//! Text processing and chunking functionality

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::config::TextConfig;
use crate::error::{MemvidError, Result};

/// Metadata associated with a text chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Unique chunk ID
    pub id: usize,
    
    /// Source document or file
    pub source: Option<String>,
    
    /// Page number (for PDFs)
    pub page: Option<u32>,
    
    /// Character offset in original document
    pub char_offset: usize,
    
    /// Length of the chunk
    pub length: usize,
    
    /// Frame number in video
    pub frame: u32,
    
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

/// A text chunk with its content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// The text content
    pub content: String,
    
    /// Associated metadata
    pub metadata: ChunkMetadata,
}

/// Text processor for chunking and document parsing
pub struct TextProcessor {
    config: TextConfig,
}

impl TextProcessor {
    /// Create a new text processor
    pub fn new(config: TextConfig) -> Self {
        Self { config }
    }

    /// Create processor with default configuration
    pub fn default() -> Self {
        Self::new(crate::config::Config::default().text)
    }

    /// Split text into overlapping chunks
    pub fn chunk_text(&self, text: &str, source: Option<String>) -> Result<Vec<TextChunk>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_id = 0;

        while start < text.len() {
            let end = std::cmp::min(start + self.config.chunk_size, text.len());
            let mut chunk_end = end;

            // Try to break at sentence boundary if not at end of text
            if end < text.len() {
                if let Some(sentence_end) = self.find_sentence_boundary(&text[start..end]) {
                    let absolute_sentence_end = start + sentence_end;
                    // Only use sentence boundary if it's not too small
                    if absolute_sentence_end - start >= self.config.min_chunk_size {
                        chunk_end = absolute_sentence_end;
                    }
                }
            }

            let chunk_text = text[start..chunk_end].trim().to_string();
            
            if !chunk_text.is_empty() && chunk_text.len() >= self.config.min_chunk_size {
                let metadata = ChunkMetadata {
                    id: chunk_id,
                    source: source.clone(),
                    page: None,
                    char_offset: start,
                    length: chunk_text.len(),
                    frame: chunk_id as u32,
                    extra: HashMap::new(),
                };

                chunks.push(TextChunk {
                    content: chunk_text,
                    metadata,
                });

                chunk_id += 1;
            }

            // Move start position with overlap
            start = if chunk_end >= text.len() {
                text.len()
            } else {
                std::cmp::max(chunk_end.saturating_sub(self.config.overlap), start + 1)
            };
        }

        Ok(chunks)
    }

    /// Find the best sentence boundary within a text segment
    fn find_sentence_boundary(&self, text: &str) -> Option<usize> {
        // Look for sentence endings from the end backwards
        let sentence_endings = ['.', '!', '?'];
        
        for (i, ch) in text.char_indices().rev() {
            if sentence_endings.contains(&ch) {
                // Check if this is likely a real sentence ending
                if i + 1 < text.len() {
                    let next_chars: String = text[i + 1..].chars().take(3).collect();
                    if next_chars.starts_with(' ') || next_chars.starts_with('\n') {
                        return Some(i + 1);
                    }
                } else {
                    return Some(i + 1);
                }
            }
        }

        // Fallback to word boundary
        self.find_word_boundary(text)
    }

    /// Find the best word boundary within a text segment
    fn find_word_boundary(&self, text: &str) -> Option<usize> {
        // Look for whitespace from the end backwards
        for (i, ch) in text.char_indices().rev() {
            if ch.is_whitespace() {
                return Some(i);
            }
        }
        None
    }

    /// Process a PDF file and extract text chunks (simplified implementation)
    pub async fn process_pdf(&self, pdf_path: &str) -> Result<Vec<TextChunk>> {
        if !Path::new(pdf_path).exists() {
            return Err(MemvidError::file_not_found(pdf_path));
        }

        // In a real implementation, this would use pdf-extract or similar
        // For now, return an error indicating PDF support is not implemented
        Err(MemvidError::pdf("PDF processing not implemented in simplified version. Use text files instead.".to_string()))
    }

    /// Process an EPUB file and extract text chunks (simplified implementation)
    pub async fn process_epub(&self, epub_path: &str) -> Result<Vec<TextChunk>> {
        if !Path::new(epub_path).exists() {
            return Err(MemvidError::file_not_found(epub_path));
        }

        // In a real implementation, this would use epub and html2text crates
        // For now, return an error indicating EPUB support is not implemented
        Err(MemvidError::epub("EPUB processing not implemented in simplified version. Use text files instead.".to_string()))
    }

    /// Process a plain text file
    pub async fn process_text_file(&self, file_path: &str) -> Result<Vec<TextChunk>> {
        if !Path::new(file_path).exists() {
            return Err(MemvidError::file_not_found(file_path));
        }

        let text = std::fs::read_to_string(file_path)?;
        let source = Some(Path::new(file_path).file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string());

        self.chunk_text(&text, source)
    }

    /// Process multiple files in a directory
    pub async fn process_directory(&self, dir_path: &str) -> Result<Vec<TextChunk>> {
        use walkdir::WalkDir;

        let mut all_chunks = Vec::new();

        for entry in WalkDir::new(dir_path).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();
                let extension = path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                let chunks = match extension.as_str() {
                    "pdf" => self.process_pdf(path.to_str().unwrap()).await?,
                    "epub" => self.process_epub(path.to_str().unwrap()).await?,
                    "txt" | "md" | "rst" => self.process_text_file(path.to_str().unwrap()).await?,
                    _ => continue, // Skip unsupported files
                };

                all_chunks.extend(chunks);
            }
        }

        // Update frame numbers to be sequential
        for (i, chunk) in all_chunks.iter_mut().enumerate() {
            chunk.metadata.frame = i as u32;
            chunk.metadata.id = i;
        }

        Ok(all_chunks)
    }

    /// Merge overlapping chunks if they're too similar
    pub fn deduplicate_chunks(&self, chunks: Vec<TextChunk>) -> Vec<TextChunk> {
        if chunks.len() <= 1 {
            return chunks;
        }

        let mut deduplicated = Vec::new();
        let mut i = 0;

        while i < chunks.len() {
            let current = &chunks[i];
            let mut should_merge = false;

            // Check if next chunk has significant overlap
            if i + 1 < chunks.len() {
                let next = &chunks[i + 1];
                let overlap_ratio = self.calculate_overlap_ratio(&current.content, &next.content);
                
                if overlap_ratio > 0.8 {
                    should_merge = true;
                }
            }

            if should_merge && i + 1 < chunks.len() {
                // Merge current and next chunk
                let next = &chunks[i + 1];
                let merged_content = self.merge_chunk_content(&current.content, &next.content);
                
                let mut merged_metadata = current.metadata.clone();
                merged_metadata.length = merged_content.len();
                
                deduplicated.push(TextChunk {
                    content: merged_content,
                    metadata: merged_metadata,
                });
                
                i += 2; // Skip the next chunk since we merged it
            } else {
                deduplicated.push(current.clone());
                i += 1;
            }
        }

        deduplicated
    }

    /// Calculate overlap ratio between two text chunks
    fn calculate_overlap_ratio(&self, text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Merge content from two overlapping chunks
    fn merge_chunk_content(&self, content1: &str, content2: &str) -> String {
        // Simple merge: take the longer content
        // In a more sophisticated implementation, we could find the actual overlap
        // and merge more intelligently
        if content1.len() >= content2.len() {
            content1.to_string()
        } else {
            content2.to_string()
        }
    }

    /// Get statistics about the chunks
    pub fn get_chunk_stats(&self, chunks: &[TextChunk]) -> ChunkStats {
        if chunks.is_empty() {
            return ChunkStats::default();
        }

        let lengths: Vec<usize> = chunks.iter().map(|c| c.content.len()).collect();
        let total_length: usize = lengths.iter().sum();
        let avg_length = total_length as f64 / chunks.len() as f64;
        
        let mut sorted_lengths = lengths.clone();
        sorted_lengths.sort_unstable();
        
        let median_length = if sorted_lengths.len() % 2 == 0 {
            (sorted_lengths[sorted_lengths.len() / 2 - 1] + sorted_lengths[sorted_lengths.len() / 2]) / 2
        } else {
            sorted_lengths[sorted_lengths.len() / 2]
        };

        ChunkStats {
            total_chunks: chunks.len(),
            total_characters: total_length,
            avg_chunk_length: avg_length,
            median_chunk_length: median_length,
            min_chunk_length: *lengths.iter().min().unwrap_or(&0),
            max_chunk_length: *lengths.iter().max().unwrap_or(&0),
            sources: chunks.iter()
                .filter_map(|c| c.metadata.source.as_ref())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }
}

/// Statistics about text chunks
#[derive(Debug, Clone, Default)]
pub struct ChunkStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub avg_chunk_length: f64,
    pub median_chunk_length: usize,
    pub min_chunk_length: usize,
    pub max_chunk_length: usize,
    pub sources: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_chunking() {
        let processor = TextProcessor::default();
        // Create a longer text that will definitely create chunks
        let text = "This is the first sentence with enough content to meet the minimum chunk size requirements. This is the second sentence that also has sufficient length to be processed correctly. This is the third sentence that continues the pattern of having adequate length for proper chunking. This is the fourth sentence that ensures we have enough content for multiple chunks. This is the fifth sentence that completes our test text with sufficient length.";

        let chunks = processor.chunk_text(text, None).unwrap();
        assert!(!chunks.is_empty());

        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.content.len() >= processor.config.min_chunk_size);
        }
    }

    #[test]
    fn test_sentence_boundary() {
        let processor = TextProcessor::default();
        let text = "First sentence. Second sentence";
        
        let boundary = processor.find_sentence_boundary(text);
        assert_eq!(boundary, Some(15)); // After "First sentence."
    }

    #[test]
    fn test_overlap_calculation() {
        let processor = TextProcessor::default();
        let text1 = "the quick brown fox";
        let text2 = "brown fox jumps over";
        
        let ratio = processor.calculate_overlap_ratio(text1, text2);
        assert!(ratio > 0.0 && ratio < 1.0);
    }

    #[tokio::test]
    async fn test_chunk_stats() {
        let processor = TextProcessor::default();
        let chunks = vec![
            TextChunk {
                content: "Short".to_string(),
                metadata: ChunkMetadata {
                    id: 0,
                    source: None,
                    page: None,
                    char_offset: 0,
                    length: 5,
                    frame: 0,
                    extra: HashMap::new(),
                },
            },
            TextChunk {
                content: "This is a longer chunk".to_string(),
                metadata: ChunkMetadata {
                    id: 1,
                    source: None,
                    page: None,
                    char_offset: 5,
                    length: 22,
                    frame: 1,
                    extra: HashMap::new(),
                },
            },
        ];
        
        let stats = processor.get_chunk_stats(&chunks);
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.min_chunk_length, 5);
        assert_eq!(stats.max_chunk_length, 22);
    }
}
