use rust_mem_vid::{MemvidEncoder, DataProcessor, DataFileType, Config};
use std::fs;
use tempfile::TempDir;

#[tokio::test]
async fn test_csv_processing() {
    let temp_dir = TempDir::new().unwrap();
    let csv_path = temp_dir.path().join("test.csv");
    
    let csv_content = r#"id,name,value
1,Alice,100
2,Bob,200
3,Carol,300"#;
    
    fs::write(&csv_path, csv_content).unwrap();
    
    let processor = DataProcessor::default();
    let chunks = processor.process_file(csv_path.to_str().unwrap()).await.unwrap();
    
    assert!(!chunks.is_empty());
    assert!(chunks[0].content.contains("Alice"));
    assert_eq!(chunks[0].metadata.file_type, DataFileType::Csv);
}

#[tokio::test]
async fn test_json_processing() {
    let temp_dir = TempDir::new().unwrap();
    let json_path = temp_dir.path().join("test.json");
    
    let json_content = r#"{
  "users": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ]
}"#;
    
    fs::write(&json_path, json_content).unwrap();
    
    let processor = DataProcessor::default();
    let chunks = processor.process_file(json_path.to_str().unwrap()).await.unwrap();
    
    assert!(!chunks.is_empty());
    assert!(chunks[0].content.contains("Alice"));
    assert_eq!(chunks[0].metadata.file_type, DataFileType::Json);
}

#[tokio::test]
async fn test_rust_code_processing() {
    let temp_dir = TempDir::new().unwrap();
    let rust_path = temp_dir.path().join("test.rs");
    
    let rust_content = r#"//! Test module
use std::collections::HashMap;

/// A test function
pub fn hello_world() -> String {
    "Hello, World!".to_string()
}

/// Another test function
pub fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hello() {
        assert_eq!(hello_world(), "Hello, World!");
    }
}"#;
    
    fs::write(&rust_path, rust_content).unwrap();
    
    let processor = DataProcessor::default();
    let chunks = processor.process_file(rust_path.to_str().unwrap()).await.unwrap();
    
    assert!(!chunks.is_empty());
    
    // Check that code metadata is present
    let has_code_metadata = chunks.iter().any(|chunk| {
        chunk.metadata.code_metadata.is_some()
    });
    assert!(has_code_metadata);
    
    // Check that functions are detected
    let has_functions = chunks.iter().any(|chunk| {
        if let Some(ref code_meta) = chunk.metadata.code_metadata {
            !code_meta.functions.is_empty()
        } else {
            false
        }
    });
    assert!(has_functions);
    
    assert_eq!(chunks[0].metadata.file_type, DataFileType::Rust);
}

#[tokio::test]
async fn test_python_code_processing() {
    let temp_dir = TempDir::new().unwrap();
    let python_path = temp_dir.path().join("test.py");
    
    let python_content = r#"#!/usr/bin/env python3
"""Test Python module"""

import os
import sys

def hello_world():
    """Return a greeting"""
    return "Hello, World!"

class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        """Add two numbers"""
        return x + y
    
    def multiply(self, x, y):
        """Multiply two numbers"""
        return x * y

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
"#;
    
    fs::write(&python_path, python_content).unwrap();
    
    let processor = DataProcessor::default();
    let chunks = processor.process_file(python_path.to_str().unwrap()).await.unwrap();
    
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].metadata.file_type, DataFileType::Python);
    
    // Check for code metadata
    let has_code_metadata = chunks.iter().any(|chunk| {
        chunk.metadata.code_metadata.is_some()
    });
    assert!(has_code_metadata);
}

#[tokio::test]
async fn test_log_processing() {
    let temp_dir = TempDir::new().unwrap();
    let log_path = temp_dir.path().join("test.log");
    
    let log_content = r#"2024-01-15 10:30:15 INFO  [main] Application started
2024-01-15 10:30:16 DEBUG [db] Database connection established
2024-01-15 10:30:17 ERROR [api] Failed to process request
2024-01-15 10:30:18 WARN  [cache] Cache miss for key: user_123"#;
    
    fs::write(&log_path, log_content).unwrap();
    
    let processor = DataProcessor::default();
    let chunks = processor.process_file(log_path.to_str().unwrap()).await.unwrap();
    
    assert!(!chunks.is_empty());
    assert!(chunks[0].content.contains("Application started"));
    assert_eq!(chunks[0].metadata.file_type, DataFileType::Log);
}

#[tokio::test]
async fn test_encoder_with_data_files() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create test files
    let csv_path = temp_dir.path().join("data.csv");
    let rust_path = temp_dir.path().join("code.rs");
    let json_path = temp_dir.path().join("config.json");
    
    fs::write(&csv_path, "id,name\n1,Alice\n2,Bob").unwrap();
    fs::write(&rust_path, "fn main() { println!(\"Hello\"); }").unwrap();
    fs::write(&json_path, r#"{"version": "1.0", "debug": true}"#).unwrap();
    
    // Test encoder with data files - skip if embedding model not available
    let config = Config::default();
    let mut encoder = match MemvidEncoder::new_with_config(config).await {
        Ok(encoder) => encoder,
        Err(_) => {
            // Skip test if embedding model not available (e.g., in CI)
            return;
        }
    };
    
    // Add data files
    encoder.add_data_file(csv_path.to_str().unwrap()).await.unwrap();
    encoder.add_data_file(rust_path.to_str().unwrap()).await.unwrap();
    encoder.add_data_file(json_path.to_str().unwrap()).await.unwrap();
    
    // Check that chunks were added
    assert!(encoder.len() >= 3);
    
    // Check that different file types are represented
    let chunk_contents: Vec<String> = encoder.chunks().iter().map(|c| c.content.clone()).collect();
    let has_csv = chunk_contents.iter().any(|c| c.contains("Alice"));
    let has_rust = chunk_contents.iter().any(|c| c.contains("println!"));
    let has_json = chunk_contents.iter().any(|c| c.contains("version"));
    
    assert!(has_csv, "CSV content not found");
    assert!(has_rust, "Rust content not found");
    assert!(has_json, "JSON content not found");
}

#[tokio::test]
async fn test_file_type_detection() {
    assert_eq!(DataFileType::from_extension("csv"), Some(DataFileType::Csv));
    assert_eq!(DataFileType::from_extension("rs"), Some(DataFileType::Rust));
    assert_eq!(DataFileType::from_extension("py"), Some(DataFileType::Python));
    assert_eq!(DataFileType::from_extension("js"), Some(DataFileType::JavaScript));
    assert_eq!(DataFileType::from_extension("json"), Some(DataFileType::Json));
    assert_eq!(DataFileType::from_extension("log"), Some(DataFileType::Log));
    assert_eq!(DataFileType::from_extension("unknown"), None);
}

#[test]
fn test_data_file_type_syntax_language() {
    assert_eq!(DataFileType::Rust.syntax_language(), "rust");
    assert_eq!(DataFileType::Python.syntax_language(), "python");
    assert_eq!(DataFileType::JavaScript.syntax_language(), "javascript");
    assert_eq!(DataFileType::TypeScript.syntax_language(), "typescript");
    assert_eq!(DataFileType::Html.syntax_language(), "html");
    assert_eq!(DataFileType::Css.syntax_language(), "css");
}

#[tokio::test]
async fn test_auto_file_detection() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create test files with different extensions
    let files = vec![
        ("test.csv", "id,name\n1,Alice", DataFileType::Csv),
        ("test.rs", "fn main() {}", DataFileType::Rust),
        ("test.py", "def hello(): pass", DataFileType::Python),
        ("test.js", "function hello() {}", DataFileType::JavaScript),
        ("test.json", r#"{"key": "value"}"#, DataFileType::Json),
    ];
    
    let config = Config::default();
    let mut encoder = match MemvidEncoder::new_with_config(config).await {
        Ok(encoder) => encoder,
        Err(_) => {
            // Skip test if embedding model not available (e.g., in CI)
            return;
        }
    };

    for (filename, content, _expected_type) in files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).unwrap();
        
        // Test auto-detection through add_file
        encoder.add_file(file_path.to_str().unwrap()).await.unwrap();
    }
    
    // Should have added chunks for all files
    assert!(encoder.len() >= 5);
}
