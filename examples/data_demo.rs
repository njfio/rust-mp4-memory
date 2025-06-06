use rust_mem_vid::{MemvidEncoder, Config, DataProcessor, DataFileType};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🎬 Rust MemVid Data Processing Demo");
    println!("===================================");
    
    // Create sample data files for demonstration
    create_sample_files().await?;
    
    // Create encoder
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;
    
    let start = Instant::now();
    
    // Process different file types
    println!("\n📊 Processing CSV data...");
    encoder.add_csv_file("sample_data.csv").await?;
    
    println!("🐍 Processing Python code...");
    encoder.add_code_file("sample_code.py").await?;
    
    println!("🦀 Processing Rust code...");
    encoder.add_code_file("sample_code.rs").await?;
    
    println!("📋 Processing log file...");
    encoder.add_log_file("sample.log").await?;
    
    println!("📄 Processing JSON data...");
    encoder.add_data_file("sample_data.json").await?;
    
    let processing_time = start.elapsed();
    
    // Build the video
    println!("\n🎥 Building video...");
    let video_path = "data_demo.mp4";
    let index_path = "data_demo_index.json";
    
    let stats = encoder.build_video(video_path, index_path).await?;
    let total_time = start.elapsed();
    
    println!("\n✅ Data processing demo completed!");
    println!("📊 Statistics:");
    println!("   • Total chunks: {}", stats.total_chunks);
    println!("   • Total characters: {}", stats.total_characters);
    println!("   • Processing time: {:.2}s", processing_time.as_secs_f64());
    println!("   • Total time: {:.2}s", total_time.as_secs_f64());
    println!("   • Video file: {} ({:.2} MB)", video_path, stats.video_stats.file_size_bytes as f64 / 1024.0 / 1024.0);
    
    // Test extraction of different data types
    println!("\n🔍 Testing data extraction:");
    test_extraction(video_path, &config).await?;
    
    // Cleanup
    cleanup_sample_files().await?;
    
    Ok(())
}

async fn create_sample_files() -> anyhow::Result<()> {
    println!("📝 Creating sample data files...");
    
    // Create sample CSV
    let csv_content = r#"id,name,age,city,salary,department
1,Alice Johnson,28,New York,75000,Engineering
2,Bob Smith,34,San Francisco,85000,Engineering
3,Carol Davis,29,Chicago,70000,Marketing
4,David Wilson,31,Boston,80000,Engineering
5,Eve Brown,26,Seattle,72000,Design
6,Frank Miller,35,Austin,78000,Marketing
7,Grace Lee,30,Denver,76000,Engineering
8,Henry Taylor,33,Portland,82000,Design
9,Ivy Chen,27,Miami,74000,Marketing
10,Jack Anderson,32,Phoenix,79000,Engineering"#;
    
    std::fs::write("sample_data.csv", csv_content)?;
    
    // Create sample Python code
    let python_content = r#"#!/usr/bin/env python3
"""
Sample Python module for data processing demonstration.
This module contains various functions for data analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class DataProcessor:
    """A class for processing and analyzing data."""
    
    def __init__(self, data_source: str):
        """Initialize the data processor.
        
        Args:
            data_source: Path to the data source file
        """
        self.data_source = data_source
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from the source file."""
        try:
            if self.data_source.endswith('.csv'):
                self.data = pd.read_csv(self.data_source)
            elif self.data_source.endswith('.json'):
                self.data = pd.read_json(self.data_source)
            else:
                raise ValueError(f"Unsupported file format: {self.data_source}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_data(self) -> Dict[str, float]:
        """Perform basic statistical analysis."""
        if self.data is None:
            return {}
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        analysis = {}
        
        for col in numeric_columns:
            analysis[f"{col}_mean"] = self.data[col].mean()
            analysis[f"{col}_std"] = self.data[col].std()
            analysis[f"{col}_min"] = self.data[col].min()
            analysis[f"{col}_max"] = self.data[col].max()
        
        return analysis

def process_file(filename: str) -> Optional[Dict]:
    """Process a data file and return analysis results."""
    processor = DataProcessor(filename)
    data = processor.load_data()
    
    if data is not None:
        return processor.analyze_data()
    return None

if __name__ == "__main__":
    # Example usage
    results = process_file("sample_data.csv")
    if results:
        print("Analysis results:", results)
"#;
    
    std::fs::write("sample_code.py", python_content)?;
    
    // Create sample Rust code
    let rust_content = r#"//! Sample Rust module for data processing demonstration
//! 
//! This module demonstrates various Rust programming patterns
//! including error handling, async programming, and data structures.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result};
use serde::{Deserialize, Serialize};

/// Configuration for data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum number of records to process
    pub max_records: usize,
    /// Enable verbose logging
    pub verbose: bool,
    /// Output format
    pub output_format: OutputFormat,
}

/// Supported output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Yaml,
}

/// A data record with various fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    pub id: u64,
    pub name: String,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

/// Main data processor struct
pub struct DataProcessor {
    config: ProcessingConfig,
    records: Vec<DataRecord>,
}

impl DataProcessor {
    /// Create a new data processor with the given configuration
    pub fn new(config: ProcessingConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
        }
    }
    
    /// Load data from a CSV file
    pub async fn load_from_csv(&mut self, filename: &str) -> Result<usize> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut count = 0;
        
        for (line_num, line) in reader.lines().enumerate() {
            if count >= self.config.max_records {
                break;
            }
            
            let line = line?;
            if line_num == 0 {
                continue; // Skip header
            }
            
            if let Some(record) = self.parse_csv_line(&line) {
                self.records.push(record);
                count += 1;
            }
        }
        
        if self.config.verbose {
            println!("Loaded {} records from {}", count, filename);
        }
        
        Ok(count)
    }
    
    /// Parse a single CSV line into a DataRecord
    fn parse_csv_line(&self, line: &str) -> Option<DataRecord> {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 3 {
            return None;
        }
        
        let id = fields[0].parse().ok()?;
        let name = fields[1].to_string();
        let value = fields[2].parse().ok()?;
        
        let mut metadata = HashMap::new();
        for (i, field) in fields.iter().enumerate().skip(3) {
            metadata.insert(format!("field_{}", i), field.to_string());
        }
        
        Some(DataRecord {
            id,
            name,
            value,
            metadata,
        })
    }
    
    /// Calculate statistics for the loaded data
    pub fn calculate_statistics(&self) -> DataStatistics {
        if self.records.is_empty() {
            return DataStatistics::default();
        }
        
        let values: Vec<f64> = self.records.iter().map(|r| r.value).collect();
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;
        
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        DataStatistics {
            count: self.records.len(),
            mean,
            variance,
            min: values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max: values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        }
    }
}

/// Statistical summary of data
#[derive(Debug, Default)]
pub struct DataStatistics {
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_record_creation() {
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());
        
        let record = DataRecord {
            id: 1,
            name: "test".to_string(),
            value: 42.0,
            metadata,
        };
        
        assert_eq!(record.id, 1);
        assert_eq!(record.value, 42.0);
    }
}
"#;
    
    std::fs::write("sample_code.rs", rust_content)?;
    
    // Create sample log file
    let log_content = r#"2024-01-15 10:30:15 INFO  [main] Application started successfully
2024-01-15 10:30:16 DEBUG [db] Database connection established
2024-01-15 10:30:17 INFO  [api] REST API server listening on port 8080
2024-01-15 10:30:20 INFO  [auth] User alice@example.com logged in
2024-01-15 10:30:25 DEBUG [cache] Cache hit for key: user_profile_123
2024-01-15 10:30:30 WARN  [api] Rate limit approaching for IP 192.168.1.100
2024-01-15 10:30:35 INFO  [db] Query executed in 45ms: SELECT * FROM users WHERE active = true
2024-01-15 10:30:40 ERROR [payment] Payment processing failed for order #12345: Invalid card number
2024-01-15 10:30:45 INFO  [email] Notification email sent to customer@example.com
2024-01-15 10:30:50 DEBUG [cache] Cache miss for key: product_details_456
2024-01-15 10:30:55 INFO  [api] Health check endpoint accessed
2024-01-15 10:31:00 WARN  [security] Multiple failed login attempts from IP 10.0.0.50
2024-01-15 10:31:05 INFO  [backup] Daily backup completed successfully (2.3GB)
2024-01-15 10:31:10 DEBUG [scheduler] Cron job 'cleanup_temp_files' executed
2024-01-15 10:31:15 ERROR [network] Connection timeout to external service api.partner.com"#;
    
    std::fs::write("sample.log", log_content)?;
    
    // Create sample JSON
    let json_content = r#"{
  "users": [
    {
      "id": 1,
      "username": "alice_j",
      "profile": {
        "firstName": "Alice",
        "lastName": "Johnson",
        "email": "alice@example.com",
        "preferences": {
          "theme": "dark",
          "notifications": true,
          "language": "en"
        }
      },
      "activity": {
        "lastLogin": "2024-01-15T10:30:20Z",
        "loginCount": 127,
        "isActive": true
      }
    },
    {
      "id": 2,
      "username": "bob_smith",
      "profile": {
        "firstName": "Bob",
        "lastName": "Smith",
        "email": "bob@example.com",
        "preferences": {
          "theme": "light",
          "notifications": false,
          "language": "en"
        }
      },
      "activity": {
        "lastLogin": "2024-01-14T15:22:10Z",
        "loginCount": 89,
        "isActive": true
      }
    }
  ],
  "metadata": {
    "version": "1.0",
    "generatedAt": "2024-01-15T10:00:00Z",
    "totalUsers": 2
  }
}"#;
    
    std::fs::write("sample_data.json", json_content)?;
    
    println!("✅ Sample files created successfully!");
    Ok(())
}

async fn test_extraction(video_path: &str, config: &Config) -> anyhow::Result<()> {
    let qr_processor = rust_mem_vid::qr::QrProcessor::default();
    let video_decoder = rust_mem_vid::video::VideoDecoder::new(config.clone());
    
    // Test extracting a few frames
    for frame in 0..std::cmp::min(3, 10) {
        match video_decoder.extract_frame(video_path, frame).await {
            Ok(image) => {
                match qr_processor.decode_qr(&image) {
                    Ok(decoded) => {
                        // Try to parse as JSON to see the structure
                        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&decoded) {
                            if let Some(text) = json_value.get("text").and_then(|t| t.as_str()) {
                                let preview = if text.len() > 80 {
                                    format!("{}...", &text[..80])
                                } else {
                                    text.to_string()
                                };
                                println!("   Frame {}: {}", frame, preview);
                            }
                        }
                    }
                    Err(_) => println!("   Frame {}: Failed to decode QR", frame),
                }
            }
            Err(_) => break,
        }
    }
    
    Ok(())
}

async fn cleanup_sample_files() -> anyhow::Result<()> {
    let files = [
        "sample_data.csv",
        "sample_code.py", 
        "sample_code.rs",
        "sample.log",
        "sample_data.json",
        "data_demo.mp4",
        "data_demo_index.json",
        "data_demo_index.metadata",
        "data_demo_index.vector",
    ];
    
    for file in &files {
        let _ = std::fs::remove_file(file);
    }
    
    let _ = std::fs::remove_dir_all("data_demo_frames");
    
    println!("🧹 Cleaned up sample files");
    Ok(())
}
