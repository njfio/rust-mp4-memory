use rust_mem_vid::{MemvidEncoder, Config, FolderProcessor};
use rust_mem_vid::config::FolderConfig;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🗂️  Rust MemVid Folder Processing Demo");
    println!("=====================================");
    
    // Create a temporary directory structure for demonstration
    let temp_dir = create_demo_directory_structure().await?;
    let demo_path = temp_dir.path();
    
    println!("📁 Created demo directory structure at: {}", demo_path.display());
    
    // Demo 1: Basic folder processing with default settings
    println!("\n🔍 Demo 1: Basic folder processing");
    demo_basic_folder_processing(demo_path).await?;
    
    // Demo 2: Custom folder configuration
    println!("\n⚙️  Demo 2: Custom folder configuration");
    demo_custom_folder_config(demo_path).await?;
    
    // Demo 3: Preview files before processing
    println!("\n👀 Demo 3: Preview files before processing");
    demo_preview_files(demo_path).await?;
    
    // Demo 4: Selective file type processing
    println!("\n🎯 Demo 4: Selective file type processing");
    demo_selective_processing(demo_path).await?;
    
    // Demo 5: Multiple directories with different configs
    println!("\n🔄 Demo 5: Multiple directories processing");
    demo_multiple_directories(demo_path).await?;
    
    println!("\n✅ All folder processing demos completed!");
    
    Ok(())
}

async fn create_demo_directory_structure() -> anyhow::Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create directory structure
    fs::create_dir_all(base_path.join("src"))?;
    fs::create_dir_all(base_path.join("data"))?;
    fs::create_dir_all(base_path.join("docs"))?;
    fs::create_dir_all(base_path.join("config"))?;
    fs::create_dir_all(base_path.join("logs"))?;
    fs::create_dir_all(base_path.join("build"))?; // Should be excluded
    fs::create_dir_all(base_path.join(".git"))?; // Should be excluded
    fs::create_dir_all(base_path.join("src/utils"))?;
    fs::create_dir_all(base_path.join("data/processed"))?;
    
    // Create Rust source files
    fs::write(base_path.join("src/main.rs"), r#"
fn main() {
    println!("Hello, world!");
    let data = load_data();
    process_data(data);
}

fn load_data() -> Vec<i32> {
    vec![1, 2, 3, 4, 5]
}

fn process_data(data: Vec<i32>) {
    for item in data {
        println!("Processing: {}", item);
    }
}
"#)?;
    
    fs::write(base_path.join("src/lib.rs"), r#"
//! Demo library for folder processing
//! 
//! This library demonstrates various Rust patterns.

pub mod utils;

/// A simple calculator
pub struct Calculator {
    value: f64,
}

impl Calculator {
    pub fn new() -> Self {
        Self { value: 0.0 }
    }
    
    pub fn add(&mut self, x: f64) -> &mut Self {
        self.value += x;
        self
    }
    
    pub fn multiply(&mut self, x: f64) -> &mut Self {
        self.value *= x;
        self
    }
    
    pub fn result(&self) -> f64 {
        self.value
    }
}
"#)?;
    
    fs::write(base_path.join("src/utils/mod.rs"), r#"
//! Utility functions

pub fn format_number(n: f64) -> String {
    format!("{:.2}", n)
}

pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}
"#)?;
    
    // Create Python files
    fs::write(base_path.join("data/analysis.py"), r#"
#!/usr/bin/env python3
"""Data analysis script"""

import pandas as pd
import numpy as np

def load_data(filename):
    """Load data from CSV file"""
    return pd.read_csv(filename)

def analyze_data(df):
    """Perform basic analysis"""
    return {
        'mean': df.mean(),
        'std': df.std(),
        'count': len(df)
    }

if __name__ == "__main__":
    data = load_data("sample.csv")
    results = analyze_data(data)
    print(results)
"#)?;
    
    // Create data files
    fs::write(base_path.join("data/sample.csv"), r#"id,name,value,category
1,Alice,100,A
2,Bob,200,B
3,Carol,150,A
4,David,300,C
5,Eve,250,B
"#)?;
    
    fs::write(base_path.join("data/config.json"), r#"{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "demo_db"
  },
  "features": {
    "enable_cache": true,
    "max_connections": 100
  }
}"#)?;
    
    // Create configuration files
    fs::write(base_path.join("config/app.yaml"), r#"
app:
  name: "Demo Application"
  version: "1.0.0"
  debug: true

server:
  host: "0.0.0.0"
  port: 8080
  
logging:
  level: "info"
  file: "app.log"
"#)?;
    
    fs::write(base_path.join("config/database.toml"), r#"
[database]
host = "localhost"
port = 5432
username = "demo_user"
password = "demo_pass"
database = "demo_db"

[connection_pool]
max_size = 10
min_size = 2
timeout = 30
"#)?;
    
    // Create log files
    fs::write(base_path.join("logs/app.log"), r#"
2024-01-15 10:00:00 INFO  [main] Application starting
2024-01-15 10:00:01 DEBUG [db] Connecting to database
2024-01-15 10:00:02 INFO  [db] Database connection established
2024-01-15 10:00:03 INFO  [server] HTTP server listening on :8080
2024-01-15 10:00:10 INFO  [api] GET /health - 200 OK
2024-01-15 10:00:15 WARN  [auth] Invalid token for user: unknown_user
2024-01-15 10:00:20 ERROR [db] Query timeout: SELECT * FROM large_table
"#)?;
    
    // Create documentation
    fs::write(base_path.join("docs/README.md"), r#"
# Demo Project

This is a demonstration project for folder processing capabilities.

## Features

- Rust source code processing
- Python script analysis
- Data file handling (CSV, JSON)
- Configuration file support (YAML, TOML)
- Log file processing

## Usage

Run the application with:

```bash
cargo run
```

## Configuration

See the `config/` directory for configuration files.
"#)?;
    
    // Create files that should be excluded
    fs::write(base_path.join("build/output.bin"), "binary content")?;
    fs::write(base_path.join(".git/config"), "git config")?;
    fs::write(base_path.join("large_file.dat"), "x".repeat(200 * 1024 * 1024))?; // 200MB file
    
    Ok(temp_dir)
}

async fn demo_basic_folder_processing(demo_path: &Path) -> anyhow::Result<()> {
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    println!("Processing folder with default settings...");
    let stats = encoder.add_directory(&demo_path.to_string_lossy()).await?;
    
    println!("📊 Results:");
    println!("   • Files found: {}", stats.files_found);
    println!("   • Files processed: {}", stats.files_processed);
    println!("   • Files failed: {}", stats.files_failed);
    println!("   • Data processed: {:.2} KB", stats.bytes_processed as f64 / 1024.0);
    println!("   • Processing time: {}ms", stats.processing_time_ms);
    println!("   • Total chunks created: {}", encoder.len());
    
    Ok(())
}

async fn demo_custom_folder_config(demo_path: &Path) -> anyhow::Result<()> {
    let mut config = Config::default();
    
    // Customize folder processing
    config.folder = FolderConfig {
        max_depth: Some(2), // Limit depth
        include_extensions: Some(vec!["rs".to_string(), "py".to_string()]), // Only Rust and Python
        exclude_extensions: vec!["dat".to_string()], // Exclude .dat files
        exclude_patterns: vec![
            "*/build/*".to_string(),
            "*/.git/*".to_string(),
        ],
        min_file_size: 10, // At least 10 bytes
        max_file_size: 50 * 1024, // Max 50KB
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    println!("Processing with custom configuration (Rust and Python only, max depth 2)...");
    let stats = encoder.add_directory(&demo_path.to_string_lossy()).await?;
    
    println!("📊 Results:");
    println!("   • Files found: {}", stats.files_found);
    println!("   • Files processed: {}", stats.files_processed);
    println!("   • Total chunks created: {}", encoder.len());
    
    Ok(())
}

async fn demo_preview_files(demo_path: &Path) -> anyhow::Result<()> {
    let config = Config::default();
    let encoder = MemvidEncoder::new_with_config(config).await?;
    
    println!("Previewing files that would be processed...");
    let files = encoder.preview_directory(&demo_path.to_string_lossy())?;
    
    println!("📋 Files to be processed:");
    for (i, file) in files.iter().enumerate().take(10) { // Show first 10
        let size_kb = file.size as f64 / 1024.0;
        println!("   {}. {} ({:.1} KB) - {:?}", 
                 i + 1, 
                 file.path.file_name().unwrap().to_string_lossy(),
                 size_kb,
                 file.file_type);
    }
    
    if files.len() > 10 {
        println!("   ... and {} more files", files.len() - 10);
    }
    
    println!("Total files that would be processed: {}", files.len());
    
    Ok(())
}

async fn demo_selective_processing(demo_path: &Path) -> anyhow::Result<()> {
    // Process only data files
    let data_config = FolderConfig {
        max_depth: Some(5),
        include_extensions: Some(vec!["csv".to_string(), "json".to_string()]),
        exclude_extensions: vec![],
        exclude_patterns: vec![],
        min_file_size: 1,
        max_file_size: 10 * 1024 * 1024, // 10MB
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let mut config = Config::default();
    config.folder = data_config;
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    println!("Processing only data files (CSV, JSON)...");
    let stats = encoder.add_directory(&demo_path.to_string_lossy()).await?;
    
    println!("📊 Data files processing results:");
    println!("   • Files processed: {}", stats.files_processed);
    println!("   • Chunks created: {}", encoder.len());
    
    Ok(())
}

async fn demo_multiple_directories(demo_path: &Path) -> anyhow::Result<()> {
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    // Process different subdirectories
    let subdirs = vec![
        demo_path.join("src").to_string_lossy().to_string(),
        demo_path.join("data").to_string_lossy().to_string(),
        demo_path.join("config").to_string_lossy().to_string(),
    ];
    
    println!("Processing multiple directories separately...");
    
    let mut total_processed = 0;
    for subdir in subdirs {
        if Path::new(&subdir).exists() {
            println!("  Processing: {}", subdir);
            match encoder.add_directory(&subdir).await {
                Ok(stats) => {
                    println!("    ✅ {} files processed", stats.files_processed);
                    total_processed += stats.files_processed;
                }
                Err(e) => {
                    println!("    ❌ Error: {}", e);
                }
            }
        }
    }
    
    println!("📊 Multiple directories results:");
    println!("   • Total files processed: {}", total_processed);
    println!("   • Total chunks created: {}", encoder.len());
    
    Ok(())
}
