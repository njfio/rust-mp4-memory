use rust_mem_vid::{MemvidEncoder, Config};
use rust_mem_vid::config::FolderConfig;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Rust MemVid Comprehensive Demo");
    println!("==================================");
    println!("This demo showcases all the enhanced capabilities:");
    println!("• Data processing (CSV, Parquet, JSON, code files)");
    println!("• Recursive folder processing with filtering");
    println!("• Mixed content types in a single video");
    println!("• Advanced configuration options");
    println!();
    
    // Create a comprehensive test environment
    let temp_dir = create_comprehensive_test_environment().await?;
    let demo_path = temp_dir.path();
    
    println!("📁 Created comprehensive test environment at: {}", demo_path.display());
    
    // Demo 1: Process everything with default settings
    println!("\n🔄 Demo 1: Processing everything with default settings");
    demo_default_processing(demo_path).await?;
    
    // Demo 2: Selective processing by file type
    println!("\n🎯 Demo 2: Selective processing by file type");
    demo_selective_processing(demo_path).await?;
    
    // Demo 3: Advanced folder configuration
    println!("\n⚙️  Demo 3: Advanced folder configuration");
    demo_advanced_configuration(demo_path).await?;
    
    // Demo 4: Mixed content processing
    println!("\n🔀 Demo 4: Mixed content processing");
    demo_mixed_content(demo_path).await?;
    
    // Demo 5: Performance comparison
    println!("\n⚡ Demo 5: Performance comparison");
    demo_performance_comparison(demo_path).await?;
    
    println!("\n✅ All comprehensive demos completed successfully!");
    println!("🎉 Rust MemVid now supports:");
    println!("   • 15+ file formats with intelligent processing");
    println!("   • Recursive folder traversal with advanced filtering");
    println!("   • Code analysis with complexity metrics");
    println!("   • Structured data processing with statistics");
    println!("   • Configurable chunking strategies");
    println!("   • Preview and monitoring capabilities");
    
    Ok(())
}

async fn create_comprehensive_test_environment() -> anyhow::Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create complex directory structure
    let dirs = [
        "project/src/core",
        "project/src/utils", 
        "project/tests",
        "project/data/raw",
        "project/data/processed",
        "project/docs",
        "project/config",
        "project/logs",
        "project/scripts",
        "project/target/debug", // Should be excluded
        "project/.git/objects", // Should be excluded
        "external_lib/src",
        "external_lib/examples",
    ];
    
    for dir in &dirs {
        fs::create_dir_all(base_path.join(dir))?;
    }
    
    // Create diverse file types
    
    // Rust source files
    fs::write(base_path.join("project/src/main.rs"), r#"
//! Main application entry point
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub port: u16,
    pub debug: bool,
}

impl Config {
    pub fn new() -> Self {
        Self {
            database_url: "postgresql://localhost/mydb".to_string(),
            port: 8080,
            debug: false,
        }
    }
    
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        // Complex configuration loading logic
        let mut config = Self::new();
        
        if let Ok(url) = std::env::var("DATABASE_URL") {
            config.database_url = url;
        }
        
        if let Ok(port_str) = std::env::var("PORT") {
            config.port = port_str.parse()?;
        }
        
        config.debug = std::env::var("DEBUG").is_ok();
        
        Ok(config)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    
    println!("Starting server on port {}", config.port);
    
    // Initialize database connection
    let db = connect_database(&config.database_url).await?;
    
    // Start HTTP server
    start_server(config.port, db).await?;
    
    Ok(())
}

async fn connect_database(url: &str) -> Result<Database, DatabaseError> {
    // Database connection logic
    Database::connect(url).await
}

async fn start_server(port: u16, db: Database) -> Result<(), ServerError> {
    // Server startup logic
    Server::new(port, db).run().await
}
"#)?;
    
    fs::write(base_path.join("project/src/core/database.rs"), r#"
//! Database abstraction layer
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait DatabaseConnection {
    async fn execute(&self, query: &str) -> Result<Vec<Value>, DatabaseError>;
    async fn transaction<F, R>(&self, f: F) -> Result<R, DatabaseError>
    where
        F: FnOnce(&mut Transaction) -> Result<R, DatabaseError> + Send,
        R: Send;
}

pub struct PostgresConnection {
    pool: deadpool_postgres::Pool,
}

impl PostgresConnection {
    pub async fn new(database_url: &str) -> Result<Self, DatabaseError> {
        let config = tokio_postgres::Config::from_str(database_url)?;
        let manager = deadpool_postgres::Manager::new(config, tokio_postgres::NoTls);
        let pool = deadpool_postgres::Pool::builder(manager).build()?;
        
        Ok(Self { pool })
    }
}

#[async_trait]
impl DatabaseConnection for PostgresConnection {
    async fn execute(&self, query: &str) -> Result<Vec<Value>, DatabaseError> {
        let client = self.pool.get().await?;
        let rows = client.query(query, &[]).await?;
        
        let mut results = Vec::new();
        for row in rows {
            let mut obj = serde_json::Map::new();
            for (i, column) in row.columns().iter().enumerate() {
                let value: Value = match column.type_() {
                    &tokio_postgres::types::Type::INT4 => {
                        row.get::<_, i32>(i).into()
                    }
                    &tokio_postgres::types::Type::TEXT => {
                        row.get::<_, String>(i).into()
                    }
                    _ => Value::Null,
                };
                obj.insert(column.name().to_string(), value);
            }
            results.push(Value::Object(obj));
        }
        
        Ok(results)
    }
    
    async fn transaction<F, R>(&self, f: F) -> Result<R, DatabaseError>
    where
        F: FnOnce(&mut Transaction) -> Result<R, DatabaseError> + Send,
        R: Send,
    {
        let mut client = self.pool.get().await?;
        let transaction = client.transaction().await?;
        let mut tx = Transaction::new(transaction);
        
        match f(&mut tx) {
            Ok(result) => {
                tx.commit().await?;
                Ok(result)
            }
            Err(e) => {
                tx.rollback().await?;
                Err(e)
            }
        }
    }
}
"#)?;
    
    // Python data analysis script
    fs::write(base_path.join("project/scripts/analyze_data.py"), r#"#!/usr/bin/env python3
"""
Advanced data analysis script for processing various data formats.
Supports CSV, JSON, and Parquet files with statistical analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class DataAnalyzer:
    def __init__(self):
        self.results = {}

    def load_data(self, file_path):
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def analyze_dataframe(self, df, name="dataset"):
        analysis = {
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_summary'] = df[numeric_cols].describe().to_dict()

        return analysis

def main():
    analyzer = DataAnalyzer()
    print("Data analysis script ready")

if __name__ == "__main__":
    main()
"#)?;
    
    // Data files
    fs::write(base_path.join("project/data/raw/sales.csv"), r#"date,product,category,quantity,price,customer_id,region
2024-01-01,Widget A,Electronics,10,29.99,1001,North
2024-01-01,Widget B,Electronics,5,49.99,1002,South
2024-01-01,Gadget X,Home,3,19.99,1003,East
2024-01-02,Widget A,Electronics,8,29.99,1004,West
2024-01-02,Tool Y,Industrial,2,99.99,1005,North
2024-01-03,Widget B,Electronics,12,49.99,1006,South
2024-01-03,Gadget X,Home,7,19.99,1007,East
2024-01-04,Widget A,Electronics,15,29.99,1008,West
2024-01-04,Tool Y,Industrial,4,99.99,1009,North
2024-01-05,Gadget X,Home,9,19.99,1010,South
"#)?;
    
    fs::write(base_path.join("project/data/processed/analytics.json"), r#"{
  "summary": {
    "total_sales": 1547.89,
    "total_orders": 10,
    "average_order_value": 154.79,
    "top_products": [
      {"name": "Widget A", "sales": 449.85, "orders": 3},
      {"name": "Widget B", "sales": 849.83, "orders": 2},
      {"name": "Gadget X", "sales": 379.81, "orders": 3}
    ]
  },
  "regional_breakdown": {
    "North": {"sales": 549.85, "orders": 3},
    "South": {"sales": 469.81, "orders": 3},
    "East": {"sales": 239.82, "orders": 2},
    "West": {"sales": 288.41, "orders": 2}
  },
  "trends": {
    "daily_sales": [
      {"date": "2024-01-01", "sales": 399.95},
      {"date": "2024-01-02", "sales": 339.97},
      {"date": "2024-01-03", "sales": 739.87},
      {"date": "2024-01-04", "sales": 649.96},
      {"date": "2024-01-05", "sales": 199.99}
    ]
  }
}"#)?;
    
    // Configuration files
    fs::write(base_path.join("project/config/app.toml"), r#"
[app]
name = "Comprehensive Demo App"
version = "1.0.0"
debug = true

[database]
host = "localhost"
port = 5432
name = "demo_db"
pool_size = 10

[server]
host = "0.0.0.0"
port = 8080
workers = 4

[logging]
level = "info"
file = "app.log"
max_size = "100MB"
rotate = true

[features]
enable_metrics = true
enable_tracing = true
enable_cache = true
cache_ttl = 3600
"#)?;
    
    // Log files
    fs::write(base_path.join("project/logs/application.log"), r#"
2024-01-15 09:00:00.123 INFO  [main] Application starting up
2024-01-15 09:00:00.456 DEBUG [config] Loading configuration from app.toml
2024-01-15 09:00:00.789 INFO  [database] Connecting to database at localhost:5432
2024-01-15 09:00:01.012 INFO  [database] Connection pool initialized with 10 connections
2024-01-15 09:00:01.345 INFO  [server] HTTP server starting on 0.0.0.0:8080
2024-01-15 09:00:01.678 INFO  [server] Server ready to accept connections
2024-01-15 09:00:05.123 INFO  [api] GET /health - 200 OK (2ms)
2024-01-15 09:00:10.456 INFO  [api] POST /api/users - 201 Created (45ms)
2024-01-15 09:00:15.789 WARN  [auth] Failed login attempt for user: invalid_user
2024-01-15 09:00:20.012 ERROR [database] Query timeout: SELECT * FROM large_table WHERE complex_condition = true
2024-01-15 09:00:25.345 INFO  [cache] Cache hit rate: 85.3% (1234 hits, 213 misses)
2024-01-15 09:00:30.678 DEBUG [metrics] Memory usage: 245MB, CPU: 12.5%
"#)?;
    
    // Documentation
    fs::write(base_path.join("project/docs/API.md"), r#"
# API Documentation

## Overview

This API provides comprehensive data processing and analysis capabilities.

## Endpoints

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T09:00:00Z",
  "version": "1.0.0"
}
```

### POST /api/data/analyze
Analyze uploaded data files.

**Request:**
```json
{
  "file_url": "https://example.com/data.csv",
  "format": "csv",
  "options": {
    "include_correlations": true,
    "generate_summary": true
  }
}
```

**Response:**
```json
{
  "analysis_id": "abc123",
  "status": "completed",
  "results": {
    "summary": {...},
    "correlations": {...}
  }
}
```

## Authentication

All API endpoints require authentication via Bearer token:

```
Authorization: Bearer <your-token>
```

## Rate Limiting

- 1000 requests per hour per API key
- 100 requests per minute for analysis endpoints
"#)?;
    
    // Files that should be excluded
    fs::write(base_path.join("project/target/debug/app"), "binary executable")?;
    fs::write(base_path.join("project/.git/objects/abc123"), "git object")?;
    
    Ok(temp_dir)
}

async fn demo_default_processing(demo_path: &Path) -> anyhow::Result<()> {
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    let start = std::time::Instant::now();
    let stats = encoder.add_directory(&demo_path.to_string_lossy()).await?;
    let duration = start.elapsed();
    
    println!("📊 Default Processing Results:");
    println!("   • Files found: {}", stats.files_found);
    println!("   • Files processed: {}", stats.files_processed);
    println!("   • Files failed: {}", stats.files_failed);
    println!("   • Data processed: {:.2} KB", stats.bytes_processed as f64 / 1024.0);
    println!("   • Processing time: {:.2}s", duration.as_secs_f64());
    println!("   • Total chunks: {}", encoder.len());
    
    Ok(())
}

async fn demo_selective_processing(demo_path: &Path) -> anyhow::Result<()> {
    let folder_config = FolderConfig {
        max_depth: Some(3),
        include_extensions: Some(vec!["rs".to_string(), "py".to_string()]),
        exclude_extensions: vec![],
        exclude_patterns: vec!["*/target/*".to_string(), "*/.git/*".to_string()],
        min_file_size: 100,
        max_file_size: 50 * 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let mut config = Config::default();
    config.folder = folder_config;
    
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    let start = std::time::Instant::now();
    let stats = encoder.add_directory(&demo_path.to_string_lossy()).await?;
    let duration = start.elapsed();
    
    println!("📊 Selective Processing Results (Rust + Python only):");
    println!("   • Files found: {}", stats.files_found);
    println!("   • Files processed: {}", stats.files_processed);
    println!("   • Processing time: {:.2}s", duration.as_secs_f64());
    println!("   • Total chunks: {}", encoder.len());
    
    Ok(())
}

async fn demo_advanced_configuration(demo_path: &Path) -> anyhow::Result<()> {
    let folder_config = FolderConfig {
        max_depth: Some(2),
        include_extensions: Some(vec![
            "toml".to_string(), 
            "json".to_string(), 
            "md".to_string()
        ]),
        exclude_extensions: vec![],
        exclude_patterns: vec![],
        min_file_size: 50,
        max_file_size: 100 * 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let mut config = Config::default();
    config.folder = folder_config;
    
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    // Preview first
    let files = encoder.preview_directory(&demo_path.to_string_lossy())?;
    println!("📋 Preview - would process {} files:", files.len());
    for file in files.iter().take(5) {
        println!("   • {} ({} bytes)", 
                 file.path.file_name().unwrap().to_string_lossy(),
                 file.size);
    }
    
    let start = std::time::Instant::now();
    let stats = encoder.add_directory(&demo_path.to_string_lossy()).await?;
    let duration = start.elapsed();
    
    println!("📊 Advanced Configuration Results (Config + Docs only):");
    println!("   • Files processed: {}", stats.files_processed);
    println!("   • Processing time: {:.2}s", duration.as_secs_f64());
    println!("   • Total chunks: {}", encoder.len());
    
    Ok(())
}

async fn demo_mixed_content(demo_path: &Path) -> anyhow::Result<()> {
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    // Process different subdirectories with different approaches
    let project_path = demo_path.join("project");
    
    // Add specific files
    encoder.add_file(&project_path.join("data/raw/sales.csv").to_string_lossy()).await?;
    encoder.add_file(&project_path.join("data/processed/analytics.json").to_string_lossy()).await?;
    encoder.add_file(&project_path.join("config/app.toml").to_string_lossy()).await?;
    
    // Add directories
    encoder.add_directory(&project_path.join("src").to_string_lossy()).await?;
    encoder.add_directory(&project_path.join("docs").to_string_lossy()).await?;
    
    println!("📊 Mixed Content Processing Results:");
    println!("   • Total chunks from mixed sources: {}", encoder.len());
    println!("   • Successfully combined structured data, code, and documentation");
    
    Ok(())
}

async fn demo_performance_comparison(demo_path: &Path) -> anyhow::Result<()> {
    println!("⚡ Comparing processing strategies:");
    
    // Strategy 1: Process everything
    let start = std::time::Instant::now();
    let config1 = Config::default();
    let mut encoder1 = MemvidEncoder::new_with_config(config1).await?;
    let stats1 = encoder1.add_directory(&demo_path.to_string_lossy()).await?;
    let duration1 = start.elapsed();
    
    // Strategy 2: Selective processing
    let start = std::time::Instant::now();
    let mut config2 = Config::default();
    config2.folder.include_extensions = Some(vec!["rs".to_string(), "py".to_string()]);
    let mut encoder2 = MemvidEncoder::new_with_config(config2).await?;
    let stats2 = encoder2.add_directory(&demo_path.to_string_lossy()).await?;
    let duration2 = start.elapsed();
    
    println!("📊 Performance Comparison:");
    println!("   Strategy 1 (All files):");
    println!("     • Files: {} processed in {:.2}s", stats1.files_processed, duration1.as_secs_f64());
    println!("     • Chunks: {}", encoder1.len());
    println!("   Strategy 2 (Code only):");
    println!("     • Files: {} processed in {:.2}s", stats2.files_processed, duration2.as_secs_f64());
    println!("     • Chunks: {}", encoder2.len());
    println!("   Speedup: {:.1}x faster with selective processing", 
             duration1.as_secs_f64() / duration2.as_secs_f64());
    
    Ok(())
}
