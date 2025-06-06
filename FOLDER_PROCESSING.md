# Folder Processing Guide

This guide covers the comprehensive folder processing capabilities added to Rust MemVid, enabling recursive directory traversal with advanced filtering and configuration options.

## Overview

The folder processing system allows you to:
- **Recursively process entire directory trees**
- **Filter files by type, size, and patterns**
- **Control recursion depth and behavior**
- **Preview files before processing**
- **Track detailed processing statistics**
- **Handle edge cases like symlinks and binary files**

## Quick Start

### Basic Folder Processing

```rust
use rust_mem_vid::MemvidEncoder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut encoder = MemvidEncoder::new().await?;
    
    // Process all supported files in a directory
    let stats = encoder.add_directory("./my_project").await?;
    
    println!("Processed {} files", stats.files_processed);
    
    encoder.build_video("project.mp4", "project.json").await?;
    Ok(())
}
```

### CLI Usage

```bash
# Basic directory processing
memvid encode --output project.mp4 --index project.json --dirs ./src ./docs

# Advanced filtering
memvid encode --output codebase.mp4 --index codebase.json \
  --dirs ./src ./tests \
  --max-depth 5 \
  --include-extensions "rs,py,js" \
  --exclude-extensions "exe,dll" \
  --max-file-size 10 \
  --follow-symlinks
```

## Configuration Options

### FolderConfig Structure

```rust
use rust_mem_vid::config::FolderConfig;

let config = FolderConfig {
    // Recursion control
    max_depth: Some(10),                    // Maximum directory depth
    
    // File type filtering
    include_extensions: Some(vec![          // Only process these extensions
        "rs".to_string(),
        "py".to_string(),
        "js".to_string(),
    ]),
    exclude_extensions: vec![               // Skip these extensions
        "exe".to_string(),
        "dll".to_string(),
        "bin".to_string(),
    ],
    
    // Pattern-based exclusion
    exclude_patterns: vec![
        "*/target/*".to_string(),           // Rust build artifacts
        "*/node_modules/*".to_string(),     // Node.js dependencies
        "*/.git/*".to_string(),             // Git repository
        "*/build/*".to_string(),            // Build directories
        "*/__pycache__/*".to_string(),      // Python cache
    ],
    
    // Size filtering
    min_file_size: 1,                       // Minimum file size in bytes
    max_file_size: 100 * 1024 * 1024,      // Maximum file size (100MB)
    
    // Behavior options
    follow_symlinks: false,                 // Follow symbolic links
    include_hidden: false,                  // Include hidden files
    skip_binary: true,                      // Skip binary files
};
```

### Default Configuration

The default configuration provides sensible defaults for most use cases:

- **Max Depth**: 10 levels
- **Excluded Extensions**: Common binary and build artifacts
- **Excluded Patterns**: Build directories, version control, caches
- **File Size Limits**: 1 byte minimum, 100MB maximum
- **Binary Detection**: Enabled
- **Hidden Files**: Excluded
- **Symlinks**: Not followed

## Advanced Usage

### Custom Configuration

```rust
use rust_mem_vid::{MemvidEncoder, Config};
use rust_mem_vid::config::FolderConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create custom folder configuration
    let folder_config = FolderConfig {
        max_depth: Some(3),
        include_extensions: Some(vec!["rs".to_string(), "toml".to_string()]),
        exclude_extensions: vec![],
        exclude_patterns: vec!["*/target/*".to_string()],
        min_file_size: 10,
        max_file_size: 1024 * 1024, // 1MB
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    // Apply to encoder configuration
    let mut config = Config::default();
    config.folder = folder_config;
    
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    // Process with custom configuration
    let stats = encoder.add_directory("./rust_project").await?;
    
    println!("Found {} files, processed {}", stats.files_found, stats.files_processed);
    
    Ok(())
}
```

### Preview Mode

Preview files before processing to understand what will be included:

```rust
let encoder = MemvidEncoder::new().await?;

// Preview files that would be processed
let files = encoder.preview_directory("./my_project")?;

for file in &files {
    println!("{} ({} bytes) - {:?}", 
             file.path.display(), 
             file.size, 
             file.file_type);
}

println!("Would process {} files", files.len());
```

### Multiple Directories

Process multiple directories with different configurations:

```rust
let mut encoder = MemvidEncoder::new().await?;

// Process source code with strict filtering
let src_config = FolderConfig {
    include_extensions: Some(vec!["rs".to_string()]),
    max_depth: Some(5),
    ..Default::default()
};

// Process documentation with different settings
let docs_config = FolderConfig {
    include_extensions: Some(vec!["md".to_string(), "txt".to_string()]),
    max_depth: Some(2),
    ..Default::default()
};

// Process each directory with its own configuration
encoder.add_directory_with_config("./src", Some(src_config)).await?;
encoder.add_directory_with_config("./docs", Some(docs_config)).await?;

// Also process data files with default settings
encoder.add_directory("./data").await?;
```

### Statistics and Monitoring

Track detailed statistics about folder processing:

```rust
let stats = encoder.add_directory("./large_project").await?;

println!("📊 Processing Statistics:");
println!("   Directories scanned: {}", stats.directories_scanned);
println!("   Files found: {}", stats.files_found);
println!("   Files processed: {}", stats.files_processed);
println!("   Files failed: {}", stats.files_failed);
println!("   Files skipped: {}", stats.files_skipped);
println!("   Data processed: {:.2} MB", stats.bytes_processed as f64 / 1024.0 / 1024.0);
println!("   Processing time: {}ms", stats.processing_time_ms);
```

## File Type Support

The folder processor automatically detects and processes these file types:

### Structured Data
- **CSV**: Comma-separated values
- **Parquet**: Columnar data format
- **JSON/JSONL**: JavaScript Object Notation

### Programming Languages
- **Rust**: `.rs` files with function/impl detection
- **Python**: `.py`, `.pyw` files with class/function analysis
- **JavaScript**: `.js`, `.jsx` files
- **TypeScript**: `.ts`, `.tsx` files
- **HTML**: `.html`, `.htm` files
- **CSS**: `.css`, `.scss`, `.sass` files
- **SQL**: `.sql` files

### Configuration & Documentation
- **YAML**: `.yaml`, `.yml` files
- **TOML**: `.toml` files
- **XML**: `.xml` files
- **Markdown**: `.md`, `.markdown` files

### Logs & Text
- **Log files**: `.log`, `.logs` files
- **Plain text**: `.txt` files

### Documents
- **PDF**: Portable Document Format
- **EPUB**: Electronic publication format

## Best Practices

### Performance Optimization

1. **Set appropriate depth limits** to avoid deep recursion
2. **Use file size limits** to skip very large files
3. **Exclude build directories** and caches
4. **Use specific file extensions** when possible

```rust
let optimized_config = FolderConfig {
    max_depth: Some(5),                     // Reasonable depth limit
    max_file_size: 10 * 1024 * 1024,       // 10MB limit
    exclude_patterns: vec![
        "*/target/*".to_string(),           // Rust builds
        "*/node_modules/*".to_string(),     // Node deps
        "*/.git/*".to_string(),             // Git data
        "*/build/*".to_string(),            // Build artifacts
    ],
    include_extensions: Some(vec![          // Only what you need
        "rs".to_string(),
        "py".to_string(),
        "md".to_string(),
    ]),
    ..Default::default()
};
```

### Security Considerations

1. **Don't follow symlinks** unless necessary
2. **Exclude sensitive directories** like `.git`, `.ssh`
3. **Set file size limits** to prevent memory issues
4. **Use binary detection** to avoid processing executables

### Error Handling

```rust
match encoder.add_directory("./project").await {
    Ok(stats) => {
        if stats.files_failed > 0 {
            println!("⚠️  {} files failed to process", stats.files_failed);
        }
        println!("✅ Successfully processed {} files", stats.files_processed);
    }
    Err(e) => {
        eprintln!("❌ Failed to process directory: {}", e);
    }
}
```

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure read permissions on directories
2. **Too Many Files**: Use depth limits and file filtering
3. **Large Files**: Set appropriate `max_file_size` limits
4. **Binary Files**: Enable `skip_binary` to avoid processing executables

### Debug Information

Enable debug logging to see detailed processing information:

```rust
use tracing_subscriber;

// Enable debug logging
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();

// Now folder processing will show detailed logs
let stats = encoder.add_directory("./project").await?;
```

## Examples

See the `examples/folder_demo.rs` file for comprehensive examples of all folder processing features, including:

- Basic folder processing
- Custom configuration
- Preview mode
- Selective file type processing
- Multiple directory handling
- Statistics reporting

Run the demo with:

```bash
cargo run --example folder_demo
```

This will create a temporary directory structure and demonstrate all folder processing capabilities with real examples and output.
