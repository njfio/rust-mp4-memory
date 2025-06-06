# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2024-01-15

### 🎉 Major New Features

#### Recursive Folder Processing
- **New `FolderProcessor` module** for intelligent directory traversal
- **Configurable recursion depth** with safety limits
- **Advanced file filtering** by extension, size, and patterns
- **Smart binary detection** to skip non-text files automatically
- **Gitignore integration** respects .gitignore and .ignore files
- **Symbolic link handling** with configurable follow behavior
- **Hidden file support** with optional inclusion
- **Progress tracking** with detailed statistics and reporting

#### Enhanced CLI Options
- **`--max-depth`**: Control recursion depth for directory processing
- **`--include-extensions`**: Process only specific file types
- **`--exclude-extensions`**: Skip unwanted file types
- **`--follow-symlinks`**: Follow symbolic links during traversal
- **`--include-hidden`**: Include hidden files and directories
- **`--max-file-size`**: Set maximum file size limits

## [0.2.0] - 2024-01-15

### 🎉 Major New Features

#### Data Processing Engine
- **New `DataProcessor` module** for handling structured data and code files
- **CSV Support**: Process CSV files with intelligent chunking by rows
- **Parquet Support**: Handle Parquet files with schema preservation
- **JSON Support**: Process JSON and JSON Lines files with structure-aware chunking
- **Code Analysis**: Intelligent parsing and chunking for multiple programming languages

#### Supported File Types
- **Structured Data**: CSV, Parquet, JSON, JSONL
- **Programming Languages**: Rust, Python, JavaScript, TypeScript, HTML, CSS, SQL
- **Configuration Files**: YAML, TOML, XML
- **Log Files**: Application logs with timestamp-aware chunking
- **Documentation**: Markdown files

#### Advanced Chunking Strategies
- **By Rows**: For tabular data (CSV, Parquet)
- **By Logical Units**: Functions, classes, modules for code files
- **By Size**: Configurable byte-based chunking
- **By Lines**: Line-based chunking for logs and text
- **By Semantic Similarity**: Content-aware chunking
- **By Columns**: Column-group chunking for wide datasets

#### Code Intelligence
- **Function Detection**: Automatically identify and extract function definitions
- **Class Detection**: Parse class structures and methods
- **Import Analysis**: Track dependencies and imports
- **Complexity Metrics**: Calculate cyclomatic complexity, LOC, nesting depth
- **Comment Ratio**: Analyze code documentation coverage
- **Language-Specific Parsing**: Tailored parsing for each supported language

#### Enhanced Metadata
- **File Type Information**: Automatic detection and classification
- **Code Metadata**: Functions, classes, imports, complexity metrics
- **Data Statistics**: Column types, null counts, unique values, numeric summaries
- **Format Preservation**: Schema information for reconstruction
- **Line Range Tracking**: Precise source location mapping

### 🔧 API Enhancements

#### New Encoder Methods
```rust
// Data file processing
encoder.add_data_file("data.csv").await?;
encoder.add_csv_file("sales.csv").await?;
encoder.add_parquet_file("analytics.parquet").await?;
encoder.add_code_file("main.rs").await?;
encoder.add_log_file("app.log").await?;

// Auto-detection
encoder.add_file("unknown_type.json").await?;

// Folder processing
let stats = encoder.add_directory("./src").await?;
let all_stats = encoder.add_directories(&["./src", "./tests"]).await?;

// Preview before processing
let files = encoder.preview_directory("./data")?;

// Custom folder configuration
encoder.add_directory_with_config("./code", Some(custom_folder_config)).await?;
```

#### Enhanced CLI Support
```bash
# Auto-detect and process any supported file type
memvid encode --output memory.mp4 --index memory.json --files data.csv code.rs logs.txt

# Process entire directories with mixed file types
memvid encode --output library.mp4 --index library.json --dirs ./data ./code
```

### 📊 Data Processing Features

#### CSV Processing
- Automatic header detection
- Configurable row-based chunking
- Data type inference
- Statistical analysis (mean, median, std dev)
- Null value tracking
- Unique value counting

#### Parquet Processing
- Schema preservation
- Efficient columnar processing
- Type-safe data handling
- Compression-aware chunking
- Metadata extraction

#### Code Processing
- **Rust**: Function/impl block detection, complexity analysis
- **Python**: Function/class detection, import tracking
- **JavaScript/TypeScript**: Module and function parsing
- **Generic**: Fallback processing for other languages

#### Log Processing
- Timestamp-aware chunking
- Log level detection
- Structured log parsing
- Configurable entry grouping

### 🏗️ Architecture Improvements

#### New Components
- **DataProcessor**: Core data processing engine
- **DataFileType**: File type detection and classification
- **ChunkingStrategy**: Configurable chunking algorithms
- **CodeMetadata**: Code-specific metadata structures
- **DataStats**: Statistical analysis for tabular data

#### Enhanced Integration
- Seamless integration with existing video encoding pipeline
- Backward compatibility with text-only processing
- Unified API for all file types
- Consistent metadata structure

### 📈 Performance Optimizations

#### Efficient Processing
- Streaming processing for large files
- Configurable chunk sizes based on content type
- Memory-efficient data structures
- Parallel processing where applicable

#### Smart Chunking
- Content-aware chunk boundaries
- Optimal chunk sizes for different data types
- Preservation of logical structure
- Minimal information loss

### 🧪 Testing & Quality

#### Comprehensive Test Suite
- Unit tests for all data processors
- Integration tests with encoder
- File type detection tests
- Code analysis validation
- Performance benchmarks

#### Example Applications
- **Data Demo**: Comprehensive demonstration of all features
- **Mixed File Processing**: Real-world usage examples
- **Code Analysis**: Source code indexing examples

### 📚 Documentation Updates

#### Enhanced README
- Updated feature list with data processing capabilities
- New usage examples for different file types
- Architecture documentation updates
- Performance optimization tips

#### API Documentation
- Complete documentation for new modules
- Usage examples for each file type
- Configuration options
- Best practices guide

### 🔄 Migration Guide

#### For Existing Users
- All existing APIs remain unchanged
- New functionality is additive
- No breaking changes to core features
- Optional migration to new data processing features

#### New Dependencies
- **polars**: High-performance data processing
- **arrow**: Columnar data format support
- **parquet**: Parquet file format support
- **tree-sitter**: Code parsing and analysis
- **syntect**: Syntax highlighting and analysis

### 🎯 Use Cases Enabled

#### Data Analytics
- Index and search through large CSV datasets
- Process data warehouse exports
- Analyze time-series data in Parquet format
- Create searchable data dictionaries

#### Code Documentation
- Index entire codebases for semantic search
- Create searchable API documentation
- Analyze code complexity across projects
- Track code evolution over time

#### Log Analysis
- Process application logs for troubleshooting
- Create searchable log archives
- Analyze system behavior patterns
- Monitor application performance

#### Business Intelligence
- Index structured business data
- Create searchable data catalogs
- Process ETL pipeline outputs
- Analyze customer data patterns

### 🚀 Future Roadmap

#### Planned Features
- **Database Connectors**: Direct integration with SQL databases
- **Streaming Data**: Real-time data processing capabilities
- **Advanced Analytics**: Machine learning integration
- **Custom Parsers**: Plugin system for custom file types

#### Performance Improvements
- **GPU Acceleration**: CUDA support for large-scale processing
- **Distributed Processing**: Multi-node processing capabilities
- **Caching Optimizations**: Intelligent caching strategies
- **Compression Improvements**: Better compression algorithms

---

This release represents a major expansion of rust_mem_vid's capabilities, transforming it from a document processing tool into a comprehensive data indexing and search platform. The new data processing engine opens up numerous possibilities for structured data analysis while maintaining the simplicity and performance that users expect.
