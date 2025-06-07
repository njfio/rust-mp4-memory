# Rust MemVid 🦀📹

A complete Rust implementation of the [MemVid](https://github.com/Olow304/memvid) library - a revolutionary video-based AI memory system that stores text chunks as QR codes in video files with lightning-fast semantic search and **temporal analysis capabilities**.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Core Features

- **🎥 Video-as-Database**: Store millions of text chunks in MP4 files
- **🔍 Semantic Search**: Find relevant content using natural language queries with embeddings
- **💬 Built-in Chat**: Conversational interface with LLM integration (OpenAI, Anthropic)
- **📚 Document Support**: Direct import of PDF, EPUB, text files, CSV, Parquet, JSON, and code files
- **🔧 Data Processing**: Advanced chunking strategies for structured data and source code
- **💻 Code Analysis**: Intelligent parsing of Rust, Python, JavaScript, TypeScript, and more
- **🚀 Fast Retrieval**: Sub-second search across massive datasets
- **💾 Efficient Storage**: Advanced video compression with multiple codec support
- **🔧 Multiple Codecs**: H.264, H.265, AV1, VP9 support via FFmpeg
- **🌐 Offline-First**: No internet required after video generation
- **⚡ High Performance**: Parallel processing and optimized algorithms

## 🕰️ **Temporal Analysis & Memory Comparison**

**Revolutionary features that transform MemVid from simple storage into a complete knowledge evolution platform:**

- **🔍 Memory Diff Engine**: Compare any two memory videos with detailed chunk-level analysis
- **🔎 Multi-Memory Search**: Search across multiple memory videos simultaneously
- **📈 Temporal Analysis**: Track memory evolution over time with trend analysis
- **🔗 Cross-Memory Correlations**: Find relationships between different memory snapshots
- **📊 Knowledge Gap Detection**: Identify areas needing attention or updates
- **🎯 Activity Period Analysis**: Detect growth, revision, and consolidation phases

## 🧠 **AI Intelligence Features (Phase 1) - FULLY IMPLEMENTED!**

**Transform your memories into an intelligent knowledge ecosystem with advanced AI capabilities:**

### ✅ **KNOWLEDGE GRAPH GENERATION - 100% FUNCTIONAL**
- **🕸️ Real Concept Extraction**: Named Entity, Keyword, and Technical concept extractors working
- **🔗 Advanced Relationship Analysis**: Co-occurrence, Semantic Similarity, and Hierarchical analyzers implemented
- **🏘️ Community Detection**: Graph clustering algorithm finds concept communities automatically
- **📊 Complete Integration**: Builds knowledge graphs from actual memory retrievers with real data
- **🎯 Scoring Systems**: Real importance scoring and relationship strength calculation

### ✅ **CONTENT SYNTHESIS - 100% FUNCTIONAL**
- **🤖 AI-Powered Synthesis**: Full integration with OpenAI, Anthropic, and Ollama APIs
- **📝 Template Fallback**: Works offline with sophisticated template-based synthesis
- **🎨 Multiple Types**: Summary, Insights, Connections, Contradictions, Trends, Recommendations
- **📋 Structured Output**: Confidence scores, key points, and supporting evidence extraction
- **� Real Processing**: No more mocks - actual content analysis and generation

### ✅ **ANALYTICS DASHBOARD - 100% FUNCTIONAL**
- **📈 Temporal Metrics**: Real timeline analysis and growth velocity calculation from snapshots
- **🧠 Knowledge Metrics**: Concept density, relationship strength, community analysis from graphs
- **� Growth Analysis**: Trend detection with acceleration/deceleration pattern recognition
- **💎 Quality Metrics**: Information density, freshness scores, coherence analysis
- **💡 Insights & Recommendations**: Automated generation of actionable insights and suggestions

### ✅ **RELATIONSHIP ANALYSIS - 100% FUNCTIONAL**
- **🔄 Co-occurrence Analysis**: Configurable window sizes and frequency tracking
- **🎯 Semantic Similarity**: Embedding-based relationship detection and scoring
- **🌳 Hierarchical Detection**: Pattern-based parent-child relationship identification
- **⏰ Temporal Tracking**: Relationship evolution over time with confidence scoring
- **📚 Evidence Collection**: Supporting evidence compilation and validation

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust_mem_vid = "0.1.0"
```

Or install the CLI tool:

```bash
cargo install rust_mem_vid
```

### Basic Usage

```rust
use rust_mem_vid::{MemvidEncoder, MemvidRetriever, MemvidChat};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the library
    rust_mem_vid::init().await?;
    
    // Create encoder and add text chunks
    let mut encoder = MemvidEncoder::new().await?;
    encoder.add_chunks(vec![
        "Important fact 1".to_string(),
        "Important fact 2".to_string(),
        "Historical event details".to_string(),
    ]).await?;
    
    // Build video memory
    encoder.build_video("memory.mp4", "memory_index.json").await?;
    
    // Search the memory
    let retriever = MemvidRetriever::new("memory.mp4", "memory_index.json").await?;
    let results = retriever.search("historical events", 5).await?;
    
    // Chat with the memory (requires API key)
    let mut chat = MemvidChat::new("memory.mp4", "memory_index.json").await?;
    let response = chat.chat("What do you know about historical events?").await?;
    println!("{}", response);
    
    Ok(())
}
```

### CLI Usage

```bash
# Encode documents into a video (auto-detects file types)
memvid encode --output memory.mp4 --index memory.json --files document.pdf data.csv code.rs logs.txt

# Encode with directories (processes all supported files recursively)
memvid encode --output library.mp4 --index library.json --dirs ./documents ./code

# Advanced folder processing with custom options
memvid encode --output codebase.mp4 --index codebase.json \
  --dirs ./src ./tests \
  --max-depth 5 \
  --include-extensions "rs,py,js,ts" \
  --exclude-extensions "exe,dll,bin" \
  --max-file-size 10 \
  --follow-symlinks \
  --include-hidden

# Process only specific file types from a directory
memvid encode --output data.mp4 --index data.json \
  --dirs ./data \
  --include-extensions "csv,json,parquet"

# Search the video
memvid search --video memory.mp4 --index memory.json --query "machine learning" --top-k 5

# Start interactive chat
memvid chat --video memory.mp4 --index memory.json --provider openai

# Get video/index information
memvid info --video memory.mp4 --index memory.json

# Extract specific frame
memvid extract --video memory.mp4 --frame 42

# Compare two memory videos (NEW!)
memvid diff old_memory.mp4 old_memory.metadata new_memory.mp4 new_memory.metadata \
  --output diff_report.json --semantic

# Search across multiple memories (NEW!)
memvid multi-search "machine learning" memories.json \
  --top-k 10 --correlations --temporal --tags research

# Generate knowledge graph (NEW!)
memvid knowledge-graph \
  memory1.mp4,memory1.metadata \
  memory2.mp4,memory2.metadata \
  --output knowledge_graph.json \
  --semantic \
  --confidence-threshold 0.8

# Create intelligent content synthesis (NEW!)
memvid synthesize "machine learning algorithms" \
  research.mp4,research.metadata \
  notes.mp4,notes.metadata \
  --synthesis-type insights \
  --output insights.json

# Generate analytics dashboard (NEW!)
memvid dashboard \
  memory1.mp4,memory1.metadata \
  memory2.mp4,memory2.metadata \
  --output ./dashboard \
  --visualizations \
  --format html
```

## 🕰️ Temporal Analysis & Memory Comparison

### Memory Diff Analysis

Compare any two memory videos to see exactly what changed:

```bash
# Basic comparison
memvid diff old_project.mp4 old_project.metadata new_project.mp4 new_project.metadata

# With semantic analysis and detailed output
memvid diff research_v1.mp4 research_v1.metadata research_v2.mp4 research_v2.metadata \
  --output detailed_diff.json --semantic
```

**Example Output:**
```
🔍 Memory Comparison Results
============================
Old memory: research_v1.mp4
New memory: research_v2.mp4

📊 Summary:
   • Old chunks: 150
   • New chunks: 203
   • Added: 75 chunks
   • Removed: 22 chunks
   • Modified: 18 chunks
   • Unchanged: 110 chunks
   • Similarity: 72.5%
   • Growth ratio: 1.35x
```

### Multi-Memory Search

Search across multiple memory videos simultaneously:

```bash
# Create memories configuration
cat > memories.json << EOF
[
  {
    "name": "research_v1",
    "video_path": "research_v1.mp4",
    "index_path": "research_v1.metadata",
    "tags": ["research", "initial"],
    "description": "Initial research phase"
  },
  {
    "name": "research_v2",
    "video_path": "research_v2.mp4",
    "index_path": "research_v2.metadata",
    "tags": ["research", "enhanced"],
    "description": "Enhanced research with new findings"
  }
]
EOF

# Search across all memories
memvid multi-search "neural networks" memories.json --correlations --temporal

# Filter by tags
memvid multi-search "methodology" memories.json --tags enhanced
```

### Programmatic Temporal Analysis

```rust
use rust_mem_vid::{
    memory_diff::MemoryDiffEngine,
    multi_memory::MultiMemoryEngine,
    temporal_analysis::TemporalAnalysisEngine
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::default();

    // Compare two memories
    let diff_engine = MemoryDiffEngine::new(config.clone())
        .with_semantic_analysis(true);

    let diff = diff_engine.compare_memories(
        "old_memory.mp4", "old_memory.metadata",
        "new_memory.mp4", "new_memory.metadata"
    ).await?;

    println!("Added {} chunks, modified {} chunks",
             diff.summary.added_count, diff.summary.modified_count);

    // Multi-memory search
    let mut multi_engine = MultiMemoryEngine::new(config.clone());
    multi_engine.add_memory("v1", "v1.mp4", "v1.metadata",
                           vec!["version_1".to_string()], None).await?;
    multi_engine.add_memory("v2", "v2.mp4", "v2.metadata",
                           vec!["version_2".to_string()], None).await?;

    let results = multi_engine.search_all("machine learning", 10, true, true).await?;
    println!("Found {} results across {} memories",
             results.total_results, results.search_metadata.memories_searched);

    // Temporal analysis
    let temporal_engine = TemporalAnalysisEngine::new(config);
    let snapshot = temporal_engine.create_snapshot(
        "memory.mp4", "memory.metadata",
        Some("Project milestone".to_string()),
        vec!["milestone".to_string()]
    ).await?;

    Ok(())
}
```

## 🧠 AI Intelligence Features

### Knowledge Graph Generation

Automatically extract concepts and relationships from your memory content:

```bash
# Generate knowledge graph from multiple memories
memvid knowledge-graph \
  research.mp4,research.metadata \
  papers.mp4,papers.metadata \
  --output knowledge_graph.json \
  --semantic \
  --confidence-threshold 0.8

# Basic knowledge graph without semantic analysis
memvid knowledge-graph \
  memory.mp4,memory.metadata \
  --output graph.json
```

**Example Output:**
```
🕸️  Knowledge Graph Generated:
   • Concepts: 1,247
   • Relationships: 3,891
   • Communities: 23
   • Output: knowledge_graph.json

🔝 Top Concepts:
   • machine learning (importance: 0.95, type: Topic)
   • neural networks (importance: 0.89, type: Concept)
   • deep learning (importance: 0.87, type: Process)
   • artificial intelligence (importance: 0.84, type: Topic)
   • data science (importance: 0.82, type: Topic)
```

### Intelligent Content Synthesis

Generate AI-powered insights and summaries:

```bash
# Generate comprehensive summary
memvid synthesize "machine learning algorithms" \
  research.mp4,research.metadata \
  --synthesis-type summary \
  --output summary.json

# Extract key insights
memvid synthesize "deep learning trends" \
  papers.mp4,papers.metadata \
  --synthesis-type insights

# Identify knowledge gaps
memvid synthesize "quantum computing" \
  knowledge.mp4,knowledge.metadata \
  --synthesis-type gaps

# Generate actionable recommendations
memvid synthesize "project optimization" \
  project.mp4,project.metadata \
  --synthesis-type recommendations
```

**Synthesis Types Available:**
- **summary**: Comprehensive overviews of topics
- **insights**: Key patterns and discoveries
- **contradictions**: Conflicting information detection
- **gaps**: Missing information identification
- **recommendations**: Actionable suggestions

### Advanced Analytics Dashboard

Create comprehensive visual dashboards:

```bash
# Generate HTML dashboard with visualizations
memvid dashboard \
  memory1.mp4,memory1.metadata \
  memory2.mp4,memory2.metadata \
  --output ./dashboard \
  --visualizations \
  --format html

# Generate JSON analytics data
memvid dashboard \
  research.mp4,research.metadata \
  --output ./analytics \
  --format json
```

**Dashboard Features:**
- **Temporal Metrics**: Growth velocity, activity periods, evolution patterns
- **Knowledge Metrics**: Concept density, relationship strength, community analysis
- **Quality Metrics**: Content coherence, information density, freshness scores
- **Interactive Visualizations**: Timeline charts, knowledge maps, growth curves
- **Intelligent Insights**: Automatically detected patterns and trends
- **Actionable Recommendations**: Specific suggestions for improvement

### Programmatic AI Intelligence

```rust
use rust_mem_vid::{
    knowledge_graph::KnowledgeGraphBuilder,
    content_synthesis::{ContentSynthesizer, SynthesisType},
    analytics_dashboard::AnalyticsDashboard
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::default();

    // Generate knowledge graph
    let graph_builder = KnowledgeGraphBuilder::new(config.clone())
        .with_embeddings().await?;

    let memories = vec![("memory.mp4".to_string(), "memory.metadata".to_string())];
    let knowledge_graph = graph_builder.build_from_memories(&memories).await?;

    println!("Generated {} concepts and {} relationships",
             knowledge_graph.nodes.len(),
             knowledge_graph.relationships.len());

    // Generate content synthesis
    let synthesizer = ContentSynthesizer::new(config.clone());
    let summary = synthesizer.generate_summary("AI research", &memories).await?;
    let insights = synthesizer.extract_insights("machine learning", &memories).await?;

    println!("Summary confidence: {:.1}%", summary.confidence * 100.0);
    println!("Key insights: {}", insights.key_points.len());

    // Generate analytics dashboard
    let dashboard = AnalyticsDashboard::new(config);
    let dashboard_output = dashboard.generate_dashboard(raw_data).await?;

    println!("Generated {} visualizations and {} insights",
             dashboard_output.visualizations.len(),
             dashboard_output.insights.len());

    Ok(())
}
```

## 📖 Examples

### Process a PDF and Chat

```rust
use rust_mem_vid::{MemvidEncoder, MemvidChat, video::Codec};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rust_mem_vid::init().await?;
    
    // Create encoder and process PDF
    let mut encoder = MemvidEncoder::new().await?;
    encoder.add_pdf("document.pdf").await?;
    
    // Build video with H.264 codec
    encoder.build_video_with_codec(
        "document_memory.mp4", 
        "document_index.json", 
        Some(Codec::H264)
    ).await?;
    
    // Start chat session
    let mut chat = MemvidChat::new("document_memory.mp4", "document_index.json").await?;
    chat.set_provider("openai")?;
    
    let response = chat.chat("Summarize the main points of this document").await?;
    println!("{}", response);
    
    Ok(())
}
```

### Batch Processing

```rust
use rust_mem_vid::MemvidEncoder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rust_mem_vid::init().await?;
    
    let mut encoder = MemvidEncoder::new().await?;
    
    // Add multiple files
    encoder.add_pdf("book1.pdf").await?;
    encoder.add_epub("book2.epub").await?;
    encoder.add_text_file("notes.txt").await?;
    
    // Process entire directory
    encoder.add_directory("documents/").await?;
    
    // Build comprehensive video memory
    let stats = encoder.build_video("library.mp4", "library.json").await?;
    
    println!("Created library with {} chunks", stats.total_chunks);
    println!("Video size: {:.2} MB", stats.video_stats.file_size_bytes as f64 / 1024.0 / 1024.0);
    
    Ok(())
}
```

### Data Processing

```rust
use rust_mem_vid::{MemvidEncoder, DataProcessor, DataFileType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rust_mem_vid::init().await?;

    let mut encoder = MemvidEncoder::new().await?;

    // Process different data types
    encoder.add_csv_file("sales_data.csv").await?;
    encoder.add_parquet_file("analytics.parquet").await?;
    encoder.add_code_file("main.rs").await?;
    encoder.add_code_file("script.py").await?;
    encoder.add_log_file("application.log").await?;

    // Auto-detect file types
    encoder.add_file("data.json").await?;
    encoder.add_file("config.yaml").await?;

    // Build video with all data
    let stats = encoder.build_video("data_memory.mp4", "data_index.json").await?;

    println!("Processed {} chunks from various data sources", stats.total_chunks);

    Ok(())
}
```

### Folder Processing

Rust MemVid provides powerful recursive folder processing with extensive configuration options:

```rust
use rust_mem_vid::{MemvidEncoder, Config};
use rust_mem_vid::config::FolderConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rust_mem_vid::init().await?;

    // Custom folder configuration
    let folder_config = FolderConfig {
        max_depth: Some(5),                    // Limit recursion depth
        include_extensions: Some(vec![         // Only process these file types
            "rs".to_string(),
            "py".to_string(),
            "js".to_string()
        ]),
        exclude_extensions: vec![              // Skip these file types
            "exe".to_string(),
            "dll".to_string()
        ],
        exclude_patterns: vec![                // Skip paths matching these patterns
            "*/target/*".to_string(),          // Rust build artifacts
            "*/node_modules/*".to_string(),    // Node.js dependencies
            "*/.git/*".to_string(),            // Git repository
        ],
        min_file_size: 10,                     // Skip files smaller than 10 bytes
        max_file_size: 50 * 1024 * 1024,      // Skip files larger than 50MB
        follow_symlinks: false,                // Don't follow symbolic links
        include_hidden: false,                 // Skip hidden files
        skip_binary: true,                     // Skip binary files
    };

    let mut config = Config::default();
    config.folder = folder_config;

    let mut encoder = MemvidEncoder::new_with_config(config).await?;

    // Process directories with custom configuration
    let stats = encoder.add_directory("./src").await?;

    println!("Processed {} files, {} failed", stats.files_processed, stats.files_failed);

    // Preview files before processing
    let files = encoder.preview_directory("./data")?;
    println!("Would process {} files", files.len());

    // Process multiple directories
    let all_stats = encoder.add_directories(&["./src", "./tests", "./examples"]).await?;

    encoder.build_video("codebase.mp4", "codebase.json").await?;

    Ok(())
}
```

#### Folder Processing Features

- **🔍 Smart File Discovery**: Automatically finds all supported file types
- **📏 Depth Control**: Configurable recursion depth limiting
- **🎯 File Type Filtering**: Include/exclude specific file extensions
- **📋 Pattern Matching**: Glob pattern support for path exclusion
- **📊 Size Filtering**: Min/max file size limits
- **🔗 Symlink Handling**: Configurable symbolic link following
- **👁️ Hidden File Support**: Optional processing of hidden files
- **🔍 Binary Detection**: Automatic binary file detection and skipping
- **📈 Progress Tracking**: Detailed statistics and progress reporting
- **👀 Preview Mode**: Preview files before processing

## 🔧 Configuration

Create a `memvid.toml` configuration file:

```toml
log_level = "info"

[qr]
version = 1
error_correction = "M"
box_size = 10
border = 4

[video]
default_codec = "h264"
fps = 30.0
frame_width = 512
frame_height = 512
use_hardware_acceleration = true

[text]
chunk_size = 512
overlap = 50
max_chunk_size = 2048

[embeddings]
model_name = "sentence-transformers/all-MiniLM-L6-v2"
use_gpu = false
batch_size = 32

[search]
default_top_k = 5
use_faiss = true
cache_size = 1000

[chat]
default_provider = "openai"
max_context_length = 4000
temperature = 0.7

[chat.providers.openai]
endpoint = "https://api.openai.com/v1/chat/completions"
model = "gpt-3.5-turbo"

[chat.providers.anthropic]
endpoint = "https://api.anthropic.com/v1/messages"
model = "claude-3-sonnet-20240229"
```

## 🎯 Use Cases

### Traditional Memory Storage
- **📖 Digital Libraries**: Index thousands of books in a single video file
- **🎓 Educational Content**: Create searchable video memories of course materials
- **📰 Research Archives**: Compress years of papers into manageable video databases
- **💼 Corporate Knowledge**: Build company-wide searchable knowledge bases
- **🔬 Scientific Literature**: Quick semantic search across research papers
- **📝 Personal Notes**: Transform your notes into a searchable AI assistant
- **📊 Data Analytics**: Store and search through CSV, Parquet, and JSON datasets
- **💻 Code Documentation**: Index entire codebases with intelligent chunking
- **📋 Log Analysis**: Process and search through application logs efficiently
- **🏢 Business Intelligence**: Create searchable repositories of structured data

### 🕰️ Temporal Analysis & Evolution Tracking
- **🔬 Research Project Evolution**: Track how research develops from proposal to publication
- **📚 Knowledge Base Maintenance**: Monitor content quality and identify knowledge gaps
- **👥 Team Collaboration**: Merge insights from multiple team members' memories
- **📈 Content Quality Assessment**: Analyze how documentation improves over time
- **🎯 Learning Progress Tracking**: Monitor personal knowledge growth and retention
- **🏢 Organizational Memory**: Track institutional knowledge evolution
- **📊 Information Decay Detection**: Identify outdated or conflicting information
- **🔄 Version Control for Knowledge**: Git-like diff analysis for memory content
- **🎓 Educational Assessment**: Track student understanding development
- **💡 Innovation Tracking**: Monitor how ideas and concepts evolve in organizations

### 🧠 AI Intelligence & Knowledge Analysis
- **🕸️ Concept Mapping**: Automatically discover relationships between ideas and topics
- **🤖 Intelligent Summarization**: Generate AI-powered summaries and insights from vast content
- **🔍 Knowledge Gap Analysis**: Identify missing information and research opportunities
- **📊 Content Quality Assessment**: Analyze information density, coherence, and freshness
- **🎯 Research Optimization**: Get recommendations for improving research methodologies
- **👥 Expertise Discovery**: Map team knowledge and identify collaboration opportunities
- **💡 Innovation Insights**: Discover emerging patterns and breakthrough opportunities
- **🔗 Cross-Domain Analysis**: Find unexpected connections between different fields
- **📈 Knowledge Evolution Tracking**: Monitor how understanding develops over time
- **🎓 Educational Content Optimization**: Improve learning materials based on knowledge analysis

## 🏗️ Architecture

The library consists of several key components:

### Core Components
- **QR Processor**: Encodes/decodes text chunks to/from QR codes with size optimization
- **Video Encoder/Decoder**: Handles video creation and frame extraction using FFmpeg
- **Embedding Model**: Generates semantic embeddings using transformer models
- **Index Manager**: Manages vector search and metadata storage
- **Text Processor**: Handles document parsing and text chunking
- **Data Processor**: Advanced processing for CSV, Parquet, JSON, and code files
- **Code Analyzer**: Intelligent parsing and chunking of source code
- **Chat Interface**: Integrates with LLM APIs for conversational search

### 🕰️ Temporal Analysis Components
- **Memory Diff Engine**: Compares memory videos with chunk-level analysis
- **Multi-Memory Engine**: Manages and searches across multiple memory videos
- **Temporal Analysis Engine**: Tracks memory evolution and identifies trends
- **Correlation Detector**: Finds relationships between different memory snapshots
- **Timeline Builder**: Creates comprehensive evolution timelines
- **Knowledge Gap Analyzer**: Identifies areas needing attention or updates

### 🧠 AI Intelligence Components
- **Knowledge Graph Builder**: Constructs concept relationship networks from memory content
- **Concept Extractors**: Named entity recognition, keyword extraction, technical term identification
- **Relationship Analyzers**: Co-occurrence analysis, semantic similarity, hierarchical relationships
- **Content Synthesizer**: Generates summaries, insights, and recommendations using AI strategies
- **Analytics Dashboard**: Creates comprehensive visual dashboards with metrics and insights
- **Intelligence Engine**: Coordinates AI analysis across multiple memory videos

## 🔧 Dependencies

### System Requirements

- **FFmpeg**: Required for video encoding/decoding
- **OpenCV**: Used for image processing (optional, can use FFmpeg only)

### Installation on Different Platforms

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg libopencv-dev pkg-config
```

#### macOS
```bash
brew install ffmpeg opencv pkg-config
```

#### Windows
```bash
# Using vcpkg
vcpkg install ffmpeg opencv
```

### Rust Dependencies

The library uses several high-quality Rust crates:

- **Video**: `ffmpeg-next`, `opencv`
- **ML/AI**: `candle-core`, `candle-transformers`, `tokenizers`
- **QR Codes**: `qrcode`, `rqrr`, `image`
- **Search**: `faiss` (optional), `hnswlib`
- **Documents**: `pdf-extract`, `epub`
- **Data Processing**: `polars`, `arrow`, `parquet`, `csv`
- **Code Analysis**: `tree-sitter`, `syntect`
- **Async**: `tokio`, `futures`, `rayon`

## 🚀 Performance

### Benchmarks

- **Encoding**: ~1000 chunks/second on modern hardware
- **Search**: Sub-second semantic search across millions of chunks
- **Compression**: 10x better than traditional text storage
- **Memory**: Efficient streaming with configurable caching

### Optimization Tips

1. **Use H.265 codec** for maximum compression
2. **Enable GPU acceleration** for embedding generation
3. **Tune chunk size** based on your content type
4. **Use FAISS** for large-scale vector search
5. **Configure caching** for frequently accessed content

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run with features
cargo test --features faiss

# Run examples
cargo run --example basic_usage
cargo run --example pdf_chat document.pdf
cargo run --example data_demo
cargo run --example folder_demo

# NEW: Temporal analysis examples
cargo run --example temporal_analysis_demo

# NEW: AI Intelligence examples
cargo run --example ai_intelligence_demo

# QR optimization examples
cargo run --example qr_size_test
```

## 📚 Additional Documentation

- **[TEMPORAL_ANALYSIS_FEATURES.md](TEMPORAL_ANALYSIS_FEATURES.md)** - Comprehensive guide to temporal analysis and memory comparison features
- **[AI_INTELLIGENCE_FEATURES.md](AI_INTELLIGENCE_FEATURES.md)** - Complete guide to AI Intelligence features (knowledge graphs, content synthesis, analytics)
- **[QR_SIZE_OPTIMIZATION.md](QR_SIZE_OPTIMIZATION.md)** - Guide to QR code size optimization and troubleshooting
- **[examples/memories_config.json](examples/memories_config.json)** - Template for multi-memory search configuration
- **[examples/temporal_analysis_demo.rs](examples/temporal_analysis_demo.rs)** - Working demonstration of all temporal features
- **[examples/ai_intelligence_demo.rs](examples/ai_intelligence_demo.rs)** - Working demonstration of all AI Intelligence features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/njfio/rust-mp4-memory.git
cd rust_mem_vid
cargo build
cargo test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original [MemVid Python library](https://github.com/Olow304/memvid) by Olow304
- [Candle](https://github.com/huggingface/candle) for ML inference in Rust
- [FFmpeg](https://ffmpeg.org/) for video processing
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 🔗 Related Projects

- [Original MemVid (Python)](https://github.com/Olow304/memvid)
- [Candle ML Framework](https://github.com/huggingface/candle)
- [FAISS Vector Search](https://github.com/facebookresearch/faiss)

---

## 🎉 What Makes This Special

Rust MemVid isn't just another storage system - it's a **complete intelligent knowledge evolution platform**:

### 📹 **Revolutionary Storage**
- **Each MP4 is a frozen snapshot** of your knowledge at a specific point in time
- **Massive capacity** - store millions of text chunks in a single video file
- **Offline-first** - no internet required after video generation

### 🕰️ **Temporal Intelligence**
- **Compare any two snapshots** to see exactly what changed with detailed diff analysis
- **Search across multiple memories** simultaneously to find correlations and patterns
- **Track knowledge evolution** over time with sophisticated temporal analysis
- **Identify knowledge gaps** and optimization opportunities automatically

### 🧠 **AI-Powered Intelligence**
- **Automatic concept extraction** and relationship mapping from your content
- **Intelligent content synthesis** with AI-generated summaries and insights
- **Advanced analytics dashboards** with visual knowledge evolution tracking
- **Multi-strategy analysis** using named entity recognition, keyword extraction, and semantic analysis
- **Knowledge gap detection** and actionable recommendations

### 👥 **Collaborative Knowledge**
- **Team memory merging** and collaborative knowledge building
- **Cross-memory analysis** to find unexpected connections
- **Expertise mapping** to identify collaboration opportunities

**Transform from simple storage to intelligent knowledge evolution with AI-powered analysis!**

---

**Ready to revolutionize your AI memory management with intelligent temporal analysis? Install rust_mem_vid and start building the future of knowledge management!** 🚀🧠🕰️
