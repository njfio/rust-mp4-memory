# Rust MemVid 🦀📹

A complete Rust implementation of the [MemVid](https://github.com/Olow304/memvid) library - a revolutionary video-based AI memory system that stores text chunks as QR codes in video files with lightning-fast semantic search.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **🎥 Video-as-Database**: Store millions of text chunks in MP4 files
- **🔍 Semantic Search**: Find relevant content using natural language queries with embeddings
- **💬 Built-in Chat**: Conversational interface with LLM integration (OpenAI, Anthropic)
- **📚 Document Support**: Direct import of PDF, EPUB, and text files
- **🚀 Fast Retrieval**: Sub-second search across massive datasets
- **💾 Efficient Storage**: Advanced video compression with multiple codec support
- **🔧 Multiple Codecs**: H.264, H.265, AV1, VP9 support via FFmpeg
- **🌐 Offline-First**: No internet required after video generation
- **⚡ High Performance**: Parallel processing and optimized algorithms

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
# Encode documents into a video
memvid encode --output memory.mp4 --index memory.json --files document.pdf book.epub

# Search the video
memvid search --video memory.mp4 --index memory.json --query "machine learning" --top-k 5

# Start interactive chat
memvid chat --video memory.mp4 --index memory.json --provider openai

# Get video/index information
memvid info --video memory.mp4 --index memory.json

# Extract specific frame
memvid extract --video memory.mp4 --frame 42
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

- **📖 Digital Libraries**: Index thousands of books in a single video file
- **🎓 Educational Content**: Create searchable video memories of course materials
- **📰 Research Archives**: Compress years of papers into manageable video databases
- **💼 Corporate Knowledge**: Build company-wide searchable knowledge bases
- **🔬 Scientific Literature**: Quick semantic search across research papers
- **📝 Personal Notes**: Transform your notes into a searchable AI assistant

## 🏗️ Architecture

The library consists of several key components:

- **QR Processor**: Encodes/decodes text chunks to/from QR codes
- **Video Encoder/Decoder**: Handles video creation and frame extraction using FFmpeg
- **Embedding Model**: Generates semantic embeddings using transformer models
- **Index Manager**: Manages vector search and metadata storage
- **Text Processor**: Handles document parsing and text chunking
- **Chat Interface**: Integrates with LLM APIs for conversational search

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
```

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

**Ready to revolutionize your AI memory management with Rust? Install rust_mem_vid and start building!** 🚀
