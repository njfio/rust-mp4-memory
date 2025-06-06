//! Command-line interface for rust_mem_vid

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error, warn};

use rust_mem_vid::{
    MemvidEncoder, MemvidRetriever, MemvidChat, Config,
    video::Codec, utils::format_file_size, utils::format_duration,
};

#[derive(Parser)]
#[command(name = "memvid")]
#[command(about = "A Rust implementation of MemVid - video-based AI memory")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode text into a QR code video
    Encode {
        /// Output video file
        #[arg(short, long)]
        output: String,

        /// Output index file
        #[arg(short, long)]
        index: String,

        /// Input text files
        #[arg(short, long)]
        files: Vec<PathBuf>,

        /// Input directories
        #[arg(short, long)]
        dirs: Vec<PathBuf>,

        /// Direct text input
        #[arg(short, long)]
        text: Vec<String>,

        /// Video codec
        #[arg(short = 'c', long, default_value = "mp4v")]
        codec: String,

        /// Chunk size (characters per QR code)
        #[arg(long, default_value = "1000")]
        chunk_size: usize,

        /// Chunk overlap
        #[arg(long, default_value = "50")]
        overlap: usize,

        /// Maximum recursion depth for directories (default: 10)
        #[arg(long)]
        max_depth: Option<usize>,

        /// Include only these file extensions (comma-separated, e.g., "rs,py,js")
        #[arg(long)]
        include_extensions: Option<String>,

        /// Exclude these file extensions (comma-separated, e.g., "exe,dll,bin")
        #[arg(long)]
        exclude_extensions: Option<String>,

        /// Follow symbolic links
        #[arg(long)]
        follow_symlinks: bool,

        /// Include hidden files and directories
        #[arg(long)]
        include_hidden: bool,

        /// Maximum file size to process in MB (default: 100)
        #[arg(long)]
        max_file_size: Option<usize>,
    },

    /// Search a QR code video
    Search {
        /// Video file
        #[arg(short, long)]
        video: String,

        /// Index file
        #[arg(short, long)]
        index: String,

        /// Search query
        #[arg(short, long)]
        query: String,

        /// Number of results
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Show metadata
        #[arg(short, long)]
        metadata: bool,

        /// Context window size
        #[arg(short = 'w', long)]
        context_window: Option<usize>,
    },

    /// Start interactive chat
    Chat {
        /// Video file
        #[arg(short, long)]
        video: String,

        /// Index file
        #[arg(short, long)]
        index: String,

        /// LLM provider
        #[arg(short, long, default_value = "openai")]
        provider: String,
    },

    /// Get information about a video or index
    Info {
        /// Video file
        #[arg(short, long)]
        video: Option<String>,

        /// Index file
        #[arg(short, long)]
        index: Option<String>,
    },

    /// Extract and decode a specific frame
    Extract {
        /// Video file
        #[arg(short, long)]
        video: String,

        /// Frame number
        #[arg(short, long)]
        frame: u32,

        /// Output file (optional)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Compare two memory videos and generate a diff
    Diff {
        /// Old memory video file
        old_video: String,
        /// Old memory index file
        old_index: String,
        /// New memory video file
        new_video: String,
        /// New memory index file
        new_index: String,
        /// Output diff file (JSON format)
        #[arg(short, long)]
        output: Option<String>,
        /// Enable semantic analysis
        #[arg(long)]
        semantic: bool,
    },

    /// Search across multiple memory videos
    MultiSearch {
        /// Search query
        query: String,
        /// Memory configuration file (JSON with memory paths and metadata)
        memories_config: String,
        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        /// Enable cross-memory correlations
        #[arg(long)]
        correlations: bool,
        /// Enable temporal analysis
        #[arg(long)]
        temporal: bool,
        /// Filter by memory tags
        #[arg(long)]
        tags: Option<Vec<String>>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = tracing_subscriber::EnvFilter::new(&cli.log_level);
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    // Load configuration
    let config = if let Some(config_path) = cli.config {
        Config::from_file(config_path)?
    } else {
        Config::default()
    };

    // Initialize the library
    rust_mem_vid::init_with_config(config.clone()).await?;

    match cli.command {
        Commands::Encode {
            output,
            index,
            files,
            dirs,
            text,
            codec,
            chunk_size,
            overlap,
            max_depth,
            include_extensions,
            exclude_extensions,
            follow_symlinks,
            include_hidden,
            max_file_size,
        } => {
            encode_command(
                output, index, files, dirs, text, codec, chunk_size, overlap,
                max_depth, include_extensions, exclude_extensions, follow_symlinks,
                include_hidden, max_file_size, config
            ).await?;
        }

        Commands::Search {
            video,
            index,
            query,
            top_k,
            metadata,
            context_window,
        } => {
            search_command(video, index, query, top_k, metadata, context_window, config).await?;
        }

        Commands::Chat {
            video,
            index,
            provider,
        } => {
            chat_command(video, index, provider, config).await?;
        }

        Commands::Info { video, index } => {
            info_command(video, index, config).await?;
        }

        Commands::Extract {
            video,
            frame,
            output,
        } => {
            extract_command(video, frame, output, config).await?;
        }

        Commands::Diff {
            old_video,
            old_index,
            new_video,
            new_index,
            output,
            semantic,
        } => {
            diff_command(old_video, old_index, new_video, new_index, output, semantic, config).await?;
        }

        Commands::MultiSearch {
            query,
            memories_config,
            top_k,
            correlations,
            temporal,
            tags,
        } => {
            multi_search_command(query, memories_config, top_k, correlations, temporal, tags, config).await?;
        }
    }

    Ok(())
}

async fn encode_command(
    output: String,
    index: String,
    files: Vec<PathBuf>,
    dirs: Vec<PathBuf>,
    text: Vec<String>,
    codec: String,
    chunk_size: usize,
    overlap: usize,
    max_depth: Option<usize>,
    include_extensions: Option<String>,
    exclude_extensions: Option<String>,
    follow_symlinks: bool,
    include_hidden: bool,
    max_file_size: Option<usize>,
    mut config: Config,
) -> anyhow::Result<()> {
    info!("Starting encoding process...");

    // Update config with command line parameters
    config.text.chunk_size = chunk_size;
    config.text.overlap = overlap;

    // Validate chunk size for QR compatibility
    if chunk_size > 1500 {
        warn!("Large chunk size ({}) may cause QR code encoding failures. Consider using --chunk-size 1000 or smaller.", chunk_size);
    }

    // Update folder config with command line parameters
    if let Some(depth) = max_depth {
        config.folder.max_depth = Some(depth);
    }

    if let Some(extensions) = include_extensions {
        let exts: Vec<String> = extensions.split(',').map(|s| s.trim().to_string()).collect();
        config.folder.include_extensions = Some(exts);
    }

    if let Some(extensions) = exclude_extensions {
        let mut exts: Vec<String> = extensions.split(',').map(|s| s.trim().to_string()).collect();
        config.folder.exclude_extensions.append(&mut exts);
    }

    config.folder.follow_symlinks = follow_symlinks;
    config.folder.include_hidden = include_hidden;

    if let Some(size_mb) = max_file_size {
        config.folder.max_file_size = size_mb * 1024 * 1024; // Convert MB to bytes
    }

    let mut encoder = MemvidEncoder::new_with_config(config).await?;

    // Add direct text input
    if !text.is_empty() {
        encoder.add_chunks(text).await?;
    }

    // Add files
    for file in files {
        let file_str = file.to_string_lossy();
        match encoder.add_file(&file_str).await {
            Ok(_) => {
                info!("Successfully added file: {}", file_str);
            }
            Err(e) => {
                error!("Failed to add file {}: {}", file_str, e);
                continue;
            }
        }
    }

    // Add directories
    let mut total_folder_stats = Vec::new();
    for dir in dirs {
        info!("Processing directory: {}", dir.display());
        match encoder.add_directory(&dir.to_string_lossy()).await {
            Ok(stats) => {
                info!("Directory {} processed: {} files processed, {} failed",
                      dir.display(), stats.files_processed, stats.files_failed);
                total_folder_stats.push(stats);
            }
            Err(e) => {
                error!("Failed to process directory {}: {}", dir.display(), e);
            }
        }
    }

    if encoder.is_empty() {
        error!("No content to encode. Please provide files, directories, or text.");
        return Ok(());
    }

    // Parse codec
    let codec = Codec::from_str(&codec)?;

    // Build video
    let stats = match encoder.build_video_with_codec(&output, &index, Some(codec)).await {
        Ok(stats) => stats,
        Err(e) => {
            if e.to_string().contains("data too long") || e.to_string().contains("Data too large for QR code") {
                error!("QR code encoding failed due to large chunk size!");
                error!("Current chunk size: {} characters", chunk_size);
                error!("");
                error!("💡 Solutions:");
                error!("   1. Reduce chunk size: --chunk-size 800");
                error!("   2. Use lower error correction: (modify config)");
                error!("   3. Process smaller files or use file filtering");
                error!("");
                error!("Example: memvid encode --chunk-size 800 --output {} --index {} [your files]", output, index);
                return Err(e.into());
            } else {
                return Err(e.into());
            }
        }
    };

    // Print results
    println!("✅ Encoding completed successfully!");
    println!("📊 Statistics:");
    println!("   • Total chunks: {}", stats.total_chunks);
    println!("   • Total characters: {}", stats.total_characters);
    println!("   • Video file: {} ({})", output, format_file_size(stats.video_stats.file_size_bytes));
    println!("   • Duration: {:.1}s", stats.video_stats.duration_seconds);
    println!("   • FPS: {:.1}", stats.video_stats.fps);
    println!("   • Codec: {}", stats.video_stats.codec);
    println!("   • Encoding time: {}", format_duration(stats.encoding_time_seconds));
    println!("   • Index file: {}", index);

    // Print folder processing statistics if any directories were processed
    if !total_folder_stats.is_empty() {
        println!("\n📁 Folder Processing Statistics:");
        let total_files_found: usize = total_folder_stats.iter().map(|s| s.files_found).sum();
        let total_files_processed: usize = total_folder_stats.iter().map(|s| s.files_processed).sum();
        let total_files_failed: usize = total_folder_stats.iter().map(|s| s.files_failed).sum();
        let total_bytes_processed: u64 = total_folder_stats.iter().map(|s| s.bytes_processed).sum();
        let total_processing_time: u64 = total_folder_stats.iter().map(|s| s.processing_time_ms).sum();

        println!("   • Directories scanned: {}", total_folder_stats.len());
        println!("   • Files found: {}", total_files_found);
        println!("   • Files processed: {}", total_files_processed);
        println!("   • Files failed: {}", total_files_failed);
        println!("   • Data processed: {}", format_file_size(total_bytes_processed));
        println!("   • Processing time: {}ms", total_processing_time);

        if total_files_failed > 0 {
            println!("   ⚠️  {} files failed to process", total_files_failed);
        }
    }

    Ok(())
}

async fn search_command(
    video: String,
    index: String,
    query: String,
    top_k: usize,
    metadata: bool,
    context_window: Option<usize>,
    config: Config,
) -> anyhow::Result<()> {
    info!("Searching for: '{}'", query);

    let retriever = MemvidRetriever::new_with_config(&video, &index, config).await?;

    if let Some(window_size) = context_window {
        let results = retriever.search_with_context(&query, top_k, window_size).await?;
        
        println!("🔍 Search results with context:");
        for (i, result) in results.iter().enumerate() {
            println!("\n{}. [Similarity: {:.3}] {}", 
                     i + 1, result.main_result.similarity, result.main_result.text);
            
            if metadata {
                println!("   📍 Chunk ID: {}, Frame: {}", 
                         result.main_result.chunk_id, result.main_result.frame_number);
                if let Some(ref source) = result.main_result.metadata.source {
                    println!("   📄 Source: {}", source);
                }
            }
            
            if !result.context.is_empty() {
                println!("   📝 Context:");
                for ctx in &result.context {
                    if ctx.chunk_id != result.main_result.chunk_id {
                        let preview = if ctx.text.len() > 100 {
                            format!("{}...", &ctx.text[..100])
                        } else {
                            ctx.text.clone()
                        };
                        println!("      • {}", preview);
                    }
                }
            }
        }
    } else {
        let results = retriever.search_with_metadata(&query, top_k).await?;
        
        println!("🔍 Search results:");
        for (i, result) in results.iter().enumerate() {
            println!("\n{}. [Similarity: {:.3}] {}", i + 1, result.similarity, result.text);
            
            if metadata {
                println!("   📍 Chunk ID: {}, Frame: {}", result.chunk_id, result.frame_number);
                if let Some(ref source) = result.metadata.source {
                    println!("   📄 Source: {}", source);
                }
            }
        }
    }

    Ok(())
}

async fn chat_command(
    video: String,
    index: String,
    provider: String,
    config: Config,
) -> anyhow::Result<()> {
    info!("Starting interactive chat...");

    let mut chat = MemvidChat::new_with_config(&video, &index, config).await?;
    chat.set_provider(&provider)?;

    println!("💬 MemVid Chat Interface");
    println!("Type 'quit' or 'exit' to end the conversation.");
    println!("Type 'clear' to clear conversation history.");
    println!("Type 'stats' to show statistics.\n");

    loop {
        print!("You: ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("👋 Goodbye!");
                break;
            }
            "clear" => {
                chat.clear_history();
                println!("🧹 Conversation history cleared.");
                continue;
            }
            "stats" => {
                let stats = chat.get_stats();
                println!("📊 Chat Statistics:");
                println!("   • Total messages: {}", stats.total_messages);
                println!("   • User messages: {}", stats.user_messages);
                println!("   • Assistant messages: {}", stats.assistant_messages);
                println!("   • Current provider: {}", stats.current_provider);
                println!("   • Index chunks: {}", stats.retriever_stats.index_stats.total_chunks);
                continue;
            }
            "" => continue,
            _ => {}
        }

        print!("🤖 Assistant: ");
        io::stdout().flush()?;

        match chat.chat(input).await {
            Ok(response) => {
                println!("{}\n", response);
            }
            Err(e) => {
                error!("Chat error: {}", e);
                println!("❌ Sorry, I encountered an error. Please try again.\n");
            }
        }
    }

    Ok(())
}

async fn info_command(
    video: Option<String>,
    index: Option<String>,
    config: Config,
) -> anyhow::Result<()> {
    if let Some(video_path) = video {
        let decoder = rust_mem_vid::video::VideoDecoder::new(config.clone());
        let video_info = decoder.get_video_info(&video_path).await?;
        
        println!("🎥 Video Information:");
        println!("   • File: {}", video_path);
        println!("   • Dimensions: {}x{}", video_info.width, video_info.height);
        println!("   • FPS: {:.1}", video_info.fps);
        println!("   • Duration: {}", format_duration(video_info.duration_seconds));
        println!("   • Total frames: {}", video_info.total_frames);
        println!("   • Codec: {}", video_info.codec);
        println!("   • Pixel format: {}", video_info.pixel_format);
        
        let file_size = std::fs::metadata(&video_path)?.len();
        println!("   • File size: {}", format_file_size(file_size));
    }

    if let Some(index_path) = index {
        let index_manager = rust_mem_vid::index::IndexManager::load_readonly(&index_path)?;
        let stats = index_manager.get_stats();
        
        println!("📚 Index Information:");
        println!("   • File: {}", index_path);
        println!("   • Total chunks: {}", stats.total_chunks);
        println!("   • Total characters: {}", stats.total_characters);
        println!("   • Average chunk length: {:.1}", stats.avg_chunk_length);
        println!("   • Unique sources: {}", stats.unique_sources);
        println!("   • Embedding dimension: {}", stats.embedding_dimension);
        println!("   • Has embedding model: {}", stats.has_embedding_model);
    }

    Ok(())
}

async fn extract_command(
    video: String,
    frame: u32,
    output: Option<String>,
    config: Config,
) -> anyhow::Result<()> {
    info!("Extracting frame {} from {}", frame, video);

    // Try to create a retriever, but if it fails (no index), extract directly
    match MemvidRetriever::new_with_config(&video, "dummy.json", config.clone()).await {
        Ok(mut retriever) => {
            let decoded = retriever.extract_frame(frame).await?;

            if let Some(ref output_path) = output {
                std::fs::write(output_path, &decoded)?;
                println!("💾 Decoded content saved to {}", output_path);
            } else {
                println!("📄 Decoded content from frame {}:", frame);
                println!("{}", decoded);
            }
        }
        Err(_) => {
            // If index doesn't exist, extract frame directly
            let qr_processor = rust_mem_vid::qr::QrProcessor::default();
            let video_decoder = rust_mem_vid::video::VideoDecoder::new(config);

            let image = video_decoder.extract_frame(&video, frame).await?;
            let decoded = qr_processor.decode_qr(&image)?;

            if let Some(ref output_path) = output {
                std::fs::write(output_path, &decoded)?;
                println!("💾 Decoded content saved to {}", output_path);
            } else {
                println!("📄 Decoded content from frame {}:", frame);
                println!("{}", decoded);
            }
        }
    }

    Ok(())
}

async fn diff_command(
    old_video: String,
    old_index: String,
    new_video: String,
    new_index: String,
    output: Option<String>,
    semantic: bool,
    config: Config,
) -> anyhow::Result<()> {
    use rust_mem_vid::memory_diff::MemoryDiffEngine;

    info!("Comparing memories: {} vs {}", old_video, new_video);

    let mut diff_engine = MemoryDiffEngine::new(config);
    if semantic {
        diff_engine = diff_engine.with_semantic_analysis(true);
    }

    let diff = diff_engine.compare_memories(&old_video, &old_index, &new_video, &new_index).await?;

    // Print summary
    println!("🔍 Memory Comparison Results");
    println!("============================");
    println!("Old memory: {}", old_video);
    println!("New memory: {}", new_video);
    println!();
    println!("📊 Summary:");
    println!("   • Old chunks: {}", diff.summary.total_old_chunks);
    println!("   • New chunks: {}", diff.summary.total_new_chunks);
    println!("   • Added: {} chunks", diff.summary.added_count);
    println!("   • Removed: {} chunks", diff.summary.removed_count);
    println!("   • Modified: {} chunks", diff.summary.modified_count);
    println!("   • Unchanged: {} chunks", diff.summary.unchanged_count);
    println!("   • Similarity: {:.1}%", diff.summary.similarity_score * 100.0);
    println!("   • Growth ratio: {:.2}x", diff.summary.content_growth_ratio);

    // Save detailed diff if requested
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&diff)?;
        std::fs::write(&output_path, json)?;
        println!("\n💾 Detailed diff saved to: {}", output_path);
    }

    Ok(())
}

async fn multi_search_command(
    query: String,
    memories_config: String,
    top_k: usize,
    correlations: bool,
    temporal: bool,
    tags: Option<Vec<String>>,
    config: Config,
) -> anyhow::Result<()> {
    use rust_mem_vid::multi_memory::{MultiMemoryEngine, MemoryFilter};

    info!("Multi-memory search for: '{}'", query);

    // Load memories configuration
    let config_content = std::fs::read_to_string(&memories_config)?;
    let memories_list: Vec<serde_json::Value> = serde_json::from_str(&config_content)?;

    let mut engine = MultiMemoryEngine::new(config);

    // Add memories from configuration
    for memory_config in memories_list {
        let name = memory_config["name"].as_str().unwrap_or("unknown");
        let video_path = memory_config["video_path"].as_str().unwrap();
        let index_path = memory_config["index_path"].as_str().unwrap();
        let memory_tags: Vec<String> = memory_config["tags"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        let description = memory_config["description"].as_str().map(|s| s.to_string());

        match engine.add_memory(name, video_path, index_path, memory_tags, description).await {
            Ok(_) => info!("Added memory: {}", name),
            Err(e) => warn!("Failed to add memory {}: {}", name, e),
        }
    }

    // Perform search
    let filter = if let Some(tag_filter) = tags {
        MemoryFilter::Tags(tag_filter)
    } else {
        MemoryFilter::All
    };

    let results = if matches!(filter, MemoryFilter::All) {
        engine.search_all(&query, top_k, correlations, temporal).await?
    } else {
        engine.search_filtered(&query, top_k, filter).await?
    };

    // Display results
    println!("🔍 Multi-Memory Search Results");
    println!("==============================");
    println!("Query: '{}'", query);
    println!("Searched {} memories in {}ms",
             results.search_metadata.memories_searched,
             results.search_metadata.search_time_ms);
    println!("Total results: {}", results.total_results);

    // Show aggregated results
    println!("\n📋 Top Results:");
    for (i, result) in results.aggregated_results.iter().take(top_k).enumerate() {
        println!("\n{}. [Similarity: {:.3}] [Memory: {}]",
                 i + 1, result.similarity, result.source_memory);

        let preview = if result.text.len() > 200 {
            format!("{}...", &result.text[..200])
        } else {
            result.text.clone()
        };
        println!("   {}", preview);
    }

    Ok(())
}
