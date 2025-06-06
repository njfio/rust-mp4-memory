//! Command-line interface for rust_mem_vid

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};

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

        /// Chunk size
        #[arg(long, default_value = "512")]
        chunk_size: usize,

        /// Chunk overlap
        #[arg(long, default_value = "50")]
        overlap: usize,
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
        } => {
            encode_command(output, index, files, dirs, text, codec, chunk_size, overlap, config).await?;
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
    mut config: Config,
) -> anyhow::Result<()> {
    info!("Starting encoding process...");

    // Update config with command line parameters
    config.text.chunk_size = chunk_size;
    config.text.overlap = overlap;

    let mut encoder = MemvidEncoder::new_with_config(config).await?;

    // Add direct text input
    if !text.is_empty() {
        encoder.add_chunks(text).await?;
    }

    // Add files
    for file in files {
        let file_str = file.to_string_lossy();
        match file.extension().and_then(|ext| ext.to_str()) {
            Some("pdf") => encoder.add_pdf(&file_str).await?,
            Some("epub") => encoder.add_epub(&file_str).await?,
            Some("txt") | Some("md") | Some("rst") => encoder.add_text_file(&file_str).await?,
            _ => {
                error!("Unsupported file type: {}", file_str);
                continue;
            }
        }
    }

    // Add directories
    for dir in dirs {
        encoder.add_directory(&dir.to_string_lossy()).await?;
    }

    if encoder.is_empty() {
        error!("No content to encode. Please provide files, directories, or text.");
        return Ok(());
    }

    // Parse codec
    let codec = Codec::from_str(&codec)?;

    // Build video
    let stats = encoder.build_video_with_codec(&output, &index, Some(codec)).await?;

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
