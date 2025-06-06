//! Basic usage example for rust_mem_vid

use rust_mem_vid::{MemvidEncoder, MemvidRetriever, MemvidChat};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the library
    rust_mem_vid::init().await?;

    println!("🚀 Basic MemVid Usage Example");

    // Create some sample text chunks
    let chunks = vec![
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typography.".to_string(),
        "Artificial intelligence is transforming how we work, learn, and interact with technology.".to_string(),
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.".to_string(),
        "Video compression algorithms like H.264 and H.265 enable efficient storage and transmission of video data.".to_string(),
        "Machine learning models can process vast amounts of data to identify patterns and make predictions.".to_string(),
    ];

    // Step 1: Create encoder and add chunks
    println!("\n📝 Step 1: Creating encoder and adding chunks...");
    let mut encoder = MemvidEncoder::new().await?;
    encoder.add_chunks(chunks).await?;

    let stats = encoder.get_stats();
    println!("   • Added {} chunks", stats.total_chunks);
    println!("   • Total characters: {}", stats.total_characters);
    println!("   • Average chunk length: {:.1}", stats.avg_chunk_length);

    // Step 2: Build video
    println!("\n🎬 Step 2: Building QR code video...");
    let video_path = "example_memory.mp4";
    let index_path = "example_memory.json";
    
    let encoding_stats = encoder.build_video(video_path, index_path).await?;
    
    println!("   • Video created: {} ({:.2} MB)", 
             video_path, 
             encoding_stats.video_stats.file_size_bytes as f64 / 1024.0 / 1024.0);
    println!("   • Duration: {:.1} seconds", encoding_stats.video_stats.duration_seconds);
    println!("   • FPS: {:.1}", encoding_stats.video_stats.fps);
    println!("   • Encoding time: {:.2} seconds", encoding_stats.encoding_time_seconds);

    // Step 3: Search the memory
    println!("\n🔍 Step 3: Searching the memory...");
    let index_base = index_path.trim_end_matches(".json");

    // Check if the index files exist
    let metadata_path = format!("{}.metadata", index_base);
    let vector_path = format!("{}.vector", index_base);

    if !std::path::Path::new(&metadata_path).exists() {
        println!("   ⚠️  Index files not found. Skipping search test.");
        println!("   Expected files: {} and {}", metadata_path, vector_path);
        return Ok(());
    }

    let retriever = MemvidRetriever::new(video_path, index_base).await?;
    
    let queries = vec![
        "programming language",
        "artificial intelligence",
        "video compression",
        "machine learning",
    ];

    for query in queries {
        println!("\n   Query: '{}'", query);
        let results = retriever.search(query, 2).await?;
        
        for (i, result) in results.iter().enumerate() {
            let preview = if result.len() > 80 {
                format!("{}...", &result[..80])
            } else {
                result.clone()
            };
            println!("   {}. {}", i + 1, preview);
        }
    }

    // Step 4: Interactive chat (if API key is available)
    println!("\n💬 Step 4: Testing chat functionality...");
    
    // Check if we have an API key
    if std::env::var("OPENAI_API_KEY").is_ok() || std::env::var("ANTHROPIC_API_KEY").is_ok() {
        let mut chat = MemvidChat::new(video_path, index_base).await?;
        
        let test_questions = vec![
            "What programming languages are mentioned?",
            "Tell me about video compression",
            "What is mentioned about AI?",
        ];

        for question in test_questions {
            println!("\n   Question: {}", question);
            match chat.chat(question).await {
                Ok(response) => {
                    let preview = if response.len() > 150 {
                        format!("{}...", &response[..150])
                    } else {
                        response
                    };
                    println!("   Answer: {}", preview);
                }
                Err(e) => {
                    println!("   Error: {}", e);
                }
            }
        }
    } else {
        println!("   ⚠️  No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test chat.");
    }

    // Step 5: Get statistics
    println!("\n📊 Step 5: Final statistics...");
    let retriever_stats = retriever.get_stats();
    println!("   • Video file: {}", retriever_stats.video_path);
    println!("   • Video dimensions: {}x{}", 
             retriever_stats.video_info.width, 
             retriever_stats.video_info.height);
    println!("   • Total frames: {}", retriever_stats.video_info.total_frames);
    println!("   • Index chunks: {}", retriever_stats.index_stats.total_chunks);
    println!("   • Cache size: {}/{}", 
             retriever_stats.cache_size, 
             retriever_stats.max_cache_size);

    println!("\n✅ Example completed successfully!");
    println!("   Files created:");
    println!("   • {}", video_path);
    println!("   • {}", index_path);
    println!("   • {}.vector", index_path.trim_end_matches(".json"));
    println!("   • {}.metadata", index_path.trim_end_matches(".json"));

    Ok(())
}
