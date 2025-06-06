//! PDF chat example for rust_mem_vid

use rust_mem_vid::{MemvidEncoder, MemvidChat, video::Codec};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the library
    rust_mem_vid::init().await?;

    println!("📚 PDF Chat Example for MemVid");

    // Check if a PDF file is provided as argument
    let args: Vec<String> = std::env::args().collect();
    let pdf_path = if args.len() > 1 {
        &args[1]
    } else {
        println!("Usage: cargo run --example pdf_chat <path_to_pdf>");
        println!("Example: cargo run --example pdf_chat document.pdf");
        return Ok(());
    };

    if !Path::new(pdf_path).exists() {
        eprintln!("❌ Error: PDF file '{}' not found", pdf_path);
        return Ok(());
    }

    println!("📄 Processing PDF: {}", pdf_path);

    // Step 1: Create encoder and process PDF
    println!("\n🔧 Step 1: Creating encoder and processing PDF...");
    let mut encoder = MemvidEncoder::new().await?;
    
    // Add the PDF file
    encoder.add_pdf(pdf_path).await?;
    
    let stats = encoder.get_stats();
    println!("   • Extracted {} chunks from PDF", stats.total_chunks);
    println!("   • Total characters: {}", stats.total_characters);
    println!("   • Average chunk length: {:.1}", stats.avg_chunk_length);

    if stats.total_chunks == 0 {
        println!("❌ No text could be extracted from the PDF. Please check the file.");
        return Ok(());
    }

    // Step 2: Build video with H.264 codec for better compression
    println!("\n🎬 Step 2: Building QR code video...");
    let pdf_name = Path::new(pdf_path)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    
    let video_path = format!("{}_memory.mp4", pdf_name);
    let index_path = format!("{}_memory.json", pdf_name);
    
    let encoding_stats = encoder
        .build_video_with_codec(&video_path, &index_path, Some(Codec::H264))
        .await?;
    
    println!("   • Video created: {} ({:.2} MB)", 
             video_path, 
             encoding_stats.video_stats.file_size_bytes as f64 / 1024.0 / 1024.0);
    println!("   • Duration: {:.1} seconds", encoding_stats.video_stats.duration_seconds);
    println!("   • Codec: {}", encoding_stats.video_stats.codec);
    println!("   • Encoding time: {:.2} seconds", encoding_stats.encoding_time_seconds);

    // Step 3: Test search functionality
    println!("\n🔍 Step 3: Testing search functionality...");
    let retriever = rust_mem_vid::MemvidRetriever::new(&video_path, &index_path).await?;
    
    // Perform some sample searches
    let sample_queries = vec![
        "introduction",
        "conclusion",
        "method",
        "result",
        "analysis",
    ];

    for query in sample_queries {
        println!("\n   Searching for: '{}'", query);
        let results = retriever.search_with_metadata(query, 3).await?;
        
        if results.is_empty() {
            println!("   No results found");
        } else {
            for (i, result) in results.iter().enumerate() {
                let preview = if result.text.len() > 100 {
                    format!("{}...", &result.text[..100])
                } else {
                    result.text.clone()
                };
                println!("   {}. [Score: {:.3}] {}", i + 1, result.similarity, preview);
            }
        }
    }

    // Step 4: Interactive chat
    println!("\n💬 Step 4: Starting interactive chat...");
    
    // Check for API keys
    let has_openai = std::env::var("OPENAI_API_KEY").is_ok();
    let has_anthropic = std::env::var("ANTHROPIC_API_KEY").is_ok();
    
    if !has_openai && !has_anthropic {
        println!("⚠️  No API keys found. Please set one of the following environment variables:");
        println!("   • OPENAI_API_KEY for OpenAI GPT models");
        println!("   • ANTHROPIC_API_KEY for Anthropic Claude models");
        println!("\nYou can still search the document, but chat functionality won't work.");
        
        // Offer search-only mode
        println!("\n🔍 Search-only mode available. Enter queries to search the PDF:");
        loop {
            print!("Search query (or 'quit' to exit): ");
            use std::io::{self, Write};
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.to_lowercase() == "quit" {
                break;
            }

            if input.is_empty() {
                continue;
            }

            let results = retriever.search_with_metadata(input, 5).await?;
            
            if results.is_empty() {
                println!("No results found for '{}'", input);
            } else {
                println!("\nResults for '{}':", input);
                for (i, result) in results.iter().enumerate() {
                    println!("\n{}. [Score: {:.3}]", i + 1, result.similarity);
                    println!("{}", result.text);
                }
            }
        }
        
        return Ok(());
    }

    // Initialize chat
    let mut chat = MemvidChat::new(&video_path, &index_path).await?;
    
    // Set provider based on available API keys
    if has_anthropic {
        chat.set_provider("anthropic")?;
        println!("   Using Anthropic Claude");
    } else {
        chat.set_provider("openai")?;
        println!("   Using OpenAI GPT");
    }

    println!("\n💬 Chat with your PDF!");
    println!("Ask questions about the document content.");
    println!("Type 'quit' to exit, 'clear' to clear history, 'search <query>' for direct search.");
    println!();

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
            "" => continue,
            _ => {}
        }

        // Handle search commands
        if input.to_lowercase().starts_with("search ") {
            let query = &input[7..]; // Remove "search " prefix
            println!("🔍 Searching for: '{}'", query);
            
            let results = retriever.search_with_metadata(query, 5).await?;
            
            if results.is_empty() {
                println!("No results found.");
            } else {
                for (i, result) in results.iter().enumerate() {
                    println!("\n{}. [Score: {:.3}]", i + 1, result.similarity);
                    let preview = if result.text.len() > 200 {
                        format!("{}...", &result.text[..200])
                    } else {
                        result.text.clone()
                    };
                    println!("{}", preview);
                }
            }
            continue;
        }

        // Chat with LLM
        print!("🤖 Assistant: ");
        io::stdout().flush()?;

        match chat.chat(input).await {
            Ok(response) => {
                println!("{}\n", response);
            }
            Err(e) => {
                eprintln!("❌ Chat error: {}", e);
                println!("Please try again or check your API key.\n");
            }
        }
    }

    // Final statistics
    println!("\n📊 Final Statistics:");
    let chat_stats = chat.get_stats();
    println!("   • Total messages: {}", chat_stats.total_messages);
    println!("   • User messages: {}", chat_stats.user_messages);
    println!("   • Assistant messages: {}", chat_stats.assistant_messages);
    
    println!("\n✅ PDF chat session completed!");
    println!("   Files created:");
    println!("   • {} (video memory)", video_path);
    println!("   • {} (search index)", index_path);

    Ok(())
}
