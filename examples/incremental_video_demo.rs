//! Incremental video building demonstration
//! 
//! This example shows how to use true incremental video building features:
//! - Loading existing videos
//! - Appending new chunks to existing videos
//! - Merging multiple videos
//! - Creating incremental videos with only new content

use rust_mem_vid::{MemvidEncoder, Config};
use std::time::Instant;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔧 Incremental Video Building Demo");
    println!("===================================\n");

    // Create temporary directory for output
    let temp_dir = TempDir::new()?;

    // Demo 1: Create initial video
    println!("📹 Demo 1: Creating Initial Video");
    println!("----------------------------------");
    
    let initial_video = temp_dir.path().join("initial.mp4");
    let initial_index = temp_dir.path().join("initial.json");
    
    let initial_content = vec![
        "Chapter 1: Introduction to Rust programming language and its key features.".to_string(),
        "Chapter 2: Understanding ownership, borrowing, and lifetimes in Rust.".to_string(),
        "Chapter 3: Working with structs, enums, and pattern matching.".to_string(),
        "Chapter 4: Error handling with Result and Option types.".to_string(),
        "Chapter 5: Collections and iterators in Rust programming.".to_string(),
    ];

    let start_time = Instant::now();
    {
        let mut encoder = MemvidEncoder::new().await?;
        encoder.add_chunks(initial_content.clone()).await?;
        let _stats = encoder.build_video(
            initial_video.to_str().unwrap(),
            initial_index.to_str().unwrap()
        ).await?;
    }
    let initial_time = start_time.elapsed();
    
    println!("✅ Created initial video with {} chunks in {:.2}s", 
        initial_content.len(), initial_time.as_secs_f64());

    // Demo 2: Load existing video and inspect content
    println!("\n🔍 Demo 2: Loading Existing Video");
    println!("----------------------------------");
    
    let start_time = Instant::now();
    let loaded_encoder = MemvidEncoder::load_existing(
        initial_video.to_str().unwrap(),
        initial_index.to_str().unwrap()
    ).await?;
    let load_time = start_time.elapsed();
    
    println!("✅ Loaded existing video with {} chunks in {:.2}s", 
        loaded_encoder.len(), load_time.as_secs_f64());
    
    // Show loaded content
    println!("📄 Loaded content:");
    for (i, chunk) in loaded_encoder.chunks().iter().enumerate() {
        let preview = if chunk.content.len() > 60 {
            format!("{}...", &chunk.content[..60])
        } else {
            chunk.content.clone()
        };
        println!("   {}. {}", i + 1, preview);
    }

    // Demo 3: Append new content to existing video
    println!("\n➕ Demo 3: Appending New Content");
    println!("----------------------------------");
    
    let new_content = vec![
        "Chapter 6: Concurrency and parallelism with threads and async/await.".to_string(),
        "Chapter 7: Building web applications with popular Rust frameworks.".to_string(),
        "Chapter 8: Testing strategies and best practices in Rust.".to_string(),
    ];

    let start_time = Instant::now();
    {
        let mut encoder = MemvidEncoder::load_existing(
            initial_video.to_str().unwrap(),
            initial_index.to_str().unwrap()
        ).await?;
        
        // Add new chunks
        encoder.add_chunks(new_content.clone()).await?;
        
        // Rebuild video with all content
        let _stats = encoder.build_video(
            initial_video.to_str().unwrap(),
            initial_index.to_str().unwrap()
        ).await?;
    }
    let append_time = start_time.elapsed();
    
    println!("✅ Appended {} new chunks in {:.2}s", 
        new_content.len(), append_time.as_secs_f64());

    // Verify the appended content
    let updated_encoder = MemvidEncoder::load_existing(
        initial_video.to_str().unwrap(),
        initial_index.to_str().unwrap()
    ).await?;
    
    println!("📊 Updated video now has {} chunks (was {})", 
        updated_encoder.len(), initial_content.len());

    // Demo 4: Create a second video for merging
    println!("\n📹 Demo 4: Creating Second Video for Merging");
    println!("---------------------------------------------");
    
    let second_video = temp_dir.path().join("second.mp4");
    let second_index = temp_dir.path().join("second.json");
    
    let second_content = vec![
        "Advanced Topic 1: Unsafe Rust and FFI (Foreign Function Interface).".to_string(),
        "Advanced Topic 2: Macros and metaprogramming in Rust.".to_string(),
        "Advanced Topic 3: Performance optimization and profiling.".to_string(),
        "Advanced Topic 4: Embedded programming with Rust.".to_string(),
    ];

    {
        let mut encoder = MemvidEncoder::new().await?;
        encoder.add_chunks(second_content.clone()).await?;
        let _stats = encoder.build_video(
            second_video.to_str().unwrap(),
            second_index.to_str().unwrap()
        ).await?;
    }
    
    println!("✅ Created second video with {} chunks", second_content.len());

    // Demo 5: Merge multiple videos
    println!("\n🔗 Demo 5: Merging Multiple Videos");
    println!("-----------------------------------");
    
    let merged_video = temp_dir.path().join("merged.mp4");
    let merged_index = temp_dir.path().join("merged.json");
    
    let start_time = Instant::now();
    let merge_stats = MemvidEncoder::merge_videos(
        &[
            initial_video.to_str().unwrap(),
            second_video.to_str().unwrap(),
        ],
        &[
            initial_index.to_str().unwrap(),
            second_index.to_str().unwrap(),
        ],
        merged_video.to_str().unwrap(),
        merged_index.to_str().unwrap(),
        Config::default(),
    ).await?;
    let merge_time = start_time.elapsed();
    
    println!("✅ Merged videos in {:.2}s", merge_time.as_secs_f64());
    println!("📊 Merged video statistics:");
    println!("   • Total chunks: {}", merge_stats.total_chunks);
    println!("   • Total frames: {}", merge_stats.total_frames);
    println!("   • File size: {:.2} MB", 
        merge_stats.video_file_size_bytes as f64 / 1024.0 / 1024.0);

    // Demo 6: Create incremental video with only new content
    println!("\n⚡ Demo 6: Creating Incremental Video");
    println!("-------------------------------------");
    
    let incremental_video = temp_dir.path().join("incremental.mp4");
    let incremental_index = temp_dir.path().join("incremental.json");
    
    let incremental_content = vec![
        "Bonus Chapter: Future of Rust and upcoming features.".to_string(),
        "Appendix A: Common Rust patterns and idioms.".to_string(),
        "Appendix B: Troubleshooting guide and FAQ.".to_string(),
    ];

    let start_time = Instant::now();
    {
        let mut encoder = MemvidEncoder::new().await?;
        let _stats = encoder.create_incremental_video(
            incremental_content.clone(),
            incremental_video.to_str().unwrap(),
            incremental_index.to_str().unwrap(),
        ).await?;
    }
    let incremental_time = start_time.elapsed();
    
    println!("✅ Created incremental video with {} chunks in {:.2}s", 
        incremental_content.len(), incremental_time.as_secs_f64());

    // Summary
    println!("\n📊 Performance Summary");
    println!("======================");
    println!("Initial video creation:  {:.2}s ({} chunks)", 
        initial_time.as_secs_f64(), initial_content.len());
    println!("Loading existing video:  {:.2}s", load_time.as_secs_f64());
    println!("Appending new content:   {:.2}s ({} chunks)", 
        append_time.as_secs_f64(), new_content.len());
    println!("Merging videos:          {:.2}s ({} total chunks)", 
        merge_time.as_secs_f64(), merge_stats.total_chunks);
    println!("Incremental video:       {:.2}s ({} chunks)", 
        incremental_time.as_secs_f64(), incremental_content.len());

    println!("\n💡 Use Cases for Incremental Building:");
    println!("=====================================");
    println!("🔄 Append Mode:");
    println!("   • Adding new documents to existing knowledge base");
    println!("   • Incremental updates to documentation");
    println!("   • Building knowledge over time");
    println!();
    println!("🔗 Merge Mode:");
    println!("   • Combining separate knowledge domains");
    println!("   • Consolidating team knowledge bases");
    println!("   • Creating comprehensive archives");
    println!();
    println!("⚡ Incremental Mode:");
    println!("   • Processing only new content efficiently");
    println!("   • Creating delta updates");
    println!("   • Distributing content updates");

    println!("\n🖥️  CLI Usage Examples:");
    println!("=======================");
    println!("# Append new content to existing video");
    println!("memvid append --video existing.mp4 --index existing.json --files new_docs/");
    println!();
    println!("# Merge multiple videos");
    println!("memvid merge --output combined.mp4 --index combined.json --videos video1.mp4,index1.json,video2.mp4,index2.json");
    println!();
    println!("# Load existing video programmatically");
    println!("let encoder = MemvidEncoder::load_existing(\"video.mp4\", \"index.json\").await?;");

    Ok(())
}
