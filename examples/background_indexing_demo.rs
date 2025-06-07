//! Background indexing demonstration
//! 
//! This example shows how to use background indexing for fast video creation
//! with asynchronous index building.

use rust_mem_vid::{MemvidEncoder, Config, submit_background_indexing, get_indexing_status, IndexingStatus};
use std::time::Instant;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Background Indexing Demo");
    println!("============================\n");

    // Create temporary directory for output
    let temp_dir = TempDir::new()?;
    let video_path = temp_dir.path().join("demo_video.mp4");
    let index_path = temp_dir.path().join("demo_index.json");

    // Create a large dataset to demonstrate the benefits
    let mut large_dataset = Vec::new();
    for i in 0..1200 {
        large_dataset.push(format!(
            "Document chunk #{}: This is a substantial piece of text that contains important information about topic {}. \
            It includes various details, explanations, and context that would be valuable for search and retrieval. \
            The content covers multiple aspects including technical details, historical context, and practical applications. \
            This chunk is designed to be large enough to demonstrate the performance benefits of background indexing \
            when processing thousands of similar chunks. Chunk identifier: {}", 
            i, i % 10, i
        ));
    }

    println!("📊 Dataset: {} chunks ({} characters total)", 
        large_dataset.len(), 
        large_dataset.iter().map(|s| s.len()).sum::<usize>()
    );

    // Demo 1: Traditional immediate indexing
    println!("\n🔄 Demo 1: Traditional Immediate Indexing");
    println!("------------------------------------------");
    
    let start_time = Instant::now();
    {
        let mut config = Config::default();
        config.search.enable_index_building = true;
        config.search.enable_background_indexing = false;
        
        let mut encoder = MemvidEncoder::new_with_config(config).await?;
        encoder.add_chunks(large_dataset.clone()).await?;
        
        let _stats = encoder.build_video(
            video_path.to_str().unwrap(),
            index_path.to_str().unwrap()
        ).await?;
    }
    let immediate_time = start_time.elapsed();
    println!("✅ Completed in {:.2}s (video + index)", immediate_time.as_secs_f64());

    // Demo 2: Background indexing
    println!("\n🚀 Demo 2: Background Indexing");
    println!("-------------------------------");
    
    let video_path_bg = temp_dir.path().join("demo_video_bg.mp4");
    let index_path_bg = temp_dir.path().join("demo_index_bg.json");
    
    let start_time = Instant::now();
    let job_id = {
        let mut config = Config::default();
        config.search.enable_index_building = false;
        config.search.enable_background_indexing = true;
        
        let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;
        encoder.add_chunks(large_dataset.clone()).await?;
        
        // This should return quickly
        let _stats = encoder.build_video(
            video_path_bg.to_str().unwrap(),
            index_path_bg.to_str().unwrap()
        ).await?;
        
        // Submit background indexing job
        submit_background_indexing(
            encoder.chunks().to_vec(),
            index_path_bg.clone(),
            config
        ).await?
    };
    let video_creation_time = start_time.elapsed();
    
    println!("✅ Video created in {:.2}s", video_creation_time.as_secs_f64());
    println!("🔄 Background indexing job: {}", job_id);
    println!("⚡ Speedup: {:.1}x faster video creation", 
        immediate_time.as_secs_f64() / video_creation_time.as_secs_f64());

    // Monitor background indexing progress
    println!("\n📈 Monitoring Background Indexing Progress");
    println!("-------------------------------------------");
    
    let index_start_time = Instant::now();
    let mut last_progress = 0.0;
    
    loop {
        match get_indexing_status(&job_id).await {
            Some(IndexingStatus::Queued) => {
                println!("⏳ Status: Queued");
            }
            Some(IndexingStatus::InProgress { progress }) => {
                if progress - last_progress >= 10.0 || progress == 100.0 {
                    println!("🔄 Status: In Progress ({:.1}%)", progress);
                    last_progress = progress;
                }
            }
            Some(IndexingStatus::Completed { duration_seconds }) => {
                println!("✅ Status: Completed in {:.2}s", duration_seconds);
                break;
            }
            Some(IndexingStatus::Failed { error }) => {
                println!("❌ Status: Failed - {}", error);
                break;
            }
            None => {
                println!("❓ Job not found");
                break;
            }
        }
        
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    
    let total_background_time = index_start_time.elapsed();
    
    // Demo 3: No indexing (fastest)
    println!("\n⚡ Demo 3: No Indexing (Fastest)");
    println!("--------------------------------");
    
    let video_path_fast = temp_dir.path().join("demo_video_fast.mp4");
    let index_path_fast = temp_dir.path().join("demo_index_fast.json");
    
    let start_time = Instant::now();
    {
        let mut config = Config::default();
        config.search.enable_index_building = false;
        config.search.enable_background_indexing = false;
        
        let mut encoder = MemvidEncoder::new_with_config(config).await?;
        encoder.add_chunks(large_dataset).await?;
        
        let _stats = encoder.build_video(
            video_path_fast.to_str().unwrap(),
            index_path_fast.to_str().unwrap()
        ).await?;
    }
    let no_index_time = start_time.elapsed();
    println!("✅ Completed in {:.2}s (video only)", no_index_time.as_secs_f64());

    // Summary
    println!("\n📊 Performance Summary");
    println!("======================");
    println!("Immediate Indexing:  {:.2}s (video + index together)", immediate_time.as_secs_f64());
    println!("Background Indexing: {:.2}s (video) + {:.2}s (index async)", 
        video_creation_time.as_secs_f64(), total_background_time.as_secs_f64());
    println!("No Indexing:         {:.2}s (video only)", no_index_time.as_secs_f64());
    println!();
    println!("🚀 Video Creation Speedup:");
    println!("  Background vs Immediate: {:.1}x faster", 
        immediate_time.as_secs_f64() / video_creation_time.as_secs_f64());
    println!("  No Index vs Immediate:   {:.1}x faster", 
        immediate_time.as_secs_f64() / no_index_time.as_secs_f64());
    println!();
    println!("💡 Use Cases:");
    println!("  • Immediate Indexing: Small datasets, need search immediately");
    println!("  • Background Indexing: Large datasets, need video file quickly");
    println!("  • No Indexing: Fastest processing, no search needed");

    // Demonstrate CLI usage
    println!("\n🖥️  CLI Usage Examples");
    println!("======================");
    println!("# Background indexing (recommended for large datasets)");
    println!("memvid encode --background-index --output video.mp4 --index index.json dataset/");
    println!();
    println!("# Check indexing status");
    println!("memvid index-status <job_id>");
    println!();
    println!("# List all background jobs");
    println!("memvid index-jobs");
    println!();
    println!("# Wait for job completion");
    println!("memvid index-wait <job_id> --timeout 300");
    println!();
    println!("# Fast processing without search");
    println!("memvid encode --no-index --output video.mp4 --index index.json dataset/");

    Ok(())
}
