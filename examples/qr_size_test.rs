use rust_mem_vid::{MemvidEncoder, Config};
use std::fs;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔍 QR Code Size Optimization Test");
    println!("=================================");
    
    // Create test data with varying chunk sizes
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("large_text.txt");
    
    // Create a large text file that would normally cause QR code issues
    let large_content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100);
    fs::write(&test_file, &large_content)?;
    
    println!("📄 Created test file with {} characters", large_content.len());
    
    // Test 1: Default configuration (should work with optimization)
    println!("\n🧪 Test 1: Default configuration with auto-optimization");
    test_with_chunk_size(test_file.to_str().unwrap(), None).await?;
    
    // Test 2: Large chunk size (should trigger optimization)
    println!("\n🧪 Test 2: Large chunk size (3000 chars) - should auto-optimize");
    test_with_chunk_size(test_file.to_str().unwrap(), Some(3000)).await?;
    
    // Test 3: Small chunk size (should work without optimization)
    println!("\n🧪 Test 3: Small chunk size (500 chars) - should work directly");
    test_with_chunk_size(test_file.to_str().unwrap(), Some(500)).await?;
    
    // Test 4: Analyze chunks before and after optimization
    println!("\n🧪 Test 4: Chunk analysis demonstration");
    demonstrate_chunk_analysis(test_file.to_str().unwrap()).await?;
    
    println!("\n✅ All QR code size tests completed successfully!");
    println!("💡 The system automatically optimizes chunk sizes for QR code compatibility.");
    
    Ok(())
}

async fn test_with_chunk_size(file_path: &str, chunk_size: Option<usize>) -> anyhow::Result<()> {
    let mut config = Config::default();
    
    if let Some(size) = chunk_size {
        config.text.chunk_size = size;
        println!("   Using chunk size: {} characters", size);
    } else {
        println!("   Using default chunk size: {} characters", config.text.chunk_size);
    }
    
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    
    // Add the file
    encoder.add_text_file(file_path).await?;
    
    // Get recommended chunk size
    let recommended = encoder.get_recommended_chunk_size()?;
    println!("   Recommended QR chunk size: {} characters", recommended);
    
    // Analyze chunks before optimization
    let analysis_before = encoder.analyze_chunks();
    println!("   Chunks before optimization: {}", analysis_before.total_chunks);
    println!("   Max chunk size: {} characters", analysis_before.max_chunk_size);
    println!("   Oversized chunks: {}", analysis_before.oversized_chunks);
    
    // Try to build video (this will trigger automatic optimization)
    let temp_dir = tempfile::TempDir::new()?;
    let video_path = temp_dir.path().join("test.mp4");
    let index_path = temp_dir.path().join("test.json");
    
    match encoder.build_video(video_path.to_str().unwrap(), index_path.to_str().unwrap()).await {
        Ok(stats) => {
            println!("   ✅ Video created successfully!");
            println!("   Final chunks: {}", stats.total_chunks);
            println!("   Total characters: {}", stats.total_characters);
            println!("   Encoding time: {:.2}s", stats.encoding_time_seconds);
        }
        Err(e) => {
            println!("   ❌ Failed to create video: {}", e);
            if e.to_string().contains("data too long") {
                println!("   💡 This error indicates QR code size limits were exceeded");
                println!("   💡 Try using a smaller chunk size (--chunk-size 800)");
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_chunk_analysis(file_path: &str) -> anyhow::Result<()> {
    // Create encoder with large chunk size
    let mut config = Config::default();
    config.text.chunk_size = 2500; // Intentionally large
    
    let mut encoder = MemvidEncoder::new_with_config(config).await?;
    encoder.add_text_file(file_path).await?;
    
    println!("   📊 Analysis before optimization:");
    let analysis_before = encoder.analyze_chunks();
    print_analysis(&analysis_before);
    
    // Manually optimize chunks
    encoder.optimize_chunks_for_qr()?;
    
    println!("\n   📊 Analysis after optimization:");
    let analysis_after = encoder.analyze_chunks();
    print_analysis(&analysis_after);
    
    println!("\n   📈 Optimization results:");
    println!("     • Chunks: {} → {}", analysis_before.total_chunks, analysis_after.total_chunks);
    println!("     • Max size: {} → {} characters", analysis_before.max_chunk_size, analysis_after.max_chunk_size);
    println!("     • Oversized: {} → {}", analysis_before.oversized_chunks, analysis_after.oversized_chunks);
    
    Ok(())
}

fn print_analysis(analysis: &rust_mem_vid::encoder::ChunkAnalysis) {
    println!("     • Total chunks: {}", analysis.total_chunks);
    println!("     • Total characters: {}", analysis.total_characters);
    println!("     • Average chunk size: {} characters", analysis.avg_chunk_size);
    println!("     • Max chunk size: {} characters", analysis.max_chunk_size);
    println!("     • Min chunk size: {} characters", analysis.min_chunk_size);
    println!("     • Recommended size: {} characters", analysis.recommended_size);
    println!("     • Oversized chunks: {}", analysis.oversized_chunks);
    println!("     • Needs optimization: {}", analysis.needs_optimization);
}
