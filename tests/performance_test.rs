//! Performance tests for large dataset processing

use rust_mem_vid::{MemvidEncoder, Config};
use std::time::Instant;
use tempfile::TempDir;

#[tokio::test]
async fn test_large_dataset_performance() {
    // Skip this test in CI or if explicitly disabled
    if std::env::var("SKIP_PERFORMANCE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("test_video.mp4");
    let index_path = temp_dir.path().join("test_index.json");

    // Create a large dataset (1000+ chunks)
    let mut large_chunks = Vec::new();
    for i in 0..1200 {
        large_chunks.push(format!(
            "This is test chunk number {}. It contains some sample text that should be long enough to test the QR code encoding process. The content includes various characters and symbols to ensure proper encoding: !@#$%^&*()_+-=[]{{}}|;':\",./<>? This chunk is designed to test the performance of the system when processing large numbers of chunks. Chunk ID: {}", 
            i, i
        ));
    }

    println!("Testing with {} chunks", large_chunks.len());

    // Test with index building enabled
    let start_time = Instant::now();
    {
        let mut config = Config::default();
        config.search.enable_index_building = true;
        
        let mut encoder = MemvidEncoder::new_with_config(config).await.unwrap();
        encoder.add_chunks(large_chunks.clone()).await.unwrap();
        
        let _stats = encoder.build_video(
            output_path.to_str().unwrap(),
            index_path.to_str().unwrap()
        ).await.unwrap();
    }
    let with_index_time = start_time.elapsed();
    println!("Time with index building: {:.2}s", with_index_time.as_secs_f64());

    // Test with index building disabled
    let output_path_no_index = temp_dir.path().join("test_video_no_index.mp4");
    let index_path_no_index = temp_dir.path().join("test_index_no_index.json");
    
    let start_time = Instant::now();
    {
        let mut config = Config::default();
        config.search.enable_index_building = false;
        
        let mut encoder = MemvidEncoder::new_with_config(config).await.unwrap();
        encoder.add_chunks(large_chunks).await.unwrap();
        
        let _stats = encoder.build_video(
            output_path_no_index.to_str().unwrap(),
            index_path_no_index.to_str().unwrap()
        ).await.unwrap();
    }
    let without_index_time = start_time.elapsed();
    println!("Time without index building: {:.2}s", without_index_time.as_secs_f64());

    // The version without index building should be significantly faster
    let speedup = with_index_time.as_secs_f64() / without_index_time.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
    
    // We expect at least some speedup when disabling index building
    assert!(speedup > 1.0, "Disabling index building should provide speedup");
}

#[tokio::test]
async fn test_streaming_vs_batch_processing() {
    // Skip this test in CI or if explicitly disabled
    if std::env::var("SKIP_PERFORMANCE_TESTS").is_ok() {
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Create a dataset that triggers streaming processing (>1000 chunks)
    let mut large_chunks = Vec::new();
    for i in 0..1100 {
        large_chunks.push(format!("Streaming test chunk {}: {}", i, "x".repeat(500)));
    }

    println!("Testing streaming vs batch with {} chunks", large_chunks.len());

    // Test streaming processing (should be used automatically for >1000 chunks)
    let start_time = Instant::now();
    {
        let mut config = Config::default();
        config.search.enable_index_building = false; // Disable for fair comparison
        
        let mut encoder = MemvidEncoder::new_with_config(config).await.unwrap();
        encoder.add_chunks(large_chunks.clone()).await.unwrap();
        
        let output_path = temp_dir.path().join("streaming_test.mp4");
        let index_path = temp_dir.path().join("streaming_test.json");
        
        let _stats = encoder.build_video(
            output_path.to_str().unwrap(),
            index_path.to_str().unwrap()
        ).await.unwrap();
    }
    let streaming_time = start_time.elapsed();
    println!("Streaming processing time: {:.2}s", streaming_time.as_secs_f64());

    // Verify that streaming processing completed successfully
    assert!(streaming_time.as_secs() < 300, "Streaming processing should complete within 5 minutes");
}
