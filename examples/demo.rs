use rust_mem_vid::{MemvidEncoder, Config};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🎬 Rust MemVid Demo");
    println!("==================");
    
    // Read the demo content
    let content = std::fs::read_to_string("demo_content.txt")?;
    println!("📖 Read {} characters from demo_content.txt", content.len());
    
    // Create encoder
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;

    // Process the text
    let start = Instant::now();
    encoder.add_text(&content, Some("demo_content.txt".to_string())).await?;
    let chunk_count = encoder.get_stats().total_chunks;
    println!("✂️  Split into {} chunks", chunk_count);

    // Build the video
    let video_path = "demo_video.mp4";
    let index_path = "demo_index.json";

    let stats = encoder.build_video(video_path, index_path).await?;
    let duration = start.elapsed();

    println!("🎥 Created video: {}", video_path);
    println!("   • Chunks: {}", stats.total_chunks);
    println!("   • Processing time: {:.2}s", duration.as_secs_f64());

    // Test extraction
    println!("\n🔍 Testing frame extraction:");
    for frame in 0..std::cmp::min(3, stats.total_chunks as u32) {
        let qr_processor = rust_mem_vid::qr::QrProcessor::default();
        let video_decoder = rust_mem_vid::video::VideoDecoder::new(config.clone());
        
        let image = video_decoder.extract_frame(video_path, frame).await?;
        let decoded = qr_processor.decode_qr(&image)?;
        
        // Parse the JSON to get just the text
        let chunk_data: serde_json::Value = serde_json::from_str(&decoded)?;
        let text = chunk_data["text"].as_str().unwrap_or("Unknown");
        let preview = if text.len() > 60 {
            format!("{}...", &text[..60])
        } else {
            text.to_string()
        };
        
        println!("   Frame {}: {}", frame, preview);
    }
    
    println!("\n✅ Demo completed successfully!");
    println!("Files created:");
    println!("   • {} (video)", video_path);
    println!("   • {}.metadata (index metadata)", index_path.trim_end_matches(".json"));
    println!("   • {}.vector (vector embeddings)", index_path.trim_end_matches(".json"));
    println!("   • {}_frames/ (QR code frames)", video_path.trim_end_matches(".mp4"));
    
    Ok(())
}
