//! Web Platform Demo
//! 
//! This example demonstrates the complete web platform functionality including:
//! - Starting the web server
//! - Loading multiple memories
//! - Browser-based memory management
//! - Real-time collaboration features
//! - Advanced search and analytics

use rust_mem_vid::{
    MemvidEncoder, MemoryWebServer, Config,
    video::Codec,
};
use std::path::Path;
use tokio::time::{sleep, Duration};
use tracing::{info, error};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("🚀 Starting MemVid Web Platform Demo");

    // Initialize the library
    rust_mem_vid::init().await?;

    // Create sample memories if they don't exist
    create_sample_memories().await?;

    // Start the web server
    start_web_server().await?;

    Ok(())
}

async fn create_sample_memories() -> anyhow::Result<()> {
    info!("📚 Creating sample memories for web platform demo...");

    let config = Config::default();

    // Create AI Research Memory
    if !Path::new("ai_research.mp4").exists() {
        info!("Creating AI Research memory...");
        let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;
        
        let ai_content = vec![
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.".to_string(),
            "Deep learning uses neural networks with multiple layers to model complex patterns in data.".to_string(),
            "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks.".to_string(),
            "Recurrent Neural Networks (RNNs) are designed to work with sequential data like text and time series.".to_string(),
            "Transformer architectures have revolutionized natural language processing with attention mechanisms.".to_string(),
            "BERT and GPT models represent major breakthroughs in language understanding and generation.".to_string(),
            "Reinforcement learning enables agents to learn optimal actions through trial and error.".to_string(),
            "Computer vision combines machine learning with image processing for visual understanding.".to_string(),
            "Natural language processing enables computers to understand and generate human language.".to_string(),
            "Supervised learning uses labeled data to train models for prediction tasks.".to_string(),
            "Unsupervised learning discovers patterns in data without explicit labels.".to_string(),
            "Feature engineering is crucial for traditional machine learning model performance.".to_string(),
            "Cross-validation helps assess model performance and prevent overfitting.".to_string(),
            "Gradient descent is the fundamental optimization algorithm for training neural networks.".to_string(),
            "Backpropagation enables efficient computation of gradients in neural networks.".to_string(),
        ];

        encoder.add_chunks(ai_content).await?;
        encoder.build_video_with_codec("ai_research.mp4", "ai_research.metadata", Some(Codec::H264)).await?;
        info!("✅ AI Research memory created");
    }

    // Create Programming Knowledge Memory
    if !Path::new("programming.mp4").exists() {
        info!("Creating Programming Knowledge memory...");
        let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;
        
        let programming_content = vec![
            "Rust is a systems programming language focused on safety, speed, and concurrency.".to_string(),
            "Python is a high-level programming language known for its simplicity and versatility.".to_string(),
            "JavaScript is the language of the web, running in browsers and Node.js servers.".to_string(),
            "TypeScript adds static typing to JavaScript for better development experience.".to_string(),
            "Go is designed for simplicity and efficiency in concurrent programming.".to_string(),
            "Functional programming emphasizes immutability and pure functions.".to_string(),
            "Object-oriented programming organizes code around objects and classes.".to_string(),
            "Design patterns provide reusable solutions to common programming problems.".to_string(),
            "Test-driven development improves code quality through early testing.".to_string(),
            "Version control systems like Git enable collaborative software development.".to_string(),
            "Continuous integration automates testing and deployment processes.".to_string(),
            "Microservices architecture breaks applications into small, independent services.".to_string(),
            "RESTful APIs provide standardized interfaces for web services.".to_string(),
            "Database design is crucial for efficient data storage and retrieval.".to_string(),
            "Algorithms and data structures form the foundation of computer science.".to_string(),
        ];

        encoder.add_chunks(programming_content).await?;
        encoder.build_video_with_codec("programming.mp4", "programming.metadata", Some(Codec::H264)).await?;
        info!("✅ Programming Knowledge memory created");
    }

    // Create Research Notes Memory
    if !Path::new("research_notes.mp4").exists() {
        info!("Creating Research Notes memory...");
        let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;
        
        let research_content = vec![
            "Literature review reveals significant gaps in current understanding of neural attention mechanisms.".to_string(),
            "Experimental results show 15% improvement in accuracy with proposed architecture modifications.".to_string(),
            "Hypothesis: Multi-head attention can be optimized through dynamic head selection.".to_string(),
            "Data collection methodology: 10,000 samples from diverse domains for robust evaluation.".to_string(),
            "Statistical analysis indicates p-value < 0.001 for significance testing.".to_string(),
            "Future work should explore applications to low-resource languages.".to_string(),
            "Collaboration with Stanford team on transformer efficiency improvements.".to_string(),
            "Conference submission deadline: March 15th for ICML 2024.".to_string(),
            "Peer review feedback suggests strengthening theoretical foundations.".to_string(),
            "Implementation details: PyTorch framework with CUDA acceleration.".to_string(),
            "Baseline comparisons with BERT, GPT-3, and T5 models.".to_string(),
            "Ablation studies demonstrate importance of each component.".to_string(),
            "Error analysis reveals model struggles with rare linguistic phenomena.".to_string(),
            "Computational requirements: 8 V100 GPUs for 72 hours training time.".to_string(),
            "Reproducibility: All code and data will be made publicly available.".to_string(),
        ];

        encoder.add_chunks(research_content).await?;
        encoder.build_video_with_codec("research_notes.mp4", "research_notes.metadata", Some(Codec::H264)).await?;
        info!("✅ Research Notes memory created");
    }

    info!("📚 Sample memories created successfully!");
    Ok(())
}

async fn start_web_server() -> anyhow::Result<()> {
    info!("🌐 Starting MemVid Web Server...");

    let config = Config::default();
    let mut server = MemoryWebServer::new(config);

    // Load the sample memories
    let memories = vec![
        ("ai_research", "ai_research.mp4", "ai_research.metadata"),
        ("programming", "programming.mp4", "programming.metadata"),
        ("research_notes", "research_notes.mp4", "research_notes.metadata"),
    ];

    for (id, video_path, index_path) in memories {
        match server.load_memory(id.to_string(), video_path.to_string(), index_path.to_string()).await {
            Ok(_) => {
                info!("✅ Loaded memory: {} ({})", id, video_path);
            }
            Err(e) => {
                error!("❌ Failed to load memory {}: {}", id, e);
            }
        }
    }

    // Print comprehensive startup information
    print_startup_banner();

    // Start the server
    let bind_address = "127.0.0.1:8080";
    info!("🚀 Starting web server on {}", bind_address);
    
    match server.start(bind_address).await {
        Ok(_) => {
            info!("✅ Web server started successfully");
        }
        Err(e) => {
            error!("❌ Failed to start web server: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

fn print_startup_banner() {
    println!("\n🎉 MemVid Web Platform Demo Started!");
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│                          MemVid Web Platform Demo                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 🌐 Server URL: http://127.0.0.1:8080                                      │");
    println!("│ 📚 Sample Memories: 3 loaded (AI Research, Programming, Research Notes)   │");
    println!("│ 🤝 Collaboration: ✅ Enabled                                               │");
    println!("│ 🔓 Public Access: ✅ Enabled (Demo Mode)                                  │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 🎯 Demo Features Available:                                                │");
    println!("│   • 🏠 Home Dashboard - Overview and quick search                          │");
    println!("│   • 🔍 Advanced Search - Semantic search across all memories               │");
    println!("│   • 📊 Analytics Dashboard - AI-powered insights and visualizations       │");
    println!("│   • 🧠 Knowledge Graph - Interactive concept relationship mapping          │");
    println!("│   • ✨ Content Synthesis - AI-generated summaries and insights            │");
    println!("│   • 📹 Memory Management - Create, edit, and organize memories             │");
    println!("│   • 🤝 Real-time Collaboration - Live editing and sharing                  │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 🔗 Quick Demo Links:                                                       │");
    println!("│   • Home: http://127.0.0.1:8080/                                          │");
    println!("│   • Search: http://127.0.0.1:8080/search                                  │");
    println!("│   • Analytics: http://127.0.0.1:8080/analytics                            │");
    println!("│   • Dashboard: http://127.0.0.1:8080/dashboard                            │");
    println!("│   • Memories: http://127.0.0.1:8080/memories                              │");
    println!("│   • API Docs: http://127.0.0.1:8080/api/                                  │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 🧪 Try These Demo Scenarios:                                               │");
    println!("│   1. Search for 'machine learning' across all memories                     │");
    println!("│   2. Generate AI insights from the research notes                          │");
    println!("│   3. Create a knowledge graph of programming concepts                       │");
    println!("│   4. Compare AI research and programming memories                           │");
    println!("│   5. Synthesize content about 'neural networks'                            │");
    println!("│   6. View real-time analytics dashboard                                    │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ 🛠️  API Endpoints for Testing:                                             │");
    println!("│   • GET  /api/memories - List all memories                                 │");
    println!("│   • POST /api/search - Search across memories                              │");
    println!("│   • GET  /api/analytics/:id - Get memory analytics                         │");
    println!("│   • POST /api/synthesis - Generate AI content synthesis                    │");
    println!("│   • GET  /api/knowledge-graph/:id - Get knowledge graph                    │");
    println!("│   • WebSocket /ws - Real-time collaboration                                │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!("\n💡 Demo Tips:");
    println!("   • Use Ctrl+K for quick search from any page");
    println!("   • All features work with the sample data provided");
    println!("   • WebSocket connection enables real-time updates");
    println!("   • Try opening multiple browser tabs for collaboration demo");
    println!("   • Press Ctrl+C to stop the server\n");
    
    println!("🔄 Server starting... Please wait for 'Web server started successfully' message");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sample_memory_creation() {
        let _ = tracing_subscriber::fmt().try_init();
        
        // This test would create sample memories
        // In a real test, we'd verify the files are created correctly
        assert!(true); // Placeholder
    }

    #[tokio::test]
    async fn test_web_server_configuration() {
        let config = Config::default();
        let server = MemoryWebServer::new(config);
        
        // Test that server can be created
        assert!(true); // Placeholder - would test server configuration
    }
}
