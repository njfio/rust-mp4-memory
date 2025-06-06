use rust_mem_vid::{MemvidEncoder, Config};
use rust_mem_vid::memory_diff::MemoryDiffEngine;
use rust_mem_vid::multi_memory::MultiMemoryEngine;
use rust_mem_vid::temporal_analysis::TemporalAnalysisEngine;
use std::fs;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🕰️  Temporal Analysis and Memory Comparison Demo");
    println!("===============================================");
    println!("This demo showcases:");
    println!("• Memory comparison and diff analysis");
    println!("• Multi-memory search across different versions");
    println!("• Temporal analysis of memory evolution");
    println!("• Cross-memory correlation detection");
    println!();
    
    // Create test environment with evolving content
    let temp_dir = create_evolving_content().await?;
    let demo_path = temp_dir.path();
    
    println!("📁 Created evolving content at: {}", demo_path.display());
    
    // Demo 1: Create multiple memory snapshots
    println!("\n🎬 Demo 1: Creating memory snapshots over time");
    let snapshots = create_memory_snapshots(demo_path).await?;
    
    // Demo 2: Compare memories with diff analysis
    println!("\n🔍 Demo 2: Memory comparison and diff analysis");
    demo_memory_diff(&snapshots).await?;
    
    // Demo 3: Multi-memory search
    println!("\n🔎 Demo 3: Multi-memory search across versions");
    demo_multi_memory_search(&snapshots).await?;
    
    // Demo 4: Temporal analysis
    println!("\n📈 Demo 4: Temporal analysis of memory evolution");
    demo_temporal_analysis(&snapshots).await?;
    
    println!("\n✅ All temporal analysis demos completed!");
    println!("🎉 New capabilities demonstrated:");
    println!("   • Memory comparison with detailed diff analysis");
    println!("   • Cross-memory search with correlation detection");
    println!("   • Temporal evolution tracking and trend analysis");
    println!("   • Knowledge gap identification and recommendations");
    
    Ok(())
}

async fn create_evolving_content() -> anyhow::Result<TempDir> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create directories for different versions
    fs::create_dir_all(base_path.join("v1"))?;
    fs::create_dir_all(base_path.join("v2"))?;
    fs::create_dir_all(base_path.join("v3"))?;
    
    // Version 1: Initial documentation
    fs::write(base_path.join("v1/project_overview.md"), r#"
# Project Overview

This is the initial version of our project documentation.

## Goals
- Build a memory system for storing and retrieving information
- Support text-based content initially
- Create a simple search interface

## Architecture
- Basic text processing
- Simple storage mechanism
- Linear search functionality

## Status
- Project just started
- Basic concepts defined
- Implementation pending
"#)?;
    
    fs::write(base_path.join("v1/api_docs.md"), r#"
# API Documentation v1.0

## Basic Functions

### store_text(content)
Stores text content in the memory system.

### search_text(query)
Searches for text content using simple string matching.

### list_content()
Lists all stored content.

## Examples

```
store_text("Hello world")
search_text("hello")
```
"#)?;
    
    // Version 2: Enhanced with new features
    fs::write(base_path.join("v2/project_overview.md"), r#"
# Project Overview

This is version 2 of our advanced memory system project.

## Goals
- Build a sophisticated memory system for storing and retrieving information
- Support multiple content types (text, documents, code)
- Create an intelligent search interface with semantic capabilities
- Add video encoding for persistent storage

## Architecture
- Advanced text processing with chunking
- QR code video encoding for storage
- Semantic search with embeddings
- Multi-format content support

## New Features
- PDF document processing
- Code file analysis
- Semantic search capabilities
- Video-based memory storage

## Status
- Core functionality implemented
- Multiple content types supported
- Search working with embeddings
- Ready for production use
"#)?;
    
    fs::write(base_path.join("v2/api_docs.md"), r#"
# API Documentation v2.0

## Enhanced Functions

### encode_memory(content, output_video)
Encodes content into a QR code video for persistent storage.

### search_semantic(query, top_k)
Performs semantic search using embeddings.

### add_document(file_path)
Adds documents (PDF, text, code) to the memory system.

### chat_interface(query)
Interactive chat interface for querying memories.

## New Content Types
- PDF documents
- Source code files
- Structured data (CSV, JSON)
- Log files

## Examples

```
encode_memory("Complex content", "memory.mp4")
search_semantic("machine learning", 5)
add_document("research_paper.pdf")
```
"#)?;
    
    fs::write(base_path.join("v2/advanced_features.md"), r#"
# Advanced Features

## Semantic Search
Our system now supports semantic search using state-of-the-art embeddings.

## Multi-Format Support
- PDF documents with text extraction
- Source code with syntax analysis
- Structured data with schema detection

## Video Encoding
Content is encoded into QR code videos for:
- Persistent storage
- Easy sharing
- Visual representation of data

## Performance
- Fast retrieval with vector search
- Efficient chunking strategies
- Optimized encoding pipeline
"#)?;
    
    // Version 3: Full-featured with temporal analysis
    fs::write(base_path.join("v3/project_overview.md"), r#"
# Project Overview

This is version 3 of our comprehensive memory system with temporal analysis.

## Goals
- Build a complete memory ecosystem for information management
- Support all content types with intelligent processing
- Provide temporal analysis and memory evolution tracking
- Enable cross-memory comparison and correlation analysis
- Create a unified interface for memory management

## Architecture
- Sophisticated content processing pipeline
- Multi-memory management system
- Temporal analysis engine
- Cross-memory correlation detection
- Advanced diff and comparison tools

## Revolutionary Features
- Memory comparison and diff analysis
- Multi-memory search across different versions
- Temporal evolution tracking
- Knowledge gap identification
- Automated memory optimization

## Status
- Full ecosystem implemented
- Temporal analysis working
- Cross-memory features complete
- Production-ready with advanced analytics
"#)?;
    
    fs::write(base_path.join("v3/api_docs.md"), r#"
# API Documentation v3.0

## Temporal Analysis Functions

### compare_memories(old_memory, new_memory)
Compares two memory videos and generates detailed diff analysis.

### multi_memory_search(query, memories)
Searches across multiple memory videos simultaneously.

### analyze_timeline(snapshots)
Analyzes memory evolution over time and identifies trends.

### find_correlations(memories)
Detects correlations and relationships between different memories.

## Advanced Features
- Memory diff with semantic analysis
- Cross-memory correlation detection
- Temporal trend analysis
- Knowledge gap identification
- Automated optimization recommendations

## Examples

```
diff = compare_memories("old.mp4", "new.mp4")
results = multi_memory_search("AI research", [mem1, mem2, mem3])
timeline = analyze_timeline(snapshots)
correlations = find_correlations([memory_a, memory_b])
```
"#)?;
    
    fs::write(base_path.join("v3/temporal_features.md"), r#"
# Temporal Analysis Features

## Memory Evolution Tracking
Track how your knowledge base evolves over time with detailed analytics.

## Diff Analysis
- Chunk-level comparison between memory versions
- Semantic change detection
- Content growth and reduction analysis
- Modification type classification

## Multi-Memory Search
- Search across multiple memory videos simultaneously
- Cross-reference information between different snapshots
- Find correlations across time periods
- Temporal context for search results

## Timeline Analysis
- Growth trend identification
- Activity period detection
- Content evolution patterns
- Knowledge gap analysis

## Correlation Detection
- Complementary information identification
- Contradiction detection
- Evolutionary relationship tracking
- Redundancy analysis

## Use Cases
- Research project evolution tracking
- Knowledge base maintenance
- Content quality assessment
- Information gap identification
- Collaborative knowledge building
"#)?;
    
    Ok(temp_dir)
}

async fn create_memory_snapshots(demo_path: &std::path::Path) -> anyhow::Result<Vec<(String, String)>> {
    let mut snapshots = Vec::new();
    
    for version in ["v1", "v2", "v3"] {
        println!("   Creating memory snapshot for {}", version);
        
        let config = Config::default();
        let mut encoder = MemvidEncoder::new_with_config(config).await?;
        
        // Add all files from this version
        let version_path = demo_path.join(version);
        encoder.add_directory(&version_path.to_string_lossy()).await?;
        
        // Build video
        let video_path = demo_path.join(format!("{}_memory.mp4", version));
        let index_path = demo_path.join(format!("{}_memory", version)); // IndexManager adds .metadata extension

        let stats = encoder.build_video(
            video_path.to_str().unwrap(),
            index_path.to_str().unwrap()
        ).await?;
        
        println!("     ✅ {} chunks encoded in {:.2}s", 
                 stats.total_chunks, stats.encoding_time_seconds);
        
        snapshots.push((
            video_path.to_str().unwrap().to_string(),
            format!("{}.metadata", index_path.to_str().unwrap()) // IndexManager saves with .metadata extension
        ));
    }
    
    Ok(snapshots)
}

async fn demo_memory_diff(snapshots: &[(String, String)]) -> anyhow::Result<()> {
    let config = Config::default();
    let diff_engine = MemoryDiffEngine::new(config);
    
    // Compare v1 to v2
    println!("   Comparing v1 → v2:");
    let diff_v1_v2 = diff_engine.compare_memories(
        &snapshots[0].0, &snapshots[0].1,
        &snapshots[1].0, &snapshots[1].1
    ).await?;
    
    println!("     📊 Changes: +{} -{} ~{} chunks", 
             diff_v1_v2.summary.added_count,
             diff_v1_v2.summary.removed_count,
             diff_v1_v2.summary.modified_count);
    println!("     📈 Growth: {:.1}x, Similarity: {:.1}%", 
             diff_v1_v2.summary.content_growth_ratio,
             diff_v1_v2.summary.similarity_score * 100.0);
    
    // Compare v2 to v3
    println!("   Comparing v2 → v3:");
    let diff_v2_v3 = diff_engine.compare_memories(
        &snapshots[1].0, &snapshots[1].1,
        &snapshots[2].0, &snapshots[2].1
    ).await?;
    
    println!("     📊 Changes: +{} -{} ~{} chunks", 
             diff_v2_v3.summary.added_count,
             diff_v2_v3.summary.removed_count,
             diff_v2_v3.summary.modified_count);
    println!("     📈 Growth: {:.1}x, Similarity: {:.1}%", 
             diff_v2_v3.summary.content_growth_ratio,
             diff_v2_v3.summary.similarity_score * 100.0);
    
    Ok(())
}

async fn demo_multi_memory_search(snapshots: &[(String, String)]) -> anyhow::Result<()> {
    let config = Config::default();
    let mut engine = MultiMemoryEngine::new(config);
    
    // Add all memory snapshots
    for (i, (video, index)) in snapshots.iter().enumerate() {
        let version = format!("v{}", i + 1);
        engine.add_memory(
            &version,
            video,
            index,
            vec![format!("version_{}", i + 1)],
            Some(format!("Project documentation version {}", i + 1))
        ).await?;
    }
    
    // Search across all memories
    println!("   Searching for 'semantic search' across all versions:");
    let results = engine.search_all("semantic search", 5, false, false).await?;
    
    println!("     🔍 Found {} total results across {} memories", 
             results.total_results, results.search_metadata.memories_searched);
    
    for (i, result) in results.aggregated_results.iter().take(3).enumerate() {
        println!("     {}. [{}] Similarity: {:.2}", 
                 i + 1, result.source_memory, result.similarity);
        let preview = if result.text.len() > 80 {
            format!("{}...", &result.text[..80])
        } else {
            result.text.clone()
        };
        println!("        {}", preview);
    }
    
    // Show global stats
    let stats = engine.get_global_stats();
    println!("     📊 Global: {} memories, {} chunks, {} characters", 
             stats.total_memories, stats.total_chunks, stats.total_characters);
    
    Ok(())
}

async fn demo_temporal_analysis(snapshots: &[(String, String)]) -> anyhow::Result<()> {
    let config = Config::default();
    let temporal_engine = TemporalAnalysisEngine::new(config);
    
    // Create snapshots with metadata
    let mut memory_snapshots = Vec::new();
    for (i, (video, index)) in snapshots.iter().enumerate() {
        let snapshot = temporal_engine.create_snapshot(
            video,
            index,
            Some(format!("Version {} of project documentation", i + 1)),
            vec![format!("v{}", i + 1), "documentation".to_string()]
        ).await?;
        memory_snapshots.push(snapshot);
    }
    
    // Build timeline
    println!("   Building timeline from {} snapshots:", memory_snapshots.len());
    let timeline = temporal_engine.build_timeline(memory_snapshots).await?;
    
    println!("     📅 Timeline span: {:.1} days", timeline.analysis.total_timespan_days);
    println!("     📈 Growth trend: {}", timeline.analysis.growth_trend.overall_direction);
    println!("     ⚡ Average growth: {:.1} chunks/day", timeline.analysis.growth_trend.average_growth_rate);
    println!("     🎯 Activity periods: {}", timeline.analysis.activity_periods.len());
    
    if let Some(peak) = &timeline.analysis.growth_trend.peak_growth_period {
        println!("     🏔️  Peak growth: {} chunks added ({})", 
                 peak.chunks_added, peak.activity_type);
    }
    
    Ok(())
}
