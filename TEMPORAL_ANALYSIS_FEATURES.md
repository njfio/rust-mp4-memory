# 🕰️ Temporal Analysis and Memory Comparison Features

This document describes the powerful new temporal analysis and memory comparison capabilities added to Rust MemVid.

## 🌟 Overview

Each MP4 memory video is indeed a "frozen snapshot" in time, but now you can:

- **Compare memories** to see what changed between versions
- **Search across multiple memories** simultaneously 
- **Track memory evolution** over time with detailed analytics
- **Find correlations** between different memory snapshots
- **Identify knowledge gaps** and optimization opportunities

## 🔍 Memory Comparison & Diff Analysis

### Features
- **Chunk-level comparison** between two memory videos
- **Semantic change detection** with similarity scoring
- **Content growth and reduction analysis**
- **Modification type classification** (minor edit, major rewrite, expansion, etc.)
- **Detailed diff reports** with before/after statistics

### Usage

#### CLI
```bash
# Compare two memory videos
memvid diff old_memory.mp4 old_memory.metadata new_memory.mp4 new_memory.metadata \
  --output diff_report.json --semantic

# Example output:
# 🔍 Memory Comparison Results
# ============================
# Old memory: old_memory.mp4
# New memory: new_memory.mp4
# 
# 📊 Summary:
#    • Old chunks: 5
#    • New chunks: 8
#    • Added: 3 chunks
#    • Removed: 0 chunks
#    • Modified: 2 chunks
#    • Unchanged: 3 chunks
#    • Similarity: 75.0%
#    • Growth ratio: 1.6x
```

#### Programmatic
```rust
use rust_mem_vid::memory_diff::MemoryDiffEngine;

let config = Config::default();
let diff_engine = MemoryDiffEngine::new(config)
    .with_semantic_analysis(true);

let diff = diff_engine.compare_memories(
    "old_memory.mp4", "old_memory.metadata",
    "new_memory.mp4", "new_memory.metadata"
).await?;

println!("Added {} chunks, removed {} chunks", 
         diff.summary.added_count, diff.summary.removed_count);
```

## 🔎 Multi-Memory Search

### Features
- **Search across multiple memory videos** simultaneously
- **Cross-reference information** between different snapshots
- **Find correlations** across time periods
- **Temporal context** for search results
- **Memory filtering** by tags, creation date, or names

### Usage

#### CLI
```bash
# Create memories configuration file
cat > memories.json << EOF
[
  {
    "name": "v1",
    "video_path": "v1_memory.mp4",
    "index_path": "v1_memory.metadata",
    "tags": ["version_1", "initial"],
    "description": "Initial project documentation"
  },
  {
    "name": "v2", 
    "video_path": "v2_memory.mp4",
    "index_path": "v2_memory.metadata",
    "tags": ["version_2", "enhanced"],
    "description": "Enhanced with semantic search"
  }
]
EOF

# Search across all memories
memvid multi-search "machine learning" memories.json \
  --top-k 10 --correlations --temporal

# Filter by tags
memvid multi-search "API documentation" memories.json \
  --tags enhanced --top-k 5
```

#### Programmatic
```rust
use rust_mem_vid::multi_memory::{MultiMemoryEngine, MemoryFilter};

let mut engine = MultiMemoryEngine::new(config);

// Add memories
engine.add_memory("v1", "v1.mp4", "v1.metadata", 
                  vec!["version_1".to_string()], 
                  Some("Initial version".to_string())).await?;

engine.add_memory("v2", "v2.mp4", "v2.metadata",
                  vec!["version_2".to_string()],
                  Some("Enhanced version".to_string())).await?;

// Search across all memories
let results = engine.search_all("semantic search", 10, true, true).await?;

println!("Found {} results across {} memories", 
         results.total_results, results.search_metadata.memories_searched);

// Show cross-memory correlations
for correlation in &results.cross_memory_correlations {
    println!("Correlation: {} ↔ {} ({:?})", 
             correlation.memory1, correlation.memory2, correlation.correlation_type);
}
```

## 📈 Temporal Analysis & Timeline Tracking

### Features
- **Memory evolution tracking** over time
- **Growth trend analysis** (growing, shrinking, stable, volatile)
- **Activity period detection** (high growth, major revision, consolidation)
- **Content evolution patterns** and topic trends
- **Knowledge gap identification**

### Usage

#### Programmatic
```rust
use rust_mem_vid::temporal_analysis::TemporalAnalysisEngine;

let temporal_engine = TemporalAnalysisEngine::new(config);

// Create snapshots
let mut snapshots = Vec::new();
for (video, index) in memory_files {
    let snapshot = temporal_engine.create_snapshot(
        &video, &index,
        Some("Project documentation".to_string()),
        vec!["documentation".to_string()]
    ).await?;
    snapshots.push(snapshot);
}

// Build timeline
let timeline = temporal_engine.build_timeline(snapshots).await?;

println!("Timeline Analysis:");
println!("  Timespan: {:.1} days", timeline.analysis.total_timespan_days);
println!("  Growth trend: {}", timeline.analysis.growth_trend.overall_direction);
println!("  Average growth: {:.1} chunks/day", timeline.analysis.growth_trend.average_growth_rate);
println!("  Activity periods: {}", timeline.analysis.activity_periods.len());

// Save timeline for later analysis
temporal_engine.save_timeline(&timeline, "project_timeline.json")?;
```

## 🎯 Advanced Use Cases

### 1. Research Project Evolution
Track how your research evolves over time:
```rust
// Create snapshots at key milestones
let snapshots = vec![
    create_snapshot("research_proposal.mp4", "Initial proposal"),
    create_snapshot("literature_review.mp4", "After literature review"),
    create_snapshot("methodology.mp4", "Methodology defined"),
    create_snapshot("results.mp4", "Results collected"),
    create_snapshot("final_paper.mp4", "Final paper")
];

let timeline = analyze_timeline(snapshots).await?;
identify_knowledge_gaps(&timeline);
```

### 2. Knowledge Base Maintenance
Monitor your knowledge base for quality and gaps:
```rust
// Compare monthly snapshots
let monthly_diffs = vec![];
for i in 1..snapshots.len() {
    let diff = compare_memories(&snapshots[i-1], &snapshots[i]).await?;
    monthly_diffs.push(diff);
}

// Identify patterns
analyze_content_quality_trends(&monthly_diffs);
detect_information_decay(&monthly_diffs);
```

### 3. Collaborative Knowledge Building
Track team contributions and knowledge evolution:
```rust
// Search across team member memories
let team_results = engine.search_all("project requirements", 20, true, true).await?;

// Find complementary information
for correlation in &team_results.cross_memory_correlations {
    if correlation.correlation_type == CorrelationType::Complementary {
        println!("Found complementary info between {} and {}", 
                 correlation.memory1, correlation.memory2);
    }
}
```

### 4. Content Quality Assessment
Analyze how your content quality changes over time:
```rust
let timeline = build_timeline(snapshots).await?;

for period in &timeline.analysis.activity_periods {
    match period.activity_type.as_str() {
        "high_growth" => println!("Rapid expansion period: {}", period.description),
        "major_revision" => println!("Quality improvement period: {}", period.description),
        "consolidation" => println!("Cleanup and organization period: {}", period.description),
        _ => {}
    }
}
```

## 🔧 Configuration Options

### Memory Diff Engine
```rust
let diff_engine = MemoryDiffEngine::new(config)
    .with_similarity_threshold(0.8)  // Chunks >80% similar = "modified"
    .with_semantic_analysis(true);   // Enable deep semantic analysis
```

### Multi-Memory Engine
```rust
let mut engine = MultiMemoryEngine::new(config);

// Filter memories by various criteria
let results = engine.search_filtered(
    "query",
    10,
    MemoryFilter::Tags(vec!["recent".to_string()])
).await?;

let results = engine.search_filtered(
    "query", 
    10,
    MemoryFilter::CreatedAfter(chrono::Utc::now() - chrono::Duration::days(30))
).await?;
```

### Temporal Analysis Engine
```rust
let temporal_engine = TemporalAnalysisEngine::new(config);

// Create snapshots with rich metadata
let snapshot = temporal_engine.create_snapshot(
    "memory.mp4", "memory.metadata",
    Some("Quarterly knowledge review".to_string()),
    vec!["quarterly".to_string(), "review".to_string(), "2024".to_string()]
).await?;
```

## 📊 Output Formats

### Diff Reports (JSON)
```json
{
  "old_memory": "v1_memory.mp4",
  "new_memory": "v2_memory.mp4", 
  "timestamp": "2024-01-15T10:30:00Z",
  "summary": {
    "total_old_chunks": 5,
    "total_new_chunks": 8,
    "added_count": 3,
    "removed_count": 0,
    "modified_count": 2,
    "unchanged_count": 3,
    "similarity_score": 0.75,
    "content_growth_ratio": 1.6
  },
  "added_chunks": [...],
  "removed_chunks": [...],
  "modified_chunks": [...],
  "semantic_changes": [...]
}
```

### Timeline Analysis (JSON)
```json
{
  "snapshots": [...],
  "diffs": [...],
  "analysis": {
    "total_timespan_days": 90.5,
    "growth_trend": {
      "overall_direction": "growing",
      "average_growth_rate": 2.3,
      "content_velocity": 1250.5
    },
    "activity_periods": [...],
    "knowledge_gaps": [...]
  }
}
```

## 🚀 Performance Considerations

- **Memory usage**: Each loaded memory uses ~10-50MB RAM depending on size
- **Search speed**: Multi-memory search scales linearly with number of memories
- **Diff computation**: O(n*m) where n,m are chunk counts in compared memories
- **Timeline analysis**: Efficient for up to 100+ snapshots

## 🎉 Benefits

1. **Track Knowledge Evolution**: See how your understanding develops over time
2. **Identify Knowledge Gaps**: Find areas that need more attention
3. **Quality Assurance**: Monitor content quality and consistency
4. **Collaboration**: Merge insights from multiple team members
5. **Research Insights**: Understand how projects and ideas evolve
6. **Content Optimization**: Find redundancies and optimize information architecture

## 🔮 Future Enhancements

- **Visual timeline graphs** and charts
- **Automated knowledge gap recommendations**
- **Smart memory merging** with conflict resolution
- **Topic evolution visualization**
- **Collaborative editing** with change tracking
- **Integration with version control systems**

These temporal analysis features transform Rust MemVid from a simple memory storage system into a powerful knowledge evolution and analysis platform!
