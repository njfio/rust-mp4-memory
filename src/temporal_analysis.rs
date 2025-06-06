//! Temporal analysis for tracking memory evolution over time

use std::collections::HashMap;
use std::path::Path;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::memory_diff::{MemoryDiff, MemoryDiffEngine};
use crate::retriever::MemvidRetriever;

/// Represents a memory snapshot at a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub video_path: String,
    pub index_path: String,
    pub metadata: SnapshotMetadata,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// Metadata about a memory snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub unique_sources: usize,
    pub file_size_bytes: u64,
    pub creation_duration_seconds: f64,
    pub content_hash: String, // Hash of all content for quick comparison
}

/// Timeline of memory evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimeline {
    pub snapshots: Vec<MemorySnapshot>,
    pub diffs: Vec<MemoryDiff>,
    pub analysis: TimelineAnalysis,
}

/// Analysis of memory evolution over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineAnalysis {
    pub total_timespan_days: f64,
    pub growth_trend: GrowthTrend,
    pub activity_periods: Vec<ActivityPeriod>,
    pub content_evolution: ContentEvolution,
    pub knowledge_gaps: Vec<KnowledgeGap>,
}

/// Growth trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthTrend {
    pub overall_direction: String, // "growing", "shrinking", "stable", "volatile"
    pub average_growth_rate: f64, // chunks per day
    pub peak_growth_period: Option<ActivityPeriod>,
    pub content_velocity: f64, // characters per day
}

/// Period of high activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityPeriod {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub activity_type: String, // "high_growth", "major_revision", "consolidation"
    pub chunks_added: usize,
    pub chunks_modified: usize,
    pub description: String,
}

/// Evolution of content types and topics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentEvolution {
    pub dominant_topics_over_time: Vec<TopicTrend>,
    pub source_diversity_trend: Vec<SourceDiversityPoint>,
    pub content_quality_metrics: Vec<QualityMetric>,
}

/// Trend of a specific topic over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTrend {
    pub topic: String,
    pub timeline_points: Vec<TopicTimelinePoint>,
    pub trend_direction: String, // "increasing", "decreasing", "stable", "emerging", "declining"
}

/// Topic presence at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTimelinePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub relevance_score: f64,
    pub chunk_count: usize,
}

/// Source diversity at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDiversityPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub unique_sources: usize,
    pub diversity_index: f64, // Shannon diversity index
    pub dominant_source_percentage: f64,
}

/// Quality metric over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetric {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metric_name: String,
    pub value: f64,
    pub description: String,
}

/// Identified knowledge gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub topic_area: String,
    pub gap_type: String, // "missing_coverage", "outdated_info", "conflicting_info"
    pub confidence: f64,
    pub suggested_action: String,
    pub related_snapshots: Vec<String>,
}

/// Temporal analysis engine
pub struct TemporalAnalysisEngine {
    config: Config,
    diff_engine: MemoryDiffEngine,
}

impl TemporalAnalysisEngine {
    /// Create a new temporal analysis engine
    pub fn new(config: Config) -> Self {
        let diff_engine = MemoryDiffEngine::new(config.clone());
        Self {
            config,
            diff_engine,
        }
    }

    /// Create a memory snapshot from a video and index
    pub async fn create_snapshot(
        &self,
        video_path: &str,
        index_path: &str,
        description: Option<String>,
        tags: Vec<String>,
    ) -> Result<MemorySnapshot> {
        info!("Creating memory snapshot for {}", video_path);

        // Load the memory to get metadata
        let retriever = MemvidRetriever::new_with_config(video_path, index_path, self.config.clone()).await?;
        let stats = retriever.get_stats();

        // Calculate file size
        let file_size = std::fs::metadata(video_path)?.len();

        // Calculate content hash for quick comparison
        let content_hash = self.calculate_content_hash(index_path).await?;

        let metadata = SnapshotMetadata {
            total_chunks: stats.index_stats.total_chunks,
            total_characters: stats.index_stats.total_characters,
            unique_sources: stats.index_stats.unique_sources,
            file_size_bytes: file_size,
            creation_duration_seconds: 0.0, // TODO: Track this during encoding
            content_hash,
        };

        Ok(MemorySnapshot {
            timestamp: chrono::Utc::now(),
            video_path: video_path.to_string(),
            index_path: index_path.to_string(),
            metadata,
            tags,
            description,
        })
    }

    /// Build a timeline from multiple snapshots
    pub async fn build_timeline(&self, snapshots: Vec<MemorySnapshot>) -> Result<MemoryTimeline> {
        info!("Building timeline from {} snapshots", snapshots.len());

        if snapshots.is_empty() {
            return Err(MemvidError::invalid_input("No snapshots provided"));
        }

        // Sort snapshots by timestamp
        let mut sorted_snapshots = snapshots;
        sorted_snapshots.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Generate diffs between consecutive snapshots
        let mut diffs = Vec::new();
        for i in 1..sorted_snapshots.len() {
            let old_snapshot = &sorted_snapshots[i - 1];
            let new_snapshot = &sorted_snapshots[i];

            match self.diff_engine.compare_memories(
                &old_snapshot.video_path,
                &old_snapshot.index_path,
                &new_snapshot.video_path,
                &new_snapshot.index_path,
            ).await {
                Ok(diff) => diffs.push(diff),
                Err(e) => {
                    debug!("Failed to generate diff between snapshots: {}", e);
                    // Continue with other diffs
                }
            }
        }

        // Perform timeline analysis
        let analysis = self.analyze_timeline(&sorted_snapshots, &diffs).await?;

        Ok(MemoryTimeline {
            snapshots: sorted_snapshots,
            diffs,
            analysis,
        })
    }

    /// Analyze the timeline for trends and patterns
    async fn analyze_timeline(
        &self,
        snapshots: &[MemorySnapshot],
        diffs: &[MemoryDiff],
    ) -> Result<TimelineAnalysis> {
        let timespan_days = if snapshots.len() > 1 {
            let duration = snapshots.last().unwrap().timestamp - snapshots.first().unwrap().timestamp;
            duration.num_days() as f64
        } else {
            0.0
        };

        let growth_trend = self.analyze_growth_trend(snapshots, diffs);
        let activity_periods = self.identify_activity_periods(snapshots, diffs);
        let content_evolution = self.analyze_content_evolution(snapshots, diffs).await?;
        let knowledge_gaps = self.identify_knowledge_gaps(snapshots, diffs).await?;

        Ok(TimelineAnalysis {
            total_timespan_days: timespan_days,
            growth_trend,
            activity_periods,
            content_evolution,
            knowledge_gaps,
        })
    }

    /// Analyze growth trends
    fn analyze_growth_trend(&self, snapshots: &[MemorySnapshot], diffs: &[MemoryDiff]) -> GrowthTrend {
        if snapshots.len() < 2 {
            return GrowthTrend {
                overall_direction: "insufficient_data".to_string(),
                average_growth_rate: 0.0,
                peak_growth_period: None,
                content_velocity: 0.0,
            };
        }

        let first = &snapshots[0];
        let last = &snapshots[snapshots.len() - 1];
        
        let total_chunk_growth = last.metadata.total_chunks as i64 - first.metadata.total_chunks as i64;
        let total_char_growth = last.metadata.total_characters as i64 - first.metadata.total_characters as i64;
        
        let duration = last.timestamp - first.timestamp;
        let days = duration.num_days() as f64;
        
        let average_growth_rate = if days > 0.0 {
            total_chunk_growth as f64 / days
        } else {
            0.0
        };

        let content_velocity = if days > 0.0 {
            total_char_growth as f64 / days
        } else {
            0.0
        };

        let overall_direction = if average_growth_rate > 1.0 {
            "growing"
        } else if average_growth_rate < -1.0 {
            "shrinking"
        } else if self.is_volatile(snapshots) {
            "volatile"
        } else {
            "stable"
        }.to_string();

        // Find peak growth period
        let peak_growth_period = self.find_peak_growth_period(diffs);

        GrowthTrend {
            overall_direction,
            average_growth_rate,
            peak_growth_period,
            content_velocity,
        }
    }

    /// Check if growth is volatile
    fn is_volatile(&self, snapshots: &[MemorySnapshot]) -> bool {
        if snapshots.len() < 3 {
            return false;
        }

        let mut changes = Vec::new();
        for i in 1..snapshots.len() {
            let change = snapshots[i].metadata.total_chunks as i64 - snapshots[i-1].metadata.total_chunks as i64;
            changes.push(change);
        }

        // Calculate variance
        let mean = changes.iter().sum::<i64>() as f64 / changes.len() as f64;
        let variance = changes.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / changes.len() as f64;

        variance > 100.0 // Arbitrary threshold for volatility
    }

    /// Find the period with highest growth
    fn find_peak_growth_period(&self, diffs: &[MemoryDiff]) -> Option<ActivityPeriod> {
        if diffs.is_empty() {
            return None;
        }

        let mut max_growth = 0;
        let mut peak_diff: Option<&MemoryDiff> = None;

        for diff in diffs {
            let growth = diff.summary.added_count;
            if growth > max_growth {
                max_growth = growth;
                peak_diff = Some(diff);
            }
        }

        peak_diff.map(|diff| ActivityPeriod {
            start_time: diff.timestamp - chrono::Duration::days(1), // Approximate
            end_time: diff.timestamp,
            activity_type: "high_growth".to_string(),
            chunks_added: diff.summary.added_count,
            chunks_modified: diff.summary.modified_count,
            description: format!("Peak growth period with {} new chunks added", diff.summary.added_count),
        })
    }

    /// Identify periods of high activity
    fn identify_activity_periods(&self, _snapshots: &[MemorySnapshot], diffs: &[MemoryDiff]) -> Vec<ActivityPeriod> {
        let mut periods = Vec::new();

        for diff in diffs {
            let total_changes = diff.summary.added_count + diff.summary.modified_count + diff.summary.removed_count;
            
            if total_changes > 10 { // Threshold for "high activity"
                let activity_type = if diff.summary.added_count > diff.summary.modified_count {
                    "high_growth"
                } else if diff.summary.modified_count > diff.summary.added_count {
                    "major_revision"
                } else {
                    "consolidation"
                };

                periods.push(ActivityPeriod {
                    start_time: diff.timestamp - chrono::Duration::hours(1), // Approximate
                    end_time: diff.timestamp,
                    activity_type: activity_type.to_string(),
                    chunks_added: diff.summary.added_count,
                    chunks_modified: diff.summary.modified_count,
                    description: format!("High activity period: {} changes", total_changes),
                });
            }
        }

        periods
    }

    /// Analyze content evolution (placeholder)
    async fn analyze_content_evolution(&self, _snapshots: &[MemorySnapshot], _diffs: &[MemoryDiff]) -> Result<ContentEvolution> {
        // TODO: Implement topic analysis and source diversity tracking
        Ok(ContentEvolution {
            dominant_topics_over_time: Vec::new(),
            source_diversity_trend: Vec::new(),
            content_quality_metrics: Vec::new(),
        })
    }

    /// Identify knowledge gaps (placeholder)
    async fn identify_knowledge_gaps(&self, _snapshots: &[MemorySnapshot], _diffs: &[MemoryDiff]) -> Result<Vec<KnowledgeGap>> {
        // TODO: Implement knowledge gap detection
        Ok(Vec::new())
    }

    /// Calculate content hash for quick comparison
    async fn calculate_content_hash(&self, index_path: &str) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Read index file and hash its content
        let content = std::fs::read_to_string(index_path)?;
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Save timeline to file
    pub fn save_timeline(&self, timeline: &MemoryTimeline, output_path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(timeline)?;
        std::fs::write(output_path, json)?;
        info!("Timeline saved to {}", output_path);
        Ok(())
    }

    /// Load timeline from file
    pub fn load_timeline(&self, timeline_path: &str) -> Result<MemoryTimeline> {
        let content = std::fs::read_to_string(timeline_path)?;
        let timeline: MemoryTimeline = serde_json::from_str(&content)?;
        Ok(timeline)
    }
}
