# Background Indexing System

## Overview

The background indexing system allows MemVid to create video files quickly while building search indexes asynchronously in the background. This provides the best of both worlds: fast initial processing with search capabilities available later.

## Key Benefits

1. **Fast Initial Processing**: Video creation completes immediately without waiting for index building
2. **Non-blocking**: Index building happens in the background without impacting other operations
3. **Progress Monitoring**: Real-time progress tracking and status monitoring
4. **Fault Tolerance**: Failed jobs can be retried, and the system continues processing other jobs
5. **Resource Management**: Background processing uses separate threads and manages memory efficiently

## Architecture

### Components

1. **BackgroundIndexer**: Main service that manages indexing jobs
2. **IndexingJob**: Represents a single indexing task with chunks and metadata
3. **IndexingStatus**: Tracks job progress (Queued, InProgress, Completed, Failed)
4. **Worker Loop**: Processes jobs asynchronously in the background

### Flow

```
1. Video Creation (Fast)
   ├── Process chunks → QR codes → Video file
   └── Submit background indexing job
   
2. Background Processing (Async)
   ├── Generate embeddings in batches
   ├── Build vector index
   └── Save search index
```

## Usage

### CLI Commands

#### Basic Usage with Background Indexing

```bash
# Enable background indexing for fast processing
memvid encode --background-index --output video.mp4 --index index.json large_dataset/

# The command returns immediately with a job ID
# Background indexing job submitted: idx_a1b2c3d4e5f6...
```

#### Monitoring Background Jobs

```bash
# Check status of a specific job
memvid index-status idx_a1b2c3d4e5f6

# List all background indexing jobs
memvid index-jobs

# Wait for a job to complete (with optional timeout)
memvid index-wait idx_a1b2c3d4e5f6 --timeout 300
```

### Configuration Options

#### In `memvid.toml`:
```toml
[search]
enable_index_building = false        # Disable immediate indexing
enable_background_indexing = true    # Enable background indexing
```

#### CLI Flags:
```bash
--background-index    # Enable background indexing
--no-index           # Disable all indexing (fastest)
```

### Programmatic Usage

```rust
use rust_mem_vid::{MemvidEncoder, submit_background_indexing, get_indexing_status};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure for background indexing
    let mut config = Config::default();
    config.search.enable_index_building = false;
    config.search.enable_background_indexing = true;
    
    // Create encoder and add content
    let mut encoder = MemvidEncoder::new_with_config(config.clone()).await?;
    encoder.add_directory("large_dataset/").await?;
    
    // Build video (fast - no indexing)
    let stats = encoder.build_video("video.mp4", "index.json").await?;
    println!("Video created in {:.2}s", stats.encoding_time_seconds);
    
    // Submit background indexing job manually (if needed)
    let job_id = submit_background_indexing(
        encoder.chunks().to_vec(),
        "index.json".into(),
        config
    ).await?;
    
    println!("Background indexing job: {}", job_id);
    
    // Monitor progress
    loop {
        match get_indexing_status(&job_id).await {
            Some(IndexingStatus::Completed { duration_seconds }) => {
                println!("Indexing completed in {:.2}s", duration_seconds);
                break;
            }
            Some(IndexingStatus::InProgress { progress }) => {
                println!("Progress: {:.1}%", progress);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
            Some(IndexingStatus::Failed { error }) => {
                println!("Indexing failed: {}", error);
                break;
            }
            _ => {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
    
    Ok(())
}
```

## Performance Comparison

### Processing 1115 Chunks

| Mode | Video Creation | Index Building | Total Time | Search Available |
|------|----------------|----------------|------------|------------------|
| **Immediate Indexing** | ~12s | ~45s | ~57s | Immediately |
| **Background Indexing** | ~12s | ~45s (async) | ~12s | After ~45s |
| **No Indexing** | ~12s | Skipped | ~12s | Never |

### Benefits by Dataset Size

- **Small datasets (<100 chunks)**: Minimal benefit, immediate indexing is fine
- **Medium datasets (100-1000 chunks)**: 2-3x faster initial processing
- **Large datasets (1000+ chunks)**: 3-6x faster initial processing

## Job Management

### Job Status Types

1. **Queued**: Job submitted but not yet started
2. **InProgress**: Currently processing with progress percentage
3. **Completed**: Successfully finished with duration
4. **Failed**: Error occurred with error message

### Monitoring Commands

```bash
# Quick status check
memvid index-status job_id

# Detailed job listing
memvid index-jobs
# Output:
# Job ID                                   Status          Progress
# ----------------------------------------------------------------------
# idx_a1b2c3d4e5f6                        In Progress     45.2%
# idx_f6e5d4c3b2a1                        Completed       12.34s
# idx_1234567890ab                        Failed          -

# Wait for completion
memvid index-wait job_id --timeout 300
```

### Cleanup

The system automatically cleans up old completed/failed jobs to prevent memory leaks. You can also manually clean up jobs older than a specified time.

## Error Handling

### Common Issues

1. **Job Not Found**: Job ID doesn't exist or was cleaned up
2. **Timeout**: Job took longer than expected timeout
3. **Memory Issues**: Large datasets may require more memory
4. **Disk Space**: Insufficient space for index files

### Recovery

```bash
# Check if job failed
memvid index-status job_id

# If failed, you can retry by re-running the encode command
memvid encode --background-index --output video.mp4 --index index.json dataset/
```

## Best Practices

### When to Use Background Indexing

✅ **Use background indexing when:**
- Processing large datasets (1000+ chunks)
- Need immediate video file access
- Running in production pipelines
- Processing multiple datasets sequentially

❌ **Don't use background indexing when:**
- Small datasets (<100 chunks)
- Need immediate search functionality
- Running one-off processing tasks
- Limited system resources

### Resource Management

```bash
# For very large datasets, consider smaller chunk sizes
memvid encode --background-index --chunk-size 800 --output video.mp4 --index index.json huge_dataset/

# Monitor system resources during background processing
htop  # or your preferred system monitor
```

### Production Deployment

1. **Monitor job queues**: Regularly check `memvid index-jobs`
2. **Set up alerts**: Monitor for failed jobs
3. **Resource limits**: Ensure adequate memory and CPU for background processing
4. **Cleanup**: Periodically clean up old completed jobs

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/process-data.yml
- name: Process dataset with background indexing
  run: |
    memvid encode --background-index --output ${{ matrix.dataset }}.mp4 --index ${{ matrix.dataset }}.json datasets/${{ matrix.dataset }}/
    echo "Video created, indexing in background"
    
- name: Wait for indexing (optional)
  run: |
    JOB_ID=$(memvid index-jobs | tail -1 | cut -d' ' -f1)
    memvid index-wait $JOB_ID --timeout 600
```

### Batch Processing Script

```bash
#!/bin/bash
# process_multiple_datasets.sh

for dataset in dataset1 dataset2 dataset3; do
    echo "Processing $dataset..."
    memvid encode --background-index --output "${dataset}.mp4" --index "${dataset}.json" "data/${dataset}/"
    echo "Video created for $dataset"
done

echo "All videos created. Monitoring background indexing..."
memvid index-jobs
```

## Future Enhancements

1. **Priority Queues**: High-priority jobs processed first
2. **Distributed Processing**: Multiple worker nodes
3. **Incremental Indexing**: Add new chunks to existing indexes
4. **Index Compression**: Reduce index file sizes
5. **Webhook Notifications**: Notify external systems when jobs complete
