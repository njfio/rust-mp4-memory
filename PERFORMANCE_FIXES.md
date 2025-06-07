# Performance Fixes for Large Dataset Processing

## Issue Identified

The system was experiencing significant delays when processing large datasets (1000+ chunks) after QR code generation completed. The hang occurred during the "Building search index..." phase.

## Root Cause

The performance bottleneck was in the **IndexManager's embedding generation process**:

1. **QR Code Processing**: Efficiently processed in batches of 500 chunks
2. **Index Building**: Processed ALL chunks at once (e.g., 1115 chunks) to generate embeddings
3. **Embedding Generation**: Each chunk required individual processing, causing the delay

The `IndexManager.add_chunks()` method was processing the entire dataset sequentially to generate embeddings, which became a significant bottleneck for large datasets.

## Solutions Implemented

### 1. Batched Index Building

**Before:**
```rust
// Process all chunks at once
let embeddings = model.embed_batch(&texts).await?;
```

**After:**
```rust
// Process in smaller batches with progress reporting
let batch_size = 100;
for chunk_batch in chunks.chunks(batch_size) {
    let embeddings = model.embed_batch(&texts).await?;
    // Progress reporting every 200 chunks
}
```

### 2. Enhanced Progress Reporting

Added detailed progress logging for:
- Embedding generation progress (every 50 embeddings for large batches)
- Index building progress (every 200 chunks)
- Timing information for index building phases

### 3. Configuration Option to Disable Index Building

Added `enable_index_building` option in `SearchConfig`:

```rust
pub struct SearchConfig {
    // ... other fields
    /// Enable index building (disable for faster processing when search isn't needed)
    pub enable_index_building: bool,
}
```

### 4. CLI Option for Fast Processing

Added `--no-index` flag to the encode command:

```bash
# Fast processing without search index
memvid encode --no-index --output video.mp4 --index index.json large_dataset/

# Normal processing with search index (default)
memvid encode --output video.mp4 --index index.json large_dataset/
```

### 5. Improved Embedding Batch Processing

Enhanced the embedding model to provide better progress reporting and handle large batches more efficiently:

```rust
// Progress reporting for large batches
if total_texts > 100 && (i + 1) % 50 == 0 {
    tracing::debug!("Generated {}/{} embeddings ({:.1}%)", 
        i + 1, total_texts, ((i + 1) as f64 / total_texts as f64) * 100.0);
}
```

## Performance Improvements

### Expected Results

For a dataset with 1115 chunks:

**With Index Building (Default):**
- QR Code Generation: ~12 seconds
- Index Building: ~30-60 seconds (depending on system)
- Total Time: ~42-72 seconds

**Without Index Building (`--no-index`):**
- QR Code Generation: ~12 seconds
- Index Building: Skipped
- Total Time: ~12 seconds

**Speedup: 3-6x faster** when search functionality is not needed.

### Memory Usage

- Streaming processing keeps memory usage constant regardless of dataset size
- Batched index building prevents memory spikes during embedding generation
- Frame processing uses adaptive batch sizes based on dataset size

## Usage Recommendations

### For Development/Testing
```bash
# Fast processing without search
memvid encode --no-index --output test.mp4 --index test.json large_dataset/
```

### For Production Use
```bash
# Full processing with search capabilities
memvid encode --output production.mp4 --index production.json dataset/
```

### For Very Large Datasets (10k+ chunks)
```bash
# Use smaller chunk sizes and disable index initially
memvid encode --no-index --chunk-size 800 --output large.mp4 --index large.json huge_dataset/

# Build index separately if needed later (future feature)
```

## Configuration Options

### In `memvid.toml`:
```toml
[search]
enable_index_building = false  # Disable for faster processing

[text]
chunk_size = 800  # Smaller chunks for better QR compatibility
```

### Environment Variables:
```bash
export SKIP_PERFORMANCE_TESTS=1  # Skip performance tests in CI
```

## Testing

Run performance tests to verify improvements:

```bash
# Run performance tests (may take several minutes)
cargo test test_large_dataset_performance --release

# Skip performance tests
SKIP_PERFORMANCE_TESTS=1 cargo test
```

## Monitoring

The system now provides detailed logging to monitor performance:

```
INFO Building search index for 1115 chunks...
DEBUG Generated 100/1115 embeddings (9.0%)
DEBUG Generated 200/1115 embeddings (17.9%)
...
INFO Index building completed in 45.23s
INFO Index saved successfully
```

## Future Optimizations

1. **Incremental Index Building**: Add chunks to index during streaming processing
2. **Parallel Embedding Generation**: Use multiple threads for embedding generation
3. **Index Caching**: Cache embeddings to avoid regeneration
4. **Compressed Embeddings**: Use quantized embeddings to reduce memory usage
5. **Background Index Building**: Build index asynchronously after video creation
