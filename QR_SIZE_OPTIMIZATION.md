# QR Code Size Optimization Guide

This document explains the QR code size limitations and how Rust MemVid automatically handles them.

## The Problem

QR codes have strict size limitations based on their error correction level:

- **Low (L)**: ~4,296 characters maximum
- **Medium (M)**: ~3,391 characters maximum  
- **Quartile (Q)**: ~2,420 characters maximum
- **High (H)**: ~1,852 characters maximum

When processing large files, individual chunks might exceed these limits, causing the error:
```
Failed to build video: QR code error: data too long
```

## The Solution

Rust MemVid now includes **automatic QR code size optimization** that:

1. **Detects oversized chunks** before QR generation
2. **Automatically splits large chunks** into QR-compatible sizes
3. **Preserves content integrity** while maintaining searchability
4. **Provides detailed feedback** about the optimization process

## How It Works

### Automatic Optimization

When you call `build_video()`, the system automatically:

```rust
// This happens automatically in build_video()
encoder.optimize_chunks_for_qr()?;
```

### Smart Chunk Splitting

Large chunks are intelligently split:
- **Preserves word boundaries** when possible
- **Maintains metadata** for each sub-chunk
- **Updates frame numbers** sequentially
- **Keeps source information** intact

### Configuration-Based Limits

The system calculates safe chunk sizes based on your QR configuration:

```rust
let recommended_size = encoder.get_recommended_chunk_size()?;
// Returns ~2,373 characters for Medium error correction
```

## Usage Examples

### CLI Usage

```bash
# Safe default chunk size (recommended)
memvid encode --chunk-size 1000 --output video.mp4 --index index.json large_file.txt

# If you get "data too long" error, reduce chunk size
memvid encode --chunk-size 800 --output video.mp4 --index index.json large_file.txt

# For very large files, use smaller chunks
memvid encode --chunk-size 500 --output video.mp4 --index index.json huge_file.txt
```

### Programmatic Usage

```rust
use rust_mem_vid::MemvidEncoder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut encoder = MemvidEncoder::new().await?;
    
    // Add your content
    encoder.add_text_file("large_file.txt").await?;
    
    // Check if optimization is needed
    let analysis = encoder.analyze_chunks();
    if analysis.needs_optimization {
        println!("Will optimize {} oversized chunks", analysis.oversized_chunks);
    }
    
    // Build video (automatic optimization happens here)
    let stats = encoder.build_video("output.mp4", "index.json").await?;
    
    println!("Final video has {} chunks", stats.total_chunks);
    
    Ok(())
}
```

### Manual Optimization

```rust
// Get recommended chunk size
let recommended = encoder.get_recommended_chunk_size()?;
println!("Recommended chunk size: {} characters", recommended);

// Analyze current chunks
let analysis = encoder.analyze_chunks();
println!("Oversized chunks: {}", analysis.oversized_chunks);

// Manually optimize if needed
if analysis.needs_optimization {
    encoder.optimize_chunks_for_qr()?;
}
```

## Configuration Options

### Chunk Size Settings

```rust
use rust_mem_vid::Config;

let mut config = Config::default();

// Conservative chunk size (always safe)
config.text.chunk_size = 800;

// Balanced chunk size (recommended)
config.text.chunk_size = 1000;

// Aggressive chunk size (may need optimization)
config.text.chunk_size = 1500;
```

### Error Correction Levels

Lower error correction = higher capacity:

```toml
[qr]
error_correction = "L"  # Highest capacity (~4,296 chars)
error_correction = "M"  # Balanced (~3,391 chars) - DEFAULT
error_correction = "Q"  # Conservative (~2,420 chars)
error_correction = "H"  # Lowest capacity (~1,852 chars)
```

## Troubleshooting

### Error: "data too long"

**Immediate Solutions:**
1. Reduce chunk size: `--chunk-size 800`
2. Use lower error correction in config
3. Process smaller files or use filtering

**Example Fix:**
```bash
# Instead of this (might fail):
memvid encode --chunk-size 2000 --output video.mp4 large_file.txt

# Use this (will work):
memvid encode --chunk-size 800 --output video.mp4 large_file.txt
```

### Performance Considerations

**Chunk Size Trade-offs:**

- **Smaller chunks** (500-800 chars):
  - ✅ Always QR-compatible
  - ✅ Faster processing per chunk
  - ❌ More chunks = larger video files
  - ❌ More granular search results

- **Larger chunks** (1000-1500 chars):
  - ✅ Fewer chunks = smaller video files
  - ✅ More context per search result
  - ❌ May need optimization
  - ❌ Slower processing per chunk

**Recommended Settings:**

```bash
# For most use cases (balanced)
--chunk-size 1000

# For large files (conservative)
--chunk-size 800

# For maximum compatibility (safe)
--chunk-size 500
```

## Advanced Features

### Chunk Analysis

```rust
let analysis = encoder.analyze_chunks();

println!("📊 Chunk Analysis:");
println!("   Total chunks: {}", analysis.total_chunks);
println!("   Average size: {} chars", analysis.avg_chunk_size);
println!("   Max size: {} chars", analysis.max_chunk_size);
println!("   Recommended: {} chars", analysis.recommended_size);
println!("   Oversized: {}", analysis.oversized_chunks);
println!("   Needs optimization: {}", analysis.needs_optimization);
```

### Optimization Monitoring

```rust
// Before optimization
let before = encoder.analyze_chunks();

// Optimize
encoder.optimize_chunks_for_qr()?;

// After optimization  
let after = encoder.analyze_chunks();

println!("Optimization results:");
println!("  Chunks: {} → {}", before.total_chunks, after.total_chunks);
println!("  Max size: {} → {}", before.max_chunk_size, after.max_chunk_size);
println!("  Oversized: {} → {}", before.oversized_chunks, after.oversized_chunks);
```

## Testing

Run the QR size optimization test:

```bash
cargo run --example qr_size_test
```

This demonstrates:
- Automatic optimization with different chunk sizes
- Before/after analysis
- Performance impact of different strategies
- Real QR code generation and video creation

## Best Practices

1. **Start with default settings** (chunk_size = 1000)
2. **Monitor optimization logs** during encoding
3. **Reduce chunk size** if you see many optimizations
4. **Use folder processing** with size limits for large datasets
5. **Test with sample data** before processing large files

## Summary

The QR code size optimization system ensures that Rust MemVid can handle files of any size while maintaining QR code compatibility. The automatic optimization is transparent and preserves all functionality while preventing encoding failures.

**Key Benefits:**
- ✅ **Automatic handling** of oversized chunks
- ✅ **No data loss** during optimization
- ✅ **Maintains searchability** and metadata
- ✅ **Provides clear feedback** about optimizations
- ✅ **Configurable behavior** for different use cases
- ✅ **Performance monitoring** and analysis tools
