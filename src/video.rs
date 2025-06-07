//! Video encoding and decoding functionality
//! 
//! This is a simplified implementation for the initial version.
//! In a full implementation, this would use FFmpeg for video processing.

use image::DynamicImage;
use std::path::Path;

use crate::config::{Config, CodecConfig};
use crate::error::{MemvidError, Result};

/// Supported video codecs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Mp4v,
    H264,
    H265,
    Av1,
    Vp9,
}

impl Codec {
    /// Convert codec to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Codec::Mp4v => "mp4v",
            Codec::H264 => "h264",
            Codec::H265 => "h265",
            Codec::Av1 => "av1",
            Codec::Vp9 => "vp9",
        }
    }

    /// Parse codec from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "mp4v" => Ok(Codec::Mp4v),
            "h264" | "avc" => Ok(Codec::H264),
            "h265" | "hevc" => Ok(Codec::H265),
            "av1" => Ok(Codec::Av1),
            "vp9" => Ok(Codec::Vp9),
            _ => Err(MemvidError::codec(format!("Unsupported codec: {}", s))),
        }
    }
}

/// Video encoder for creating QR code videos
pub struct VideoEncoder {
    config: Config,
}

impl VideoEncoder {
    /// Create a new video encoder
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Create encoder with default configuration
    pub fn default() -> Self {
        Self::new(Config::default())
    }

    /// Encode QR code images into a video file (optimized implementation)
    pub async fn encode_qr_video(
        &self,
        qr_images: &[DynamicImage],
        output_path: &str,
        codec: Codec,
    ) -> Result<VideoStats> {
        // In a real implementation, this would use FFmpeg to create an actual video
        // For now, we'll create a simple image sequence and metadata with optimized I/O

        tracing::info!("Encoding {} frames to {} using {:?}", qr_images.len(), output_path, codec);

        let codec_config = self.config.get_codec_config(codec.as_str())?;
        let start_time = std::time::Instant::now();

        // Create output directory if it doesn't exist
        if let Some(parent) = Path::new(output_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // For this simplified version, we'll save the images as a sequence
        // and create a metadata file that represents the "video"
        let base_path = Path::new(output_path).with_extension("");
        let frames_dir = format!("{}_frames", base_path.to_string_lossy());
        std::fs::create_dir_all(&frames_dir)?;

        // Save frames in parallel batches for better performance
        self.save_frames_optimized(qr_images, &frames_dir).await?;
        
        // Create a simple "video" metadata file
        let video_metadata = VideoMetadata {
            frame_count: qr_images.len() as u32,
            fps: codec_config.fps,
            width: codec_config.width,
            height: codec_config.height,
            codec: codec.as_str().to_string(),
            frames_dir: frames_dir.clone(),
        };
        
        let metadata_json = serde_json::to_string_pretty(&video_metadata)?;
        std::fs::write(output_path, metadata_json)?;
        
        let encoding_time = start_time.elapsed();
        let file_size = std::fs::metadata(output_path)?.len();
        
        Ok(VideoStats {
            frame_count: qr_images.len() as u32,
            duration_seconds: qr_images.len() as f64 / codec_config.fps,
            file_size_bytes: file_size,
            encoding_time_seconds: encoding_time.as_secs_f64(),
            codec: codec.as_str().to_string(),
            fps: codec_config.fps,
            width: codec_config.width,
            height: codec_config.height,
        })
    }

    /// Save frames in optimized batches with progress reporting
    async fn save_frames_optimized(
        &self,
        qr_images: &[DynamicImage],
        frames_dir: &str,
    ) -> Result<()> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let total_frames = qr_images.len();
        let batch_size = std::cmp::min(100, std::cmp::max(10, total_frames / 20)); // Adaptive batch size
        let progress_counter = Arc::new(AtomicUsize::new(0));

        tracing::info!("Saving {} frames in batches of {}", total_frames, batch_size);

        // Process frames in parallel batches
        let results: Result<Vec<_>> = qr_images
            .par_chunks(batch_size)
            .enumerate()
            .map(|(batch_idx, batch)| {
                let batch_start = batch_idx * batch_size;
                let progress = Arc::clone(&progress_counter);

                // Save each frame in the batch
                for (i, image) in batch.iter().enumerate() {
                    let frame_idx = batch_start + i;
                    let frame_path = format!("{}/frame_{:06}.png", frames_dir, frame_idx);

                    // Use JPEG for better compression and speed on intermediate frames
                    // PNG is more reliable but slower - keeping PNG for compatibility
                    if let Err(e) = image.save(&frame_path) {
                        return Err(crate::error::MemvidError::video(format!(
                            "Failed to save frame {}: {}", frame_idx, e
                        )));
                    }

                    // Update progress
                    let completed = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if completed % 100 == 0 || completed == total_frames {
                        tracing::info!("Saved {}/{} frames ({:.1}%)",
                            completed, total_frames,
                            (completed as f64 / total_frames as f64) * 100.0);
                    }
                }

                Ok(())
            })
            .collect();

        results?;
        tracing::info!("Successfully saved all {} frames", total_frames);
        Ok(())
    }
}

/// Video decoder for reading QR code videos
pub struct VideoDecoder {
    config: Config,
}

impl VideoDecoder {
    /// Create a new video decoder
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Create decoder with default configuration
    pub fn default() -> Self {
        Self::new(Config::default())
    }

    /// Extract frame from video at specified frame number (simplified implementation)
    pub async fn extract_frame(&self, video_path: &str, frame_number: u32) -> Result<DynamicImage> {
        // In a real implementation, this would use FFmpeg to extract frames
        // For now, we'll read from our image sequence
        
        // Read the video metadata
        let metadata_content = std::fs::read_to_string(video_path)?;
        let video_metadata: VideoMetadata = serde_json::from_str(&metadata_content)?;
        
        if frame_number >= video_metadata.frame_count {
            return Err(MemvidError::video(format!(
                "Frame {} not found (total frames: {})", 
                frame_number, 
                video_metadata.frame_count
            )));
        }
        
        // Load the specific frame
        let frame_path = format!("{}/frame_{:06}.png", video_metadata.frames_dir, frame_number);
        let image = image::open(&frame_path)
            .map_err(|e| MemvidError::video(format!("Failed to load frame {}: {}", frame_number, e)))?;
        
        Ok(image)
    }

    /// Get video information (simplified implementation)
    pub async fn get_video_info(&self, video_path: &str) -> Result<VideoInfo> {
        // Read the video metadata
        let metadata_content = std::fs::read_to_string(video_path)?;
        let video_metadata: VideoMetadata = serde_json::from_str(&metadata_content)?;
        
        Ok(VideoInfo {
            width: video_metadata.width,
            height: video_metadata.height,
            fps: video_metadata.fps,
            duration_seconds: video_metadata.frame_count as f64 / video_metadata.fps,
            total_frames: video_metadata.frame_count,
            codec: video_metadata.codec,
            pixel_format: "rgb24".to_string(), // Simplified
        })
    }
}

/// Video encoding statistics
#[derive(Debug, Clone)]
pub struct VideoStats {
    pub frame_count: u32,
    pub duration_seconds: f64,
    pub file_size_bytes: u64,
    pub encoding_time_seconds: f64,
    pub codec: String,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
}

/// Video information
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_seconds: f64,
    pub total_frames: u32,
    pub codec: String,
    pub pixel_format: String,
}

/// Internal video metadata structure
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct VideoMetadata {
    pub frame_count: u32,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub codec: String,
    pub frames_dir: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_video_encoding() {
        let encoder = VideoEncoder::default();
        assert_eq!(Codec::H264.as_str(), "h264");
    }

    #[test]
    fn test_codec_parsing() {
        assert_eq!(Codec::from_str("h264").unwrap(), Codec::H264);
        assert_eq!(Codec::from_str("h265").unwrap(), Codec::H265);
        assert_eq!(Codec::from_str("av1").unwrap(), Codec::Av1);
        assert!(Codec::from_str("invalid").is_err());
    }
}
