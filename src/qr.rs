//! QR code encoding and decoding functionality

use image::{DynamicImage, RgbImage};
use qrcode::QrCode;
use rqrr::PreparedImage;
use serde_json;
use std::io::Cursor;

use crate::config::{Config, QrConfig};
use crate::error::{MemvidError, Result};

/// QR code processor for encoding and decoding text data
pub struct QrProcessor {
    config: QrConfig,
}

impl QrProcessor {
    /// Create a new QR processor with the given configuration
    pub fn new(config: QrConfig) -> Self {
        Self { config }
    }

    /// Create a QR processor with default configuration
    pub fn default() -> Self {
        Self::new(Config::default().qr)
    }

    /// Encode text data into a QR code image
    pub fn encode_to_qr(&self, data: &str) -> Result<DynamicImage> {
        // Calculate QR code capacity based on error correction level
        let max_capacity = self.get_max_qr_capacity()?;

        // Compress data if it's large
        let processed_data = if data.len() > 100 {
            let compressed = self.compress_data(data)?;
            let compressed_with_prefix = format!("GZ:{}", compressed);

            // Check if compressed data still fits
            if compressed_with_prefix.len() > max_capacity {
                return Err(MemvidError::qr_code(format!(
                    "Data too large for QR code: {} bytes (max: {} bytes). Consider reducing chunk size.",
                    compressed_with_prefix.len(),
                    max_capacity
                )));
            }

            compressed_with_prefix
        } else {
            if data.len() > max_capacity {
                return Err(MemvidError::qr_code(format!(
                    "Data too large for QR code: {} bytes (max: {} bytes). Consider reducing chunk size.",
                    data.len(),
                    max_capacity
                )));
            }
            data.to_string()
        };

        // Create QR code with error handling
        let qr_code = match QrCode::with_error_correction_level(
            &processed_data,
            self.get_error_correction_level()?,
        ) {
            Ok(qr) => qr,
            Err(e) => {
                return Err(MemvidError::qr_code(format!(
                    "Failed to create QR code: {}. Data size: {} bytes. Try reducing chunk size or using lower error correction.",
                    e, processed_data.len()
                )));
            }
        };

        // Use the string renderer which is simple and works
        let qr_string = qr_code.render::<char>()
            .quiet_zone(false)
            .module_dimensions(1, 1)
            .build();

        // Parse the string representation into an image
        let lines: Vec<&str> = qr_string.lines().collect();
        let height = lines.len() as u32;
        let width = if height > 0 { lines[0].chars().count() as u32 } else { 0 };

        // Scale up the image for better visibility
        let scale = 8;
        let scaled_width = width * scale;
        let scaled_height = height * scale;

        let mut img_buffer = image::ImageBuffer::new(scaled_width, scaled_height);

        for (y, line) in lines.iter().enumerate() {
            for (x, ch) in line.chars().enumerate() {
                let color = if ch == '█' || ch == '▀' || ch == '▄' || ch == '▌' || ch == '▐' || ch == '▆' || ch == '▇' || ch == '■' {
                    image::Luma([0u8]) // Black
                } else {
                    image::Luma([255u8]) // White
                };

                // Scale up the pixel
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x as u32 * scale + dx;
                        let py = y as u32 * scale + dy;
                        if px < scaled_width && py < scaled_height {
                            img_buffer.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }

        let image = image::DynamicImage::ImageLuma8(img_buffer);

        // Convert to RGB for consistency
        let rgb_image = image.to_rgb8();
        Ok(DynamicImage::ImageRgb8(rgb_image))
    }

    /// Decode QR code from image
    pub fn decode_qr(&self, image: &DynamicImage) -> Result<String> {
        // Convert to grayscale for better detection
        let gray_image = image.to_luma8();
        
        // Prepare image for QR detection
        let mut prepared_img = PreparedImage::prepare(gray_image);
        
        // Find and decode QR codes
        let grids = prepared_img.detect_grids();
        
        if grids.is_empty() {
            return Err(MemvidError::qr_code("No QR code found in image"));
        }

        // Try to decode the first QR code found
        let grid = &grids[0];
        let (_meta, content) = grid.decode()?;

        // content is already a String from rqrr
        let decoded_content = content;

        // Check if data was compressed
        if decoded_content.starts_with("GZ:") {
            let compressed_data = &decoded_content[3..];
            self.decompress_data(compressed_data)
        } else {
            Ok(decoded_content)
        }
    }

    /// Convert QR image to video frame with specified dimensions
    pub fn qr_to_frame(&self, qr_image: &DynamicImage, frame_width: u32, frame_height: u32) -> Result<RgbImage> {
        // Resize image to fit frame while maintaining aspect ratio
        let resized = qr_image.resize_exact(
            frame_width,
            frame_height,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB
        let rgb_image = resized.to_rgb8();
        Ok(rgb_image)
    }

    /// Extract frame from video at specified frame number (simplified implementation)
    pub fn extract_frame_from_video(&self, video_path: &str, frame_number: u32) -> Result<DynamicImage> {
        // In a real implementation, this would use OpenCV or FFmpeg
        // For now, we'll use our simplified video decoder
        use crate::video::VideoDecoder;

        let decoder = VideoDecoder::default();
        // This is a blocking call in an async context, but it's simplified
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                decoder.extract_frame(video_path, frame_number).await
            })
        })
    }

    /// Compress data using gzip and base64 encoding
    fn compress_data(&self, data: &str) -> Result<String> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data.as_bytes())?;
        let compressed = encoder.finish()?;
        
        Ok(base64::encode(compressed))
    }

    /// Decompress data from base64 and gzip
    fn decompress_data(&self, compressed_data: &str) -> Result<String> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let compressed_bytes = base64::decode(compressed_data)
            .map_err(|e| MemvidError::qr_code(format!("Base64 decode error: {}", e)))?;
        
        let mut decoder = GzDecoder::new(&compressed_bytes[..]);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed)?;
        
        Ok(decompressed)
    }

    /// Convert error correction level string to qrcode enum
    fn get_error_correction_level(&self) -> Result<qrcode::EcLevel> {
        match self.config.error_correction.as_str() {
            "L" => Ok(qrcode::EcLevel::L),
            "M" => Ok(qrcode::EcLevel::M),
            "Q" => Ok(qrcode::EcLevel::Q),
            "H" => Ok(qrcode::EcLevel::H),
            _ => Err(MemvidError::config(format!(
                "Invalid error correction level: {}",
                self.config.error_correction
            ))),
        }
    }

    /// Get maximum QR code capacity in bytes for current error correction level
    fn get_max_qr_capacity(&self) -> Result<usize> {
        // QR code capacity for alphanumeric data (conservative estimate)
        // These are approximate values for Version 40 (largest) QR codes
        match self.config.error_correction.as_str() {
            "L" => Ok(4296), // Low error correction - highest capacity
            "M" => Ok(3391), // Medium error correction
            "Q" => Ok(2420), // Quartile error correction
            "H" => Ok(1852), // High error correction - lowest capacity
            _ => Ok(2000),   // Default conservative estimate
        }
    }

    /// Get recommended chunk size for QR codes
    pub fn get_recommended_chunk_size(&self) -> Result<usize> {
        let max_capacity = self.get_max_qr_capacity()?;
        // Leave room for compression overhead and metadata
        // Use 70% of capacity to be safe
        Ok((max_capacity as f64 * 0.7) as usize)
    }

    // OpenCV support removed in simplified version
}

/// Batch process multiple QR codes
pub struct BatchQrProcessor {
    processor: QrProcessor,
}

impl BatchQrProcessor {
    /// Create a new batch processor
    pub fn new(config: QrConfig) -> Self {
        Self {
            processor: QrProcessor::new(config),
        }
    }

    /// Encode multiple text chunks to QR codes in parallel with progress reporting
    pub async fn encode_batch(&self, chunks: &[String]) -> Result<Vec<DynamicImage>> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let total_chunks = chunks.len();
        let progress_counter = Arc::new(AtomicUsize::new(0));

        tracing::info!("Encoding {} chunks to QR codes in parallel", total_chunks);

        let results: Result<Vec<_>> = chunks
            .par_iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let progress = Arc::clone(&progress_counter);

                let result = self.processor.encode_to_qr(chunk);

                // Update progress
                let completed = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if completed % 50 == 0 || completed == total_chunks {
                    tracing::info!("Encoded {}/{} QR codes ({:.1}%)",
                        completed, total_chunks,
                        (completed as f64 / total_chunks as f64) * 100.0);
                }

                result.map_err(|e| {
                    crate::error::MemvidError::qr_code(format!(
                        "Failed to encode chunk {} (length: {}): {}",
                        idx, chunk.len(), e
                    ))
                })
            })
            .collect();

        let qr_images = results?;
        tracing::info!("Successfully encoded all {} chunks to QR codes", total_chunks);
        Ok(qr_images)
    }

    /// Decode multiple QR codes in parallel
    pub async fn decode_batch(&self, images: &[DynamicImage]) -> Result<Vec<String>> {
        use rayon::prelude::*;

        let results: Result<Vec<_>> = images
            .par_iter()
            .map(|image| self.processor.decode_qr(image))
            .collect();

        results
    }

    /// Extract and decode frames from video in parallel
    pub async fn extract_and_decode_frames(
        &self,
        video_path: &str,
        frame_numbers: &[u32],
    ) -> Result<Vec<(u32, Option<String>)>> {
        use rayon::prelude::*;

        let results: Vec<_> = frame_numbers
            .par_iter()
            .map(|&frame_num| {
                let result = self.processor
                    .extract_frame_from_video(video_path, frame_num)
                    .and_then(|image| self.processor.decode_qr(&image));
                
                match result {
                    Ok(decoded) => (frame_num, Some(decoded)),
                    Err(_) => (frame_num, None),
                }
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_encode_decode() {
        let processor = QrProcessor::default();
        let test_data = "Hello, World!";
        
        let qr_image = processor.encode_to_qr(test_data).unwrap();
        let decoded = processor.decode_qr(&qr_image).unwrap();
        
        assert_eq!(test_data, decoded);
    }

    #[test]
    fn test_compression() {
        let processor = QrProcessor::default();
        let long_text = "This is a very long text that should be compressed when encoded into a QR code because it exceeds the 100 character threshold that we have set for compression.";
        
        let qr_image = processor.encode_to_qr(long_text).unwrap();
        let decoded = processor.decode_qr(&qr_image).unwrap();
        
        assert_eq!(long_text, decoded);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let processor = BatchQrProcessor::new(Config::default().qr);
        let chunks = vec![
            "Chunk 1".to_string(),
            "Chunk 2".to_string(),
            "Chunk 3".to_string(),
        ];
        
        let qr_images = processor.encode_batch(&chunks).await.unwrap();
        assert_eq!(qr_images.len(), 3);
        
        let decoded = processor.decode_batch(&qr_images).await.unwrap();
        assert_eq!(decoded, chunks);
    }
}
