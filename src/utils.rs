//! Utility functions and helpers

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{MemvidError, Result};

/// Generate a unique filename with timestamp
pub fn generate_unique_filename(base_name: &str, extension: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    format!("{}_{}.{}", base_name, timestamp, extension)
}

/// Ensure directory exists, create if it doesn't
pub fn ensure_directory_exists<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get file extension from path
pub fn get_file_extension<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// Check if file has specific extension
pub fn has_extension<P: AsRef<Path>>(path: P, extension: &str) -> bool {
    get_file_extension(path)
        .map(|ext| ext == extension.to_lowercase())
        .unwrap_or(false)
}

/// Format file size in human-readable format
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: f64 = 1024.0;
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Format duration in human-readable format
pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let remaining_seconds = seconds % 60.0;
        format!("{}m {:.1}s", minutes, remaining_seconds)
    } else {
        let hours = (seconds / 3600.0).floor();
        let remaining_minutes = ((seconds % 3600.0) / 60.0).floor();
        let remaining_seconds = seconds % 60.0;
        format!("{}h {}m {:.1}s", hours, remaining_minutes, remaining_seconds)
    }
}

/// Calculate compression ratio
pub fn calculate_compression_ratio(original_size: u64, compressed_size: u64) -> f64 {
    if original_size == 0 {
        return 0.0;
    }
    
    compressed_size as f64 / original_size as f64
}

/// Validate video file path
pub fn validate_video_path<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Err(MemvidError::file_not_found(path.to_string_lossy()));
    }
    
    let valid_extensions = ["mp4", "avi", "mov", "mkv", "webm"];
    let extension = get_file_extension(path);
    
    match extension {
        Some(ext) if valid_extensions.contains(&ext.as_str()) => Ok(()),
        Some(ext) => Err(MemvidError::unsupported_format(format!("Video format: {}", ext))),
        None => Err(MemvidError::unsupported_format("No file extension")),
    }
}

/// Validate index file path
pub fn validate_index_path<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Err(MemvidError::file_not_found(path.to_string_lossy()));
    }
    
    let valid_extensions = ["json", "metadata"];
    let extension = get_file_extension(path);
    
    match extension {
        Some(ext) if valid_extensions.contains(&ext.as_str()) => Ok(()),
        Some(ext) => Err(MemvidError::unsupported_format(format!("Index format: {}", ext))),
        None => Err(MemvidError::unsupported_format("No file extension")),
    }
}

/// Progress reporter for long-running operations
pub struct ProgressReporter {
    total: usize,
    current: usize,
    last_reported: usize,
    report_interval: usize,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            last_reported: 0,
            report_interval: std::cmp::max(1, total / 100), // Report every 1%
        }
    }

    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current;
        
        if current - self.last_reported >= self.report_interval || current == self.total {
            self.report();
            self.last_reported = current;
        }
    }

    /// Increment progress by 1
    pub fn increment(&mut self) {
        self.update(self.current + 1);
    }

    /// Report current progress
    fn report(&self) {
        let percentage = if self.total > 0 {
            (self.current as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };
        
        tracing::info!("Progress: {}/{} ({:.1}%)", self.current, self.total, percentage);
    }

    /// Get current progress as percentage
    pub fn percentage(&self) -> f64 {
        if self.total > 0 {
            (self.current as f64 / self.total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

/// Simple timer for measuring elapsed time
pub struct Timer {
    start_time: std::time::Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    /// Get elapsed time in seconds
    pub fn elapsed_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get elapsed time formatted as string
    pub fn elapsed_formatted(&self) -> String {
        format_duration(self.elapsed_seconds())
    }
}

/// Memory usage tracker
pub struct MemoryTracker {
    initial_memory: u64,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            initial_memory: Self::get_memory_usage(),
        }
    }

    /// Get current memory usage in bytes
    pub fn get_memory_usage() -> u64 {
        // This is a simplified implementation
        // In a real implementation, you might use system-specific APIs
        0 // Placeholder
    }

    /// Get memory usage since creation
    pub fn memory_delta(&self) -> i64 {
        Self::get_memory_usage() as i64 - self.initial_memory as i64
    }

    /// Get formatted memory delta
    pub fn memory_delta_formatted(&self) -> String {
        let delta = self.memory_delta();
        if delta >= 0 {
            format!("+{}", format_file_size(delta as u64))
        } else {
            format!("-{}", format_file_size((-delta) as u64))
        }
    }
}

/// Batch processor for handling large collections
pub struct BatchProcessor<T> {
    items: Vec<T>,
    batch_size: usize,
}

impl<T> BatchProcessor<T> {
    /// Create a new batch processor
    pub fn new(items: Vec<T>, batch_size: usize) -> Self {
        Self { items, batch_size }
    }

    /// Process items in batches
    pub async fn process<F, R, Fut>(&self, mut processor: F) -> Vec<R>
    where
        F: FnMut(&[T]) -> Fut,
        Fut: std::future::Future<Output = Vec<R>>,
    {
        let mut results = Vec::new();
        
        for batch in self.items.chunks(self.batch_size) {
            let batch_results = processor(batch).await;
            results.extend(batch_results);
        }
        
        results
    }

    /// Get number of batches
    pub fn batch_count(&self) -> usize {
        (self.items.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get total number of items
    pub fn total_items(&self) -> usize {
        self.items.len()
    }
}

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate that required environment variables are set
    pub fn validate_env_vars(required_vars: &[&str]) -> Result<()> {
        let mut missing_vars = Vec::new();
        
        for var in required_vars {
            if std::env::var(var).is_err() {
                missing_vars.push(*var);
            }
        }
        
        if !missing_vars.is_empty() {
            return Err(MemvidError::config(format!(
                "Missing required environment variables: {}",
                missing_vars.join(", ")
            )));
        }
        
        Ok(())
    }

    /// Validate file paths exist
    pub fn validate_paths(paths: &[&str]) -> Result<()> {
        for path in paths {
            if !Path::new(path).exists() {
                return Err(MemvidError::file_not_found(*path));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_file_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.5), "30.5s");
        assert_eq!(format_duration(90.0), "1m 30.0s");
        assert_eq!(format_duration(3661.0), "1h 1m 1.0s");
    }

    #[test]
    fn test_compression_ratio() {
        assert_eq!(calculate_compression_ratio(1000, 500), 0.5);
        assert_eq!(calculate_compression_ratio(0, 100), 0.0);
        assert_eq!(calculate_compression_ratio(100, 100), 1.0);
    }

    #[test]
    fn test_get_file_extension() {
        assert_eq!(get_file_extension("test.mp4"), Some("mp4".to_string()));
        assert_eq!(get_file_extension("test.PDF"), Some("pdf".to_string()));
        assert_eq!(get_file_extension("test"), None);
        assert_eq!(get_file_extension("test."), Some("".to_string()));
    }

    #[test]
    fn test_progress_reporter() {
        let mut reporter = ProgressReporter::new(100);
        assert_eq!(reporter.percentage(), 0.0);
        assert!(!reporter.is_complete());
        
        reporter.update(50);
        assert_eq!(reporter.percentage(), 50.0);
        assert!(!reporter.is_complete());
        
        reporter.update(100);
        assert_eq!(reporter.percentage(), 100.0);
        assert!(reporter.is_complete());
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_seconds() > 0.0);
    }

    #[test]
    fn test_batch_processor_basic() {
        let items = vec![1, 2, 3, 4, 5];
        let processor = BatchProcessor::new(items, 2);

        assert_eq!(processor.batch_count(), 3);
        assert_eq!(processor.total_items(), 5);
    }
}
