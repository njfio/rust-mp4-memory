//! Folder processing for recursive file discovery and filtering

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::fs;
use walkdir::{WalkDir, DirEntry};
use glob::Pattern;
use ignore::WalkBuilder;

use crate::config::{Config, FolderConfig};
use crate::data::DataFileType;
use crate::error::{MemvidError, Result};

/// Statistics about folder processing
#[derive(Debug, Clone, Default)]
pub struct FolderStats {
    /// Total directories scanned
    pub directories_scanned: usize,
    /// Total files found
    pub files_found: usize,
    /// Files processed successfully
    pub files_processed: usize,
    /// Files skipped (filtered out)
    pub files_skipped: usize,
    /// Files failed to process
    pub files_failed: usize,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Reason why a file was skipped
#[derive(Debug, Clone, PartialEq)]
pub enum SkipReason {
    /// File extension not supported
    UnsupportedExtension,
    /// File extension explicitly excluded
    ExcludedExtension,
    /// File matches exclude pattern
    ExcludedPattern,
    /// File too small
    TooSmall,
    /// File too large
    TooLarge,
    /// File is binary
    Binary,
    /// File is hidden
    Hidden,
    /// Depth limit exceeded
    DepthExceeded,
    /// Permission denied
    PermissionDenied,
    /// File is a directory
    IsDirectory,
    /// Symlink not followed
    SymlinkNotFollowed,
}

/// Information about a discovered file
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Full path to the file
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// File extension (lowercase)
    pub extension: String,
    /// Detected file type
    pub file_type: Option<DataFileType>,
    /// Depth in directory tree
    pub depth: usize,
    /// Whether file is hidden
    pub is_hidden: bool,
    /// Whether file is a symlink
    pub is_symlink: bool,
}

/// Folder processor for recursive file discovery
pub struct FolderProcessor {
    config: FolderConfig,
    supported_extensions: HashSet<String>,
    exclude_patterns: Vec<Pattern>,
}

impl FolderProcessor {
    /// Create a new folder processor with configuration
    pub fn new(config: FolderConfig) -> Result<Self> {
        // Build set of supported extensions
        let mut supported_extensions = HashSet::new();
        
        // Add all supported data file extensions
        for ext in &["csv", "parquet", "pq", "json", "jsonl", "rs", "js", "jsx", "ts", "tsx", 
                     "py", "pyw", "html", "htm", "css", "scss", "sass", "log", "logs", 
                     "yaml", "yml", "toml", "xml", "sql", "md", "markdown", "txt", "pdf", "epub"] {
            supported_extensions.insert(ext.to_string());
        }
        
        // Compile exclude patterns
        let mut exclude_patterns = Vec::new();
        for pattern_str in &config.exclude_patterns {
            match Pattern::new(pattern_str) {
                Ok(pattern) => exclude_patterns.push(pattern),
                Err(e) => {
                    log::warn!("Invalid exclude pattern '{}': {}", pattern_str, e);
                }
            }
        }
        
        Ok(Self {
            config,
            supported_extensions,
            exclude_patterns,
        })
    }
    
    /// Create processor with default configuration
    pub fn default() -> Self {
        Self::new(Config::default().folder).unwrap()
    }
    
    /// Discover files in a directory recursively
    pub fn discover_files<P: AsRef<Path>>(&self, root_path: P) -> Result<Vec<FileInfo>> {
        let root_path = root_path.as_ref();
        
        if !root_path.exists() {
            return Err(MemvidError::file_not_found(root_path.to_string_lossy().as_ref()));
        }
        
        if !root_path.is_dir() {
            return Err(MemvidError::generic(format!("Path is not a directory: {}", root_path.display())));
        }
        
        let mut files = Vec::new();
        
        // Use ignore crate for better gitignore support if available
        if self.should_use_ignore_crate(root_path) {
            files.extend(self.discover_with_ignore(root_path)?);
        } else {
            files.extend(self.discover_with_walkdir(root_path)?);
        }
        
        // Apply additional filtering
        let filtered_files: Vec<FileInfo> = files.into_iter()
            .filter(|file| self.should_include_file(file).is_none())
            .collect();
        
        log::info!("Discovered {} files in {}", filtered_files.len(), root_path.display());
        
        Ok(filtered_files)
    }
    
    /// Check if we should use the ignore crate (if .gitignore exists)
    fn should_use_ignore_crate(&self, root_path: &Path) -> bool {
        root_path.join(".gitignore").exists() || 
        root_path.join(".ignore").exists()
    }
    
    /// Discover files using the ignore crate (respects .gitignore)
    fn discover_with_ignore(&self, root_path: &Path) -> Result<Vec<FileInfo>> {
        let mut files = Vec::new();
        
        let mut builder = WalkBuilder::new(root_path);
        builder
            .follow_links(self.config.follow_symlinks)
            .hidden(!self.config.include_hidden)
            .git_ignore(true)
            .git_exclude(true)
            .git_global(true);
        
        if let Some(max_depth) = self.config.max_depth {
            builder.max_depth(Some(max_depth));
        }
        
        for result in builder.build() {
            match result {
                Ok(entry) => {
                    if let Some(file_info) = self.process_dir_entry_ignore(&entry)? {
                        files.push(file_info);
                    }
                }
                Err(e) => {
                    log::warn!("Error walking directory: {}", e);
                }
            }
        }
        
        Ok(files)
    }
    
    /// Discover files using walkdir
    fn discover_with_walkdir(&self, root_path: &Path) -> Result<Vec<FileInfo>> {
        let mut files = Vec::new();
        
        let mut walker = WalkDir::new(root_path)
            .follow_links(self.config.follow_symlinks);
        
        if let Some(max_depth) = self.config.max_depth {
            walker = walker.max_depth(max_depth);
        }
        
        for entry in walker {
            match entry {
                Ok(entry) => {
                    if let Some(file_info) = self.process_dir_entry_walkdir(&entry)? {
                        files.push(file_info);
                    }
                }
                Err(e) => {
                    log::warn!("Error walking directory: {}", e);
                }
            }
        }
        
        Ok(files)
    }
    
    /// Process a directory entry from ignore crate
    fn process_dir_entry_ignore(&self, entry: &ignore::DirEntry) -> Result<Option<FileInfo>> {
        let path = entry.path();
        
        if !path.is_file() {
            return Ok(None);
        }
        
        self.create_file_info(path, entry.depth())
    }
    
    /// Process a directory entry from walkdir
    fn process_dir_entry_walkdir(&self, entry: &DirEntry) -> Result<Option<FileInfo>> {
        let path = entry.path();
        
        if !path.is_file() {
            return Ok(None);
        }
        
        self.create_file_info(path, entry.depth())
    }
    
    /// Create FileInfo from path and depth
    fn create_file_info(&self, path: &Path, depth: usize) -> Result<Option<FileInfo>> {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(_) => return Ok(None), // Skip files we can't read
        };
        
        let size = metadata.len();
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let file_type = DataFileType::from_extension(&extension);
        let is_hidden = path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with('.'))
            .unwrap_or(false);
        
        let is_symlink = metadata.file_type().is_symlink();
        
        let file_info = FileInfo {
            path: path.to_path_buf(),
            size,
            extension,
            file_type,
            depth,
            is_hidden,
            is_symlink,
        };
        
        Ok(Some(file_info))
    }
    
    /// Check if a file should be included (returns skip reason if not)
    pub fn should_include_file(&self, file: &FileInfo) -> Option<SkipReason> {
        // Check depth limit
        if let Some(max_depth) = self.config.max_depth {
            if file.depth > max_depth {
                return Some(SkipReason::DepthExceeded);
            }
        }
        
        // Check hidden files
        if file.is_hidden && !self.config.include_hidden {
            return Some(SkipReason::Hidden);
        }
        
        // Check symlinks
        if file.is_symlink && !self.config.follow_symlinks {
            return Some(SkipReason::SymlinkNotFollowed);
        }
        
        // Check file size
        if file.size < self.config.min_file_size as u64 {
            return Some(SkipReason::TooSmall);
        }
        
        if file.size > self.config.max_file_size as u64 {
            return Some(SkipReason::TooLarge);
        }
        
        // Check excluded extensions
        if self.config.exclude_extensions.contains(&file.extension) {
            return Some(SkipReason::ExcludedExtension);
        }
        
        // Check include extensions filter
        if let Some(ref include_extensions) = self.config.include_extensions {
            if !include_extensions.contains(&file.extension) {
                return Some(SkipReason::UnsupportedExtension);
            }
        } else {
            // If no include filter, check if extension is supported
            if file.file_type.is_none() && !self.supported_extensions.contains(&file.extension) {
                return Some(SkipReason::UnsupportedExtension);
            }
        }
        
        // Check exclude patterns
        let path_str = file.path.to_string_lossy();
        for pattern in &self.exclude_patterns {
            if pattern.matches(&path_str) {
                return Some(SkipReason::ExcludedPattern);
            }
        }
        
        // Check if file is binary (if skip_binary is enabled)
        if self.config.skip_binary && self.is_binary_file(&file.path) {
            return Some(SkipReason::Binary);
        }
        
        None // File should be included
    }
    
    /// Check if a file is binary by examining its content
    fn is_binary_file(&self, path: &Path) -> bool {
        // Read first 8KB to check for binary content
        match fs::read(path) {
            Ok(content) => {
                let sample_size = std::cmp::min(content.len(), 8192);
                let sample = &content[..sample_size];
                
                // Check for null bytes (common in binary files)
                if sample.contains(&0) {
                    return true;
                }
                
                // Check for high ratio of non-printable characters
                let non_printable = sample.iter()
                    .filter(|&&b| b < 32 && b != 9 && b != 10 && b != 13) // Tab, LF, CR are OK
                    .count();
                
                let ratio = non_printable as f64 / sample.len() as f64;
                ratio > 0.3 // More than 30% non-printable = likely binary
            }
            Err(_) => false, // If we can't read it, assume it's not binary
        }
    }
    
    /// Get configuration
    pub fn config(&self) -> &FolderConfig {
        &self.config
    }
    
    /// Get supported extensions
    pub fn supported_extensions(&self) -> &HashSet<String> {
        &self.supported_extensions
    }
}
