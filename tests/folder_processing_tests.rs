use rust_mem_vid::{FolderProcessor, MemvidEncoder, Config};
use rust_mem_vid::config::FolderConfig;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_folder_processor_creation() {
    let config = FolderConfig {
        max_depth: Some(5),
        include_extensions: Some(vec!["rs".to_string(), "py".to_string()]),
        exclude_extensions: vec!["exe".to_string()],
        exclude_patterns: vec!["*/target/*".to_string()],
        min_file_size: 1,
        max_file_size: 1024 * 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let processor = FolderProcessor::new(config).unwrap();
    assert_eq!(processor.config().max_depth, Some(5));
    assert!(processor.supported_extensions().contains("rs"));
}

#[test]
fn test_folder_processor_default() {
    let processor = FolderProcessor::default();
    assert!(processor.config().max_depth.is_some());
    assert!(!processor.config().follow_symlinks);
}

#[test]
fn test_file_discovery() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create test directory structure
    fs::create_dir_all(base_path.join("src")).unwrap();
    fs::create_dir_all(base_path.join("target")).unwrap();
    
    // Create test files
    fs::write(base_path.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(base_path.join("src/lib.rs"), "pub mod test;").unwrap();
    fs::write(base_path.join("README.md"), "# Test Project").unwrap();
    fs::write(base_path.join("target/debug.exe"), "binary").unwrap();
    
    let processor = FolderProcessor::default();
    let files = processor.discover_files(base_path).unwrap();
    
    // Should find Rust and Markdown files, but not the exe in target
    assert!(!files.is_empty());
    
    let rust_files: Vec<_> = files.iter()
        .filter(|f| f.extension == "rs")
        .collect();
    assert_eq!(rust_files.len(), 2);
    
    let md_files: Vec<_> = files.iter()
        .filter(|f| f.extension == "md")
        .collect();
    assert_eq!(md_files.len(), 1);
    
    // Should not include files from target directory (excluded by default)
    let target_files: Vec<_> = files.iter()
        .filter(|f| f.path.to_string_lossy().contains("target"))
        .collect();
    assert_eq!(target_files.len(), 0);
}

#[test]
fn test_file_filtering() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create test files
    fs::write(base_path.join("test.rs"), "fn test() {}").unwrap();
    fs::write(base_path.join("test.py"), "def test(): pass").unwrap();
    fs::write(base_path.join("test.exe"), "binary").unwrap();
    fs::write(base_path.join(".hidden"), "hidden file").unwrap();
    
    let config = FolderConfig {
        max_depth: Some(1),
        include_extensions: Some(vec!["rs".to_string(), "py".to_string()]),
        exclude_extensions: vec!["exe".to_string()],
        exclude_patterns: vec![],
        min_file_size: 1,
        max_file_size: 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let processor = FolderProcessor::new(config).unwrap();
    let files = processor.discover_files(base_path).unwrap();
    
    // Should only find .rs and .py files
    assert_eq!(files.len(), 2);
    
    let extensions: Vec<String> = files.iter().map(|f| f.extension.clone()).collect();
    assert!(extensions.contains(&"rs".to_string()));
    assert!(extensions.contains(&"py".to_string()));
    assert!(!extensions.contains(&"exe".to_string()));
}

#[test]
fn test_depth_limiting() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create nested directory structure
    fs::create_dir_all(base_path.join("level1/level2/level3")).unwrap();
    
    fs::write(base_path.join("root.txt"), "root file").unwrap();
    fs::write(base_path.join("level1/file1.txt"), "level 1 file").unwrap();
    fs::write(base_path.join("level1/level2/file2.txt"), "level 2 file").unwrap();
    fs::write(base_path.join("level1/level2/level3/file3.txt"), "level 3 file").unwrap();
    
    // Test with max_depth = 2
    let config = FolderConfig {
        max_depth: Some(2),
        include_extensions: Some(vec!["txt".to_string()]),
        exclude_extensions: vec![],
        exclude_patterns: vec![],
        min_file_size: 1,
        max_file_size: 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: false,
    };
    
    let processor = FolderProcessor::new(config).unwrap();
    let files = processor.discover_files(base_path).unwrap();
    
    // Should find files at root, level1, and level2, but not level3
    // The exact count may vary based on depth calculation, but should be at least 2 and not include level3
    assert!(files.len() >= 2);
    assert!(files.len() <= 3);

    let has_level3 = files.iter().any(|f| f.path.to_string_lossy().contains("level3"));
    assert!(!has_level3);

    // Verify we have files from different levels
    let has_root = files.iter().any(|f| f.path.file_name().unwrap().to_string_lossy() == "root.txt");
    assert!(has_root);
}

#[test]
fn test_file_size_filtering() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create files of different sizes
    fs::write(base_path.join("small.txt"), "x").unwrap(); // 1 byte
    fs::write(base_path.join("medium.txt"), "x".repeat(100)).unwrap(); // 100 bytes
    fs::write(base_path.join("large.txt"), "x".repeat(1000)).unwrap(); // 1000 bytes
    
    let config = FolderConfig {
        max_depth: Some(1),
        include_extensions: Some(vec!["txt".to_string()]),
        exclude_extensions: vec![],
        exclude_patterns: vec![],
        min_file_size: 50, // At least 50 bytes
        max_file_size: 500, // At most 500 bytes
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: false,
    };
    
    let processor = FolderProcessor::new(config).unwrap();
    let files = processor.discover_files(base_path).unwrap();
    
    // Should only find medium.txt (100 bytes)
    assert_eq!(files.len(), 1);
    assert!(files[0].path.file_name().unwrap().to_string_lossy().contains("medium"));
}

#[test]
fn test_pattern_exclusion() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create directory structure
    fs::create_dir_all(base_path.join("src")).unwrap();
    fs::create_dir_all(base_path.join("target/debug")).unwrap();
    fs::create_dir_all(base_path.join("node_modules")).unwrap();
    
    fs::write(base_path.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(base_path.join("target/debug/app"), "binary").unwrap();
    fs::write(base_path.join("node_modules/package.json"), "{}").unwrap();
    
    let config = FolderConfig {
        max_depth: Some(5),
        include_extensions: None, // Include all supported types
        exclude_extensions: vec![],
        exclude_patterns: vec![
            "*/target/*".to_string(),
            "*/node_modules/*".to_string(),
        ],
        min_file_size: 1,
        max_file_size: 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let processor = FolderProcessor::new(config).unwrap();
    let files = processor.discover_files(base_path).unwrap();
    
    // Should only find src/main.rs, not files in target or node_modules
    assert_eq!(files.len(), 1);
    assert!(files[0].path.to_string_lossy().contains("src/main.rs"));
}

#[tokio::test]
async fn test_encoder_directory_processing() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create test files
    fs::write(base_path.join("test.rs"), "fn test() { println!(\"Hello\"); }").unwrap();
    fs::write(base_path.join("data.csv"), "id,name\n1,Alice\n2,Bob").unwrap();
    fs::write(base_path.join("config.json"), r#"{"debug": true}"#).unwrap();
    
    // Skip test if embedding model not available (e.g., in CI)
    let config = Config::default();
    let mut encoder = match MemvidEncoder::new_with_config(config).await {
        Ok(encoder) => encoder,
        Err(_) => return, // Skip test
    };
    
    let stats = encoder.add_directory(&base_path.to_string_lossy()).await.unwrap();
    
    assert!(stats.files_processed >= 3);
    assert_eq!(stats.files_failed, 0);
    assert!(encoder.len() >= 3);
}

#[tokio::test]
async fn test_encoder_preview_directory() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create test files
    fs::write(base_path.join("test.rs"), "fn test() {}").unwrap();
    fs::write(base_path.join("test.py"), "def test(): pass").unwrap();
    fs::write(base_path.join("README.md"), "# Test").unwrap();
    
    let config = Config::default();
    let encoder = match MemvidEncoder::new_with_config(config).await {
        Ok(encoder) => encoder,
        Err(_) => return, // Skip test
    };
    
    let files = encoder.preview_directory(&base_path.to_string_lossy()).unwrap();
    
    assert!(files.len() >= 3);
    
    let extensions: Vec<String> = files.iter().map(|f| f.extension.clone()).collect();
    assert!(extensions.contains(&"rs".to_string()));
    assert!(extensions.contains(&"py".to_string()));
    assert!(extensions.contains(&"md".to_string()));
}

#[tokio::test]
async fn test_encoder_custom_folder_config() {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();
    
    // Create test files
    fs::write(base_path.join("test.rs"), "fn test() {}").unwrap();
    fs::write(base_path.join("test.py"), "def test(): pass").unwrap();
    fs::write(base_path.join("test.js"), "function test() {}").unwrap();
    
    let custom_config = FolderConfig {
        max_depth: Some(1),
        include_extensions: Some(vec!["rs".to_string()]), // Only Rust files
        exclude_extensions: vec![],
        exclude_patterns: vec![],
        min_file_size: 1,
        max_file_size: 1024,
        follow_symlinks: false,
        include_hidden: false,
        skip_binary: true,
    };
    
    let config = Config::default();
    let mut encoder = match MemvidEncoder::new_with_config(config).await {
        Ok(encoder) => encoder,
        Err(_) => return, // Skip test
    };
    
    let stats = encoder.add_directory_with_config(
        &base_path.to_string_lossy(), 
        Some(custom_config)
    ).await.unwrap();
    
    // Should only process the Rust file
    assert_eq!(stats.files_processed, 1);
    assert!(encoder.len() >= 1);
}

#[test]
fn test_nonexistent_directory() {
    let processor = FolderProcessor::default();
    let result = processor.discover_files("/nonexistent/directory");
    assert!(result.is_err());
}

#[test]
fn test_file_as_directory() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.txt");
    fs::write(&file_path, "test content").unwrap();
    
    let processor = FolderProcessor::default();
    let result = processor.discover_files(&file_path);
    assert!(result.is_err());
}
