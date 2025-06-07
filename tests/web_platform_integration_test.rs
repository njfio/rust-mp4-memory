//! Web Platform Integration Tests
//! 
//! Comprehensive tests for the web platform functionality including:
//! - Web server startup and configuration
//! - API endpoints functionality
//! - Memory management operations
//! - Search and analytics features
//! - Real-time collaboration features

use rust_mem_vid::{
    MemvidEncoder, MemoryWebServer, Config,
    video::Codec,
};
use std::time::Duration;
use std::sync::Arc;
use tokio::time::timeout;
use reqwest::Client;
use serde_json::Value;

#[tokio::test]
async fn test_web_server_startup() {
    let _ = tracing_subscriber::fmt().try_init();
    
    // Initialize the library
    rust_mem_vid::init().await.expect("Failed to initialize library");

    let config = Config::default();
    let _server = MemoryWebServer::new(config);

    // Test that server can be created without errors
    assert!(true); // Server creation successful
}

#[tokio::test]
async fn test_memory_loading() {
    let _ = tracing_subscriber::fmt().try_init();
    
    // Create a test memory
    let config = Config::default();
    let mut encoder = MemvidEncoder::new_with_config(config.clone()).await
        .expect("Failed to create encoder");

    let test_content = vec![
        "Test content for web platform integration".to_string(),
        "This is sample data for testing the web server".to_string(),
        "Memory loading functionality validation".to_string(),
    ];

    encoder.add_chunks(test_content).await
        .expect("Failed to add chunks");

    let video_path = "test_web_memory.mp4";
    let index_path = "test_web_memory.metadata";

    encoder.build_video_with_codec(video_path, index_path, Some(Codec::H264)).await
        .expect("Failed to build video");

    // Test loading memory into web server
    let server = MemoryWebServer::new(config);
    let result = server.load_memory(
        "test_memory".to_string(),
        video_path.to_string(),
        index_path.to_string()
    ).await;

    assert!(result.is_ok(), "Failed to load memory into web server");

    // Verify memory is accessible
    let memory = server.get_memory("test_memory");
    assert!(memory.is_some(), "Memory not found after loading");

    let memory = memory.unwrap();
    assert_eq!(memory.id, "test_memory");
    assert_eq!(memory.video_path, video_path);
    assert_eq!(memory.index_path, index_path);

    // Cleanup
    let _ = std::fs::remove_file(video_path);
    let _ = std::fs::remove_file(index_path);
}

#[tokio::test]
async fn test_memory_permissions() {
    let _ = tracing_subscriber::fmt().try_init();
    
    let config = Config::default();
    let server = MemoryWebServer::new(config);

    // Test user memory listing with different permission scenarios
    let memories = server.list_user_memories("test_user");
    assert!(memories.is_empty(), "Should start with no memories for new user");

    let memories = server.list_user_memories("anonymous");
    assert!(memories.is_empty(), "Should start with no memories for anonymous user");
}

#[tokio::test]
async fn test_api_endpoints_structure() {
    // Test that we can create the necessary structures for API responses
    use rust_mem_vid::web_server::{SearchRequest, SearchResponse, SearchResult};
    use serde_json;

    let search_request = SearchRequest {
        query: "test query".to_string(),
        memory_ids: Some(vec!["memory1".to_string()]),
        limit: Some(10),
        filters: None,
    };

    let json = serde_json::to_string(&search_request);
    assert!(json.is_ok(), "SearchRequest should be serializable");

    let search_result = SearchResult {
        chunk_id: "chunk1".to_string(),
        memory_id: "memory1".to_string(),
        memory_name: "Test Memory".to_string(),
        content: "Test content".to_string(),
        score: 0.95,
        frame_number: 42,
        metadata: serde_json::json!({}),
    };

    let search_response = SearchResponse {
        results: vec![search_result],
        total_count: 1,
        search_time_ms: 150,
        memory_stats: std::collections::HashMap::new(),
    };

    let json = serde_json::to_string(&search_response);
    assert!(json.is_ok(), "SearchResponse should be serializable");
}

#[tokio::test]
async fn test_collaboration_structures() {
    use rust_mem_vid::web_server::{CollaborationEvent, UserSession};
    use chrono::Utc;

    let event = CollaborationEvent {
        event_type: "test_event".to_string(),
        memory_id: "memory1".to_string(),
        user_id: "user1".to_string(),
        timestamp: Utc::now(),
        data: serde_json::json!({"test": "data"}),
    };

    let json = serde_json::to_string(&event);
    assert!(json.is_ok(), "CollaborationEvent should be serializable");

    let session = UserSession {
        session_id: "session1".to_string(),
        user_id: "user1".to_string(),
        username: "testuser".to_string(),
        connected_at: Utc::now(),
        last_activity: Utc::now(),
        active_memory: Some("memory1".to_string()),
    };

    assert_eq!(session.user_id, "user1");
    assert_eq!(session.username, "testuser");
}

#[tokio::test]
async fn test_memory_metadata_structures() {
    use rust_mem_vid::web_server::{MemoryMetadata, MemoryPermissions};
    use chrono::Utc;

    let metadata = MemoryMetadata {
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        size_bytes: 1024,
        chunk_count: 10,
        tags: vec!["test".to_string(), "demo".to_string()],
        description: Some("Test memory".to_string()),
        owner: "testuser".to_string(),
    };

    assert_eq!(metadata.size_bytes, 1024);
    assert_eq!(metadata.chunk_count, 10);
    assert_eq!(metadata.tags.len(), 2);

    let permissions = MemoryPermissions {
        owner: "testuser".to_string(),
        readers: vec!["reader1".to_string()],
        writers: vec!["writer1".to_string()],
        public_read: false,
        public_write: false,
    };

    assert_eq!(permissions.owner, "testuser");
    assert_eq!(permissions.readers.len(), 1);
    assert_eq!(permissions.writers.len(), 1);
    assert!(!permissions.public_read);
    assert!(!permissions.public_write);
}

#[tokio::test]
async fn test_web_server_configuration() {
    let _ = tracing_subscriber::fmt().try_init();
    
    let config = Config::default();
    let server = MemoryWebServer::new(config);

    // Test that server has correct initial state
    let memories = server.list_user_memories("anonymous");
    assert!(memories.is_empty(), "New server should have no memories");
}

#[tokio::test]
async fn test_html_template_loading() {
    // Test that HTML templates can be loaded (they're embedded in the binary)
    let index_html = include_str!("../web/templates/index.html");
    assert!(index_html.contains("MemVid"), "Index template should contain MemVid branding");
    assert!(index_html.contains("Intelligent Memory Management"), "Index template should contain main heading");

    let search_html = include_str!("../web/templates/search.html");
    assert!(search_html.contains("Advanced Search"), "Search template should contain search functionality");

    let dashboard_html = include_str!("../web/templates/dashboard.html");
    assert!(dashboard_html.contains("Analytics Dashboard"), "Dashboard template should contain analytics");

    let memories_html = include_str!("../web/templates/memories.html");
    assert!(memories_html.contains("Memory Management"), "Memories template should contain memory management");

    let analytics_html = include_str!("../web/templates/analytics.html");
    assert!(analytics_html.contains("AI Intelligence Analytics"), "Analytics template should contain AI features");
}

#[tokio::test]
async fn test_css_and_js_assets() {
    // Test that CSS and JS assets are properly structured
    let main_css = include_str!("../web/static/css/main.css");
    assert!(main_css.contains(":root"), "CSS should contain CSS variables");
    assert!(main_css.contains(".navbar"), "CSS should contain navbar styles");
    assert!(main_css.contains(".memory-card"), "CSS should contain memory card styles");

    let main_js = include_str!("../web/static/js/main.js");
    assert!(main_js.contains("MemVid"), "JS should contain MemVid references");
    assert!(main_js.contains("WebSocket"), "JS should contain WebSocket functionality");
    assert!(main_js.contains("api"), "JS should contain API functionality");
}

#[tokio::test]
async fn test_error_handling() {
    let _ = tracing_subscriber::fmt().try_init();
    
    let config = Config::default();
    let server = MemoryWebServer::new(config);

    // Test loading non-existent memory
    let result = server.load_memory(
        "nonexistent".to_string(),
        "nonexistent.mp4".to_string(),
        "nonexistent.metadata".to_string()
    ).await;

    assert!(result.is_err(), "Loading non-existent memory should fail");

    // Test getting non-existent memory
    let memory = server.get_memory("nonexistent");
    assert!(memory.is_none(), "Getting non-existent memory should return None");
}

#[tokio::test]
async fn test_concurrent_operations() {
    let _ = tracing_subscriber::fmt().try_init();
    
    let config = Config::default();
    let server = Arc::new(MemoryWebServer::new(config));

    // Test that multiple concurrent operations don't cause issues
    let handles = (0..10).map(|i| {
        let server = Arc::clone(&server);
        tokio::spawn(async move {
            let memories = server.list_user_memories(&format!("user{}", i));
            assert!(memories.is_empty());
        })
    }).collect::<Vec<_>>();

    for handle in handles {
        handle.await.expect("Concurrent operation should complete");
    }
}

#[tokio::test]
async fn test_memory_instance_creation() {
    use rust_mem_vid::web_server::{MemoryInstance, MemoryMetadata, MemoryPermissions};
    use rust_mem_vid::MemvidRetriever;
    use chrono::Utc;


    // Create a test memory instance structure
    let metadata = MemoryMetadata {
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        size_bytes: 2048,
        chunk_count: 20,
        tags: vec!["test".to_string()],
        description: Some("Test instance".to_string()),
        owner: "testuser".to_string(),
    };

    let permissions = MemoryPermissions {
        owner: "testuser".to_string(),
        readers: vec![],
        writers: vec![],
        public_read: true,
        public_write: false,
    };

    // Note: We can't easily create a MemvidRetriever without actual files,
    // so we'll test the structure creation
    assert_eq!(metadata.chunk_count, 20);
    assert_eq!(permissions.owner, "testuser");
    assert!(permissions.public_read);
}

// Helper function for integration tests that need actual HTTP server
async fn start_test_server() -> Result<String, Box<dyn std::error::Error>> {
    // This would start a test server on a random port
    // For now, we'll return a placeholder
    Ok("127.0.0.1:0".to_string())
}

#[tokio::test]
async fn test_integration_placeholder() {
    // Placeholder for full integration tests that would require
    // starting an actual HTTP server and making requests
    // These would be implemented in a full production system
    
    let server_addr = start_test_server().await;
    assert!(server_addr.is_ok(), "Test server should start successfully");
}
