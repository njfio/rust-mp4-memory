//! Web server for browser-based memory management and collaboration

use std::collections::HashMap;
use std::sync::Arc;
use axum::{
    extract::{Path, Query, State, WebSocketUpgrade, ws::WebSocket},
    http::{StatusCode, HeaderValue},
    response::{Html, Response},
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use tracing::{info, debug, warn};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::retriever::MemvidRetriever;
use crate::analytics_dashboard::DashboardOutput;

/// Web server for memory management
pub struct MemoryWebServer {
    config: Config,
    memories: Arc<DashMap<String, MemoryInstance>>,
    active_sessions: Arc<DashMap<String, UserSession>>,
    collaboration_hub: Arc<CollaborationHub>,
    analytics_cache: Arc<RwLock<HashMap<String, DashboardOutput>>>,
}

/// Memory instance managed by the web server
#[derive(Clone)]
pub struct MemoryInstance {
    pub id: String,
    pub name: String,
    pub video_path: String,
    pub index_path: String,
    pub retriever: Arc<MemvidRetriever>,
    pub metadata: MemoryMetadata,
    pub permissions: MemoryPermissions,
}

/// Serializable version of MemoryInstance for API responses
#[derive(Clone, Serialize)]
pub struct MemoryInstanceResponse {
    pub id: String,
    pub name: String,
    pub video_path: String,
    pub index_path: String,
    pub metadata: MemoryMetadata,
    pub permissions: MemoryPermissions,
}

impl From<&MemoryInstance> for MemoryInstanceResponse {
    fn from(instance: &MemoryInstance) -> Self {
        Self {
            id: instance.id.clone(),
            name: instance.name.clone(),
            video_path: instance.video_path.clone(),
            index_path: instance.index_path.clone(),
            metadata: instance.metadata.clone(),
            permissions: instance.permissions.clone(),
        }
    }
}

/// Memory metadata for web interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub size_bytes: u64,
    pub chunk_count: usize,
    pub tags: Vec<String>,
    pub description: Option<String>,
    pub owner: String,
}

/// Memory access permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPermissions {
    pub owner: String,
    pub readers: Vec<String>,
    pub writers: Vec<String>,
    pub public_read: bool,
    pub public_write: bool,
}

/// User session information
#[derive(Debug, Clone)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub username: String,
    pub connected_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub active_memory: Option<String>,
}

/// Collaboration hub for real-time features
pub struct CollaborationHub {
    pub broadcast_tx: broadcast::Sender<CollaborationEvent>,
    pub active_collaborations: DashMap<String, CollaborationSession>,
}

/// Collaboration session for a memory
#[derive(Debug, Clone)]
pub struct CollaborationSession {
    pub memory_id: String,
    pub participants: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Real-time collaboration events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationEvent {
    pub event_type: String,
    pub memory_id: String,
    pub user_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data: serde_json::Value,
}

/// Search request from web interface
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub memory_ids: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub filters: Option<SearchFilters>,
}

/// Search filters
#[derive(Debug, Deserialize)]
pub struct SearchFilters {
    pub tags: Option<Vec<String>>,
    pub date_range: Option<DateRange>,
    pub content_types: Option<Vec<String>>,
}

/// Date range filter
#[derive(Debug, Deserialize)]
pub struct DateRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Search response
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_count: usize,
    pub search_time_ms: u64,
    pub memory_stats: HashMap<String, usize>,
}

/// Individual search result
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub memory_id: String,
    pub memory_name: String,
    pub content: String,
    pub score: f64,
    pub frame_number: u32,
    pub metadata: serde_json::Value,
}

/// Memory creation request
#[derive(Debug, Deserialize)]
pub struct CreateMemoryRequest {
    pub name: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub content_files: Vec<String>,
    pub permissions: MemoryPermissions,
}

/// Memory update request
#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub permissions: Option<MemoryPermissions>,
}

impl MemoryWebServer {
    /// Create a new web server
    pub fn new(config: Config) -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        
        Self {
            config,
            memories: Arc::new(DashMap::new()),
            active_sessions: Arc::new(DashMap::new()),
            collaboration_hub: Arc::new(CollaborationHub {
                broadcast_tx,
                active_collaborations: DashMap::new(),
            }),
            analytics_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start the web server
    pub async fn start(self, bind_address: &str) -> Result<()> {
        info!("Starting MemVid web server on {}", bind_address);

        let app_state = Arc::new(self);

        // Build the router
        let app = Router::new()
            // Static file serving
            .nest_service("/static", ServeDir::new("web/static"))
            
            // Main web interface
            .route("/", get(serve_index))
            .route("/dashboard", get(serve_dashboard))
            .route("/memories", get(serve_memories))
            .route("/search", get(serve_search))
            .route("/analytics", get(serve_analytics))
            
            // API routes
            .route("/api/memories", get(list_memories))
            .route("/api/memories/:id", get(get_memory))
            .route("/api/search", post(search_memories))
            
            // WebSocket for real-time collaboration
            .route("/ws", get(websocket_handler))
            
            // Health check
            .route("/health", get(health_check))
            
            .layer(
                CorsLayer::new()
                    .allow_origin("*".parse::<HeaderValue>().unwrap())
                    .allow_methods([axum::http::Method::GET, axum::http::Method::POST, 
                                   axum::http::Method::PUT, axum::http::Method::DELETE])
                    .allow_headers([axum::http::header::CONTENT_TYPE])
            )
            .layer(TraceLayer::new_for_http())
            .with_state(app_state);

        // Start the server
        let listener = tokio::net::TcpListener::bind(bind_address).await
            .map_err(|e| MemvidError::generic(format!("Failed to bind to {}: {}", bind_address, e)))?;

        info!("MemVid web server listening on {}", bind_address);
        
        axum::serve(listener, app).await
            .map_err(|e| MemvidError::generic(format!("Server error: {}", e)))?;

        Ok(())
    }

    /// Load existing memories into the server
    pub async fn load_memory(&self, memory_id: String, video_path: String, index_path: String) -> Result<()> {
        info!("Loading memory: {} from {}", memory_id, video_path);

        let retriever = MemvidRetriever::new_with_config(&video_path, &index_path, self.config.clone()).await?;
        
        // Get memory statistics
        let stats = retriever.get_stats();
        
        let memory_instance = MemoryInstance {
            id: memory_id.clone(),
            name: memory_id.clone(), // TODO: Extract from metadata
            video_path,
            index_path,
            retriever: Arc::new(retriever),
            metadata: MemoryMetadata {
                created_at: chrono::Utc::now(),
                last_accessed: chrono::Utc::now(),
                size_bytes: 0, // TODO: Get actual file size
                chunk_count: stats.index_stats.total_chunks,
                tags: Vec::new(),
                description: None,
                owner: "system".to_string(),
            },
            permissions: MemoryPermissions {
                owner: "system".to_string(),
                readers: Vec::new(),
                writers: Vec::new(),
                public_read: true,
                public_write: false,
            },
        };

        self.memories.insert(memory_id, memory_instance);
        Ok(())
    }

    /// Get memory by ID
    pub fn get_memory(&self, memory_id: &str) -> Option<MemoryInstance> {
        self.memories.get(memory_id).map(|entry| entry.clone())
    }

    /// List all accessible memories for a user
    pub fn list_user_memories(&self, user_id: &str) -> Vec<MemoryInstance> {
        self.memories
            .iter()
            .filter(|entry| {
                let memory = entry.value();
                memory.permissions.owner == user_id ||
                memory.permissions.readers.contains(&user_id.to_string()) ||
                memory.permissions.writers.contains(&user_id.to_string()) ||
                memory.permissions.public_read
            })
            .map(|entry| entry.value().clone())
            .collect()
    }
}

// Web route handlers
async fn serve_index() -> Html<&'static str> {
    Html(include_str!("../web/templates/index.html"))
}

async fn serve_dashboard() -> Html<&'static str> {
    Html(include_str!("../web/templates/dashboard.html"))
}

async fn serve_memories() -> Html<&'static str> {
    Html(include_str!("../web/templates/memories.html"))
}

async fn serve_search() -> Html<&'static str> {
    Html(include_str!("../web/templates/search.html"))
}

async fn serve_analytics() -> Html<&'static str> {
    Html(include_str!("../web/templates/analytics.html"))
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[axum::debug_handler]
async fn list_memories(
    State(server): State<Arc<MemoryWebServer>>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<Vec<MemoryInstanceResponse>> {
    let user_id = params.get("user_id").unwrap_or(&"anonymous".to_string()).clone();
    let memories = server.list_user_memories(&user_id);
    let responses: Vec<MemoryInstanceResponse> = memories.iter().map(|m| m.into()).collect();
    Json(responses)
}

#[axum::debug_handler]
async fn get_memory(
    State(server): State<Arc<MemoryWebServer>>,
    Path(memory_id): Path<String>,
) -> std::result::Result<Json<MemoryInstanceResponse>, StatusCode> {
    match server.get_memory(&memory_id) {
        Some(memory) => Ok(Json((&memory).into())),
        None => Err(StatusCode::NOT_FOUND),
    }
}



#[axum::debug_handler]
async fn search_memories(
    State(server): State<Arc<MemoryWebServer>>,
    Json(request): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let start_time = std::time::Instant::now();
    let mut results = Vec::new();
    let mut memory_stats = HashMap::new();

    // Determine which memories to search
    let memory_ids = request.memory_ids.unwrap_or_else(|| {
        server.memories.iter().map(|entry| entry.key().clone()).collect()
    });

    // Search each memory
    for memory_id in memory_ids {
        if let Some(memory) = server.get_memory(&memory_id) {
            match memory.retriever.search_with_metadata(&request.query, request.limit.unwrap_or(10)).await {
                Ok(search_results) => {
                    let count = search_results.len();
                    memory_stats.insert(memory_id.clone(), count);
                    
                    for result in search_results {
                        results.push(SearchResult {
                            chunk_id: result.chunk_id.to_string(),
                            memory_id: memory_id.clone(),
                            memory_name: memory.name.clone(),
                            content: result.text,
                            score: result.similarity as f64,
                            frame_number: result.frame_number,
                            metadata: serde_json::json!({}),
                        });
                    }
                }
                Err(e) => {
                    warn!("Search failed for memory {}: {}", memory_id, e);
                }
            }
        }
    }

    // Sort results by score
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Apply limit
    if let Some(limit) = request.limit {
        results.truncate(limit);
    }

    let search_time_ms = start_time.elapsed().as_millis() as u64;

    Json(SearchResponse {
        total_count: results.len(),
        results,
        search_time_ms,
        memory_stats,
    })
}



async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(_server): State<Arc<MemoryWebServer>>,
) -> Response {
    ws.on_upgrade(handle_websocket)
}

async fn handle_websocket(mut socket: WebSocket) {
    // TODO: Implement WebSocket handling for real-time collaboration
    while let Some(msg) = socket.recv().await {
        if let Ok(msg) = msg {
            if let axum::extract::ws::Message::Text(text) = msg {
                debug!("Received WebSocket message: {}", text);
                // Echo back for now
                if socket.send(axum::extract::ws::Message::Text(format!("Echo: {}", text))).await.is_err() {
                    break;
                }
            }
        } else {
            break;
        }
    }
}
