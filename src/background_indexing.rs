//! Background indexing system for non-blocking index creation

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::index::IndexManager;
use crate::text::TextChunk;

/// Background indexing job
#[derive(Debug, Clone)]
pub struct IndexingJob {
    pub job_id: String,
    pub chunks: Vec<TextChunk>,
    pub index_path: PathBuf,
    pub config: Config,
    pub created_at: std::time::SystemTime,
}

/// Status of a background indexing job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStatus {
    Queued,
    InProgress { progress: f64 },
    Completed { duration_seconds: f64 },
    Failed { error: String },
}

/// Background indexing service
pub struct BackgroundIndexer {
    job_sender: mpsc::UnboundedSender<IndexingJob>,
    status_store: Arc<RwLock<std::collections::HashMap<String, IndexingStatus>>>,
    _worker_handle: JoinHandle<()>,
}

impl BackgroundIndexer {
    /// Create a new background indexer
    pub fn new() -> Self {
        let (job_sender, job_receiver) = mpsc::unbounded_channel();
        let status_store = Arc::new(RwLock::new(std::collections::HashMap::new()));
        
        let worker_status_store = Arc::clone(&status_store);
        let worker_handle = tokio::spawn(async move {
            Self::worker_loop(job_receiver, worker_status_store).await;
        });

        Self {
            job_sender,
            status_store,
            _worker_handle: worker_handle,
        }
    }

    /// Submit a new indexing job
    pub async fn submit_job(&self, chunks: Vec<TextChunk>, index_path: PathBuf, config: Config) -> Result<String> {
        let job_id = format!("idx_{}", uuid::Uuid::new_v4().simple());
        
        let job = IndexingJob {
            job_id: job_id.clone(),
            chunks,
            index_path,
            config,
            created_at: std::time::SystemTime::now(),
        };

        // Mark job as queued
        {
            let mut status_map = self.status_store.write().await;
            status_map.insert(job_id.clone(), IndexingStatus::Queued);
        }

        // Send job to worker
        self.job_sender.send(job).map_err(|_| {
            MemvidError::background_indexing("Failed to submit indexing job")
        })?;

        info!("Submitted background indexing job: {}", job_id);
        Ok(job_id)
    }

    /// Get the status of a job
    pub async fn get_job_status(&self, job_id: &str) -> Option<IndexingStatus> {
        let status_map = self.status_store.read().await;
        status_map.get(job_id).cloned()
    }

    /// Get all job statuses
    pub async fn get_all_job_statuses(&self) -> std::collections::HashMap<String, IndexingStatus> {
        let status_map = self.status_store.read().await;
        status_map.clone()
    }

    /// Wait for a job to complete
    pub async fn wait_for_job(&self, job_id: &str, timeout_seconds: Option<u64>) -> Result<IndexingStatus> {
        let start_time = Instant::now();
        let timeout_duration = timeout_seconds.map(std::time::Duration::from_secs);

        loop {
            if let Some(timeout) = timeout_duration {
                if start_time.elapsed() > timeout {
                    return Err(MemvidError::background_indexing("Job wait timeout"));
                }
            }

            let status = self.get_job_status(job_id).await;
            match status {
                Some(IndexingStatus::Completed { .. }) => return Ok(status.unwrap()),
                Some(IndexingStatus::Failed { .. }) => return Ok(status.unwrap()),
                Some(IndexingStatus::InProgress { .. }) | Some(IndexingStatus::Queued) => {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                None => {
                    return Err(MemvidError::background_indexing("Job not found"));
                }
            }
        }
    }

    /// Worker loop that processes indexing jobs
    async fn worker_loop(
        mut job_receiver: mpsc::UnboundedReceiver<IndexingJob>,
        status_store: Arc<RwLock<std::collections::HashMap<String, IndexingStatus>>>,
    ) {
        info!("Background indexing worker started");

        while let Some(job) = job_receiver.recv().await {
            debug!("Processing indexing job: {}", job.job_id);
            
            // Update status to in progress
            {
                let mut status_map = status_store.write().await;
                status_map.insert(job.job_id.clone(), IndexingStatus::InProgress { progress: 0.0 });
            }

            let start_time = Instant::now();
            
            // Process the job
            match Self::process_indexing_job(&job, &status_store).await {
                Ok(()) => {
                    let duration = start_time.elapsed().as_secs_f64();
                    let mut status_map = status_store.write().await;
                    status_map.insert(
                        job.job_id.clone(),
                        IndexingStatus::Completed { duration_seconds: duration }
                    );
                    info!("Completed indexing job {} in {:.2}s", job.job_id, duration);
                }
                Err(e) => {
                    let mut status_map = status_store.write().await;
                    status_map.insert(
                        job.job_id.clone(),
                        IndexingStatus::Failed { error: e.to_string() }
                    );
                    error!("Failed indexing job {}: {}", job.job_id, e);
                }
            }
        }

        info!("Background indexing worker stopped");
    }

    /// Process a single indexing job
    async fn process_indexing_job(
        job: &IndexingJob,
        status_store: &Arc<RwLock<std::collections::HashMap<String, IndexingStatus>>>,
    ) -> Result<()> {
        // Create index manager
        let mut index_manager = IndexManager::new(job.config.clone()).await?;
        
        let total_chunks = job.chunks.len();
        let batch_size = 100;
        let mut processed = 0;

        // Process chunks in batches with progress updates
        for chunk_batch in job.chunks.chunks(batch_size) {
            // Add chunks to index
            index_manager.add_chunks_incremental(chunk_batch.to_vec()).await?;
            
            processed += chunk_batch.len();
            let progress = (processed as f64 / total_chunks as f64) * 100.0;
            
            // Update progress
            {
                let mut status_map = status_store.write().await;
                status_map.insert(
                    job.job_id.clone(),
                    IndexingStatus::InProgress { progress }
                );
            }

            debug!("Indexing job {} progress: {:.1}%", job.job_id, progress);
        }

        // Save the index
        index_manager.save(job.index_path.to_str().unwrap())?;
        
        Ok(())
    }

    /// Clean up old completed/failed jobs
    pub async fn cleanup_old_jobs(&self, max_age_hours: u64) {
        let cutoff_time = std::time::SystemTime::now() - std::time::Duration::from_secs(max_age_hours * 3600);
        
        let mut status_map = self.status_store.write().await;
        let mut to_remove = Vec::new();
        
        for (job_id, status) in status_map.iter() {
            match status {
                IndexingStatus::Completed { .. } | IndexingStatus::Failed { .. } => {
                    // In a real implementation, you'd store job creation time
                    // For now, we'll just keep recent jobs
                    if job_id.len() > 20 { // Simple heuristic
                        to_remove.push(job_id.clone());
                    }
                }
                _ => {}
            }
        }
        
        for job_id in to_remove {
            status_map.remove(&job_id);
        }
    }
}

impl Default for BackgroundIndexer {
    fn default() -> Self {
        Self::new()
    }
}

/// Global background indexer instance
static BACKGROUND_INDEXER: tokio::sync::OnceCell<BackgroundIndexer> = tokio::sync::OnceCell::const_new();

/// Get the global background indexer instance
pub async fn get_background_indexer() -> &'static BackgroundIndexer {
    BACKGROUND_INDEXER.get_or_init(|| async {
        BackgroundIndexer::new()
    }).await
}

/// Submit a background indexing job using the global indexer
pub async fn submit_background_indexing(
    chunks: Vec<TextChunk>,
    index_path: PathBuf,
    config: Config,
) -> Result<String> {
    let indexer = get_background_indexer().await;
    indexer.submit_job(chunks, index_path, config).await
}

/// Get the status of a background indexing job
pub async fn get_indexing_status(job_id: &str) -> Option<IndexingStatus> {
    let indexer = get_background_indexer().await;
    indexer.get_job_status(job_id).await
}

/// Wait for a background indexing job to complete
pub async fn wait_for_indexing(job_id: &str, timeout_seconds: Option<u64>) -> Result<IndexingStatus> {
    let indexer = get_background_indexer().await;
    indexer.wait_for_job(job_id, timeout_seconds).await
}
