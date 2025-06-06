//! Data processing for structured data formats (CSV, Parquet) and code files

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::text::{ChunkMetadata, TextChunk};

// Polars imports
use polars::prelude::*;

/// Supported data file types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataFileType {
    Csv,
    Parquet,
    Json,
    Rust,
    JavaScript,
    TypeScript,
    Python,
    Html,
    Css,
    Log,
    Yaml,
    Toml,
    Xml,
    Sql,
    Markdown,
}

impl DataFileType {
    /// Detect file type from extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "csv" => Some(Self::Csv),
            "parquet" | "pq" => Some(Self::Parquet),
            "json" | "jsonl" => Some(Self::Json),
            "rs" => Some(Self::Rust),
            "js" | "jsx" => Some(Self::JavaScript),
            "ts" | "tsx" => Some(Self::TypeScript),
            "py" | "pyw" => Some(Self::Python),
            "html" | "htm" => Some(Self::Html),
            "css" | "scss" | "sass" => Some(Self::Css),
            "log" | "logs" => Some(Self::Log),
            "yaml" | "yml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            "xml" => Some(Self::Xml),
            "sql" => Some(Self::Sql),
            "md" | "markdown" => Some(Self::Markdown),
            _ => None,
        }
    }

    /// Get syntax highlighting language identifier
    pub fn syntax_language(&self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Parquet => "text",
            Self::Json => "json",
            Self::Rust => "rust",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Python => "python",
            Self::Html => "html",
            Self::Css => "css",
            Self::Log => "log",
            Self::Yaml => "yaml",
            Self::Toml => "toml",
            Self::Xml => "xml",
            Self::Sql => "sql",
            Self::Markdown => "markdown",
        }
    }
}

/// Chunking strategy for different data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Chunk by number of lines
    ByLines(usize),
    /// Chunk by data size in bytes
    BySize(usize),
    /// Chunk by logical units (functions, classes, etc.)
    ByLogicalUnits,
    /// Chunk by rows (for tabular data)
    ByRows(usize),
    /// Chunk by semantic similarity
    BySemantic,
    /// Chunk by column groups (for wide datasets)
    ByColumns(Vec<String>),
}

/// Enhanced metadata for data files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunkMetadata {
    /// Base metadata
    pub base: ChunkMetadata,
    /// File type
    pub file_type: DataFileType,
    /// Chunking strategy used
    pub strategy: ChunkingStrategy,
    /// Line range in original file
    pub line_range: Option<(usize, usize)>,
    /// Column information (for tabular data)
    pub columns: Option<Vec<String>>,
    /// Row count (for tabular data)
    pub row_count: Option<usize>,
    /// Code analysis metadata
    pub code_metadata: Option<CodeMetadata>,
    /// Data statistics
    pub data_stats: Option<DataStats>,
}

/// Code-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMetadata {
    /// Programming language
    pub language: String,
    /// Functions defined in this chunk
    pub functions: Vec<String>,
    /// Classes defined in this chunk
    pub classes: Vec<String>,
    /// Imports/dependencies
    pub imports: Vec<String>,
    /// Comments ratio
    pub comment_ratio: f64,
    /// Complexity metrics
    pub complexity: Option<ComplexityMetrics>,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic: u32,
    /// Lines of code
    pub loc: u32,
    /// Lines of comments
    pub comments: u32,
    /// Nesting depth
    pub max_depth: u32,
}

/// Data statistics for tabular data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStats {
    /// Column data types
    pub column_types: HashMap<String, String>,
    /// Null counts per column
    pub null_counts: HashMap<String, usize>,
    /// Unique value counts
    pub unique_counts: HashMap<String, usize>,
    /// Summary statistics for numeric columns
    pub numeric_stats: HashMap<String, NumericStats>,
}

/// Numeric column statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
}

/// Enhanced text chunk with data-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunk {
    /// The content (could be text, JSON, etc.)
    pub content: String,
    /// Enhanced metadata
    pub metadata: DataChunkMetadata,
    /// Original format preservation (for reconstruction)
    pub format_info: Option<FormatInfo>,
}

/// Information for preserving original format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatInfo {
    /// Original headers (for CSV/tabular data)
    pub headers: Option<Vec<String>>,
    /// Delimiter used (for CSV)
    pub delimiter: Option<char>,
    /// Encoding information
    pub encoding: Option<String>,
    /// Schema information (for Parquet)
    pub schema: Option<String>,
}

/// Data processor for handling various file types
pub struct DataProcessor {
    config: Config,
}

impl DataProcessor {
    /// Create a new data processor
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Create processor with default configuration
    pub fn default() -> Self {
        Self::new(Config::default())
    }

    /// Process any supported file type
    pub async fn process_file(&self, file_path: &str) -> Result<Vec<DataChunk>> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(MemvidError::file_not_found(file_path));
        }

        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        let file_type = DataFileType::from_extension(extension)
            .ok_or_else(|| MemvidError::unsupported_format(extension))?;

        match file_type {
            DataFileType::Csv => self.process_csv(file_path).await,
            DataFileType::Parquet => self.process_parquet(file_path).await,
            DataFileType::Json => self.process_json(file_path).await,
            DataFileType::Rust => self.process_code(file_path, file_type).await,
            DataFileType::JavaScript => self.process_code(file_path, file_type).await,
            DataFileType::TypeScript => self.process_code(file_path, file_type).await,
            DataFileType::Python => self.process_code(file_path, file_type).await,
            DataFileType::Html => self.process_code(file_path, file_type).await,
            DataFileType::Css => self.process_code(file_path, file_type).await,
            DataFileType::Log => self.process_log(file_path).await,
            DataFileType::Yaml => self.process_structured_text(file_path, file_type).await,
            DataFileType::Toml => self.process_structured_text(file_path, file_type).await,
            DataFileType::Xml => self.process_structured_text(file_path, file_type).await,
            DataFileType::Sql => self.process_code(file_path, file_type).await,
            DataFileType::Markdown => self.process_structured_text(file_path, file_type).await,
        }
    }

    /// Convert DataChunk to TextChunk for compatibility
    pub fn to_text_chunks(&self, data_chunks: Vec<DataChunk>) -> Vec<TextChunk> {
        data_chunks.into_iter().map(|chunk| {
            TextChunk {
                content: chunk.content,
                metadata: chunk.metadata.base,
            }
        }).collect()
    }

    /// Process CSV file
    async fn process_csv(&self, file_path: &str) -> Result<Vec<DataChunk>> {
        // Read CSV using polars DataFrame directly
        let df = CsvReader::from_path(file_path)
            .map_err(|e| MemvidError::data_processing(format!("CSV reader error: {}", e)))?
            .finish()
            .map_err(|e| MemvidError::data_processing(format!("CSV parsing error: {}", e)))?;

        let headers = df.get_column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let total_rows = df.height();

        // Calculate chunk size based on configuration
        let rows_per_chunk = self.config.text.chunk_size / (headers.len() * 20); // Estimate 20 chars per cell
        let rows_per_chunk = std::cmp::max(rows_per_chunk, 10); // Minimum 10 rows per chunk

        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        for start_row in (0..total_rows).step_by(rows_per_chunk) {
            let end_row = std::cmp::min(start_row + rows_per_chunk, total_rows);

            // Extract chunk data
            let chunk_df = df.slice(start_row as i64, (end_row - start_row) as usize);

            // Convert to CSV string
            let mut csv_content = String::new();

            // Add headers
            csv_content.push_str(&headers.join(","));
            csv_content.push('\n');

            // Add data rows
            for row_idx in 0..chunk_df.height() {
                let row_values: Vec<String> = headers.iter().map(|col| {
                    chunk_df.column(col)
                        .unwrap()
                        .get(row_idx)
                        .unwrap()
                        .to_string()
                }).collect();
                csv_content.push_str(&row_values.join(","));
                csv_content.push('\n');
            }

            // Calculate statistics
            let data_stats = self.calculate_csv_stats(&chunk_df, &headers)?;

            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: chunk_id,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0, // Not applicable for CSV
                    length: csv_content.len(),
                    frame: chunk_id as u32,
                    extra: HashMap::new(),
                },
                file_type: DataFileType::Csv,
                strategy: ChunkingStrategy::ByRows(rows_per_chunk),
                line_range: Some((start_row + 1, end_row)), // +1 for header
                columns: Some(headers.clone()),
                row_count: Some(end_row - start_row),
                code_metadata: None,
                data_stats: Some(data_stats),
            };

            let format_info = FormatInfo {
                headers: Some(headers.clone()),
                delimiter: Some(','),
                encoding: Some("utf-8".to_string()),
                schema: None,
            };

            chunks.push(DataChunk {
                content: csv_content,
                metadata,
                format_info: Some(format_info),
            });

            chunk_id += 1;
        }

        Ok(chunks)
    }

    /// Calculate statistics for CSV chunk
    fn calculate_csv_stats(&self, df: &DataFrame, headers: &[String]) -> Result<DataStats> {
        let mut column_types = HashMap::new();
        let mut null_counts = HashMap::new();
        let mut unique_counts = HashMap::new();
        let mut numeric_stats = HashMap::new();

        for col_name in headers {
            if let Ok(column) = df.column(col_name) {
                // Data type
                column_types.insert(col_name.clone(), column.dtype().to_string());

                // Null count
                let null_count = column.null_count();
                null_counts.insert(col_name.clone(), null_count);

                // Unique count (simplified)
                let unique_count = column.n_unique().unwrap_or(0);
                unique_counts.insert(col_name.clone(), unique_count);

                // Numeric statistics
                if column.dtype().is_numeric() {
                    if let Ok(series) = column.cast(&DataType::Float64) {
                        let values: Vec<f64> = series.f64()
                            .unwrap()
                            .into_iter()
                            .filter_map(|v| v)
                            .collect();

                        if !values.is_empty() {
                            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            let mean = values.iter().sum::<f64>() / values.len() as f64;

                            let mut sorted_values = values.clone();
                            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let median = if sorted_values.len() % 2 == 0 {
                                (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
                            } else {
                                sorted_values[sorted_values.len() / 2]
                            };

                            let variance = values.iter()
                                .map(|x| (x - mean).powi(2))
                                .sum::<f64>() / values.len() as f64;
                            let std_dev = variance.sqrt();

                            numeric_stats.insert(col_name.clone(), NumericStats {
                                min, max, mean, median, std_dev
                            });
                        }
                    }
                }
            }
        }

        Ok(DataStats {
            column_types,
            null_counts,
            unique_counts,
            numeric_stats,
        })
    }

    /// Process Parquet file
    async fn process_parquet(&self, file_path: &str) -> Result<Vec<DataChunk>> {
        // Read Parquet using polars DataFrame directly
        let df = LazyFrame::scan_parquet(file_path, ScanArgsParquet::default())
            .map_err(|e| MemvidError::data_processing(format!("Parquet parsing error: {}", e)))?
            .collect()
            .map_err(|e| MemvidError::data_processing(format!("Parquet collection error: {}", e)))?;

        let headers = df.get_column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let total_rows = df.height();

        // Calculate chunk size (Parquet is typically more efficient, so larger chunks)
        let rows_per_chunk = self.config.text.chunk_size / (headers.len() * 15); // Estimate 15 chars per cell
        let rows_per_chunk = std::cmp::max(rows_per_chunk, 50); // Minimum 50 rows per chunk

        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        for start_row in (0..total_rows).step_by(rows_per_chunk) {
            let end_row = std::cmp::min(start_row + rows_per_chunk, total_rows);

            // Extract chunk data
            let chunk_df = df.slice(start_row as i64, (end_row - start_row) as usize);

            // Convert to JSON for storage (preserves data types better than CSV)
            let json_content = self.dataframe_to_json(&chunk_df)?;

            // Calculate statistics
            let data_stats = self.calculate_csv_stats(&chunk_df, &headers)?;

            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: chunk_id,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0,
                    length: json_content.len(),
                    frame: chunk_id as u32,
                    extra: HashMap::new(),
                },
                file_type: DataFileType::Parquet,
                strategy: ChunkingStrategy::ByRows(rows_per_chunk),
                line_range: Some((start_row, end_row)),
                columns: Some(headers.clone()),
                row_count: Some(end_row - start_row),
                code_metadata: None,
                data_stats: Some(data_stats),
            };

            let format_info = FormatInfo {
                headers: Some(headers.clone()),
                delimiter: None,
                encoding: Some("utf-8".to_string()),
                schema: Some(self.get_parquet_schema(&chunk_df)?),
            };

            chunks.push(DataChunk {
                content: json_content,
                metadata,
                format_info: Some(format_info),
            });

            chunk_id += 1;
        }

        Ok(chunks)
    }

    /// Convert DataFrame to JSON string
    fn dataframe_to_json(&self, df: &polars::prelude::DataFrame) -> Result<String> {
        let mut records = Vec::new();

        for row_idx in 0..df.height() {
            let mut record = serde_json::Map::new();

            for col_name in df.get_column_names() {
                if let Ok(column) = df.column(col_name) {
                    let value = column.get(row_idx).unwrap();
                    let json_value = match value {
                        polars::prelude::AnyValue::Null => serde_json::Value::Null,
                        polars::prelude::AnyValue::Boolean(b) => serde_json::Value::Bool(b),
                        _ => serde_json::Value::String(value.to_string()),
                    };
                    record.insert(col_name.to_string(), json_value);
                }
            }
            records.push(serde_json::Value::Object(record));
        }

        serde_json::to_string_pretty(&records)
            .map_err(|e| MemvidError::data_processing(format!("JSON serialization error: {}", e)))
    }

    /// Get Parquet schema information
    fn get_parquet_schema(&self, df: &polars::prelude::DataFrame) -> Result<String> {
        let schema_info: Vec<String> = df.get_columns().iter().map(|col| {
            format!("{}: {}", col.name(), col.dtype())
        }).collect();

        Ok(format!("Schema: [{}]", schema_info.join(", ")))
    }

    /// Process JSON file
    async fn process_json(&self, file_path: &str) -> Result<Vec<DataChunk>> {
        let content = std::fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Check if it's JSON Lines format
        let is_jsonl = lines.len() > 1 && lines.iter().all(|line| {
            !line.trim().is_empty() && serde_json::from_str::<serde_json::Value>(line).is_ok()
        });

        if is_jsonl {
            self.process_jsonl_content(&content, file_path).await
        } else {
            self.process_single_json_content(&content, file_path).await
        }
    }

    /// Process JSON Lines content
    async fn process_jsonl_content(&self, content: &str, file_path: &str) -> Result<Vec<DataChunk>> {
        let lines: Vec<&str> = content.lines().filter(|line| !line.trim().is_empty()).collect();
        let lines_per_chunk = std::cmp::max(self.config.text.chunk_size / 200, 10); // Estimate 200 chars per JSON line

        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        for chunk_lines in lines.chunks(lines_per_chunk) {
            let chunk_content = chunk_lines.join("\n");

            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: chunk_id,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0,
                    length: chunk_content.len(),
                    frame: chunk_id as u32,
                    extra: HashMap::new(),
                },
                file_type: DataFileType::Json,
                strategy: ChunkingStrategy::ByLines(lines_per_chunk),
                line_range: Some((chunk_id * lines_per_chunk, (chunk_id + 1) * lines_per_chunk)),
                columns: None,
                row_count: Some(chunk_lines.len()),
                code_metadata: None,
                data_stats: None,
            };

            chunks.push(DataChunk {
                content: chunk_content,
                metadata,
                format_info: Some(FormatInfo {
                    headers: None,
                    delimiter: None,
                    encoding: Some("utf-8".to_string()),
                    schema: None,
                }),
            });

            chunk_id += 1;
        }

        Ok(chunks)
    }

    /// Process single JSON file content
    async fn process_single_json_content(&self, content: &str, file_path: &str) -> Result<Vec<DataChunk>> {
        // For large JSON files, we'll chunk by logical structure or size
        let chunk_size = self.config.text.chunk_size;
        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        if content.len() <= chunk_size {
            // Small JSON, single chunk
            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: 0,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0,
                    length: content.len(),
                    frame: 0,
                    extra: HashMap::new(),
                },
                file_type: DataFileType::Json,
                strategy: ChunkingStrategy::BySize(content.len()),
                line_range: None,
                columns: None,
                row_count: None,
                code_metadata: None,
                data_stats: None,
            };

            chunks.push(DataChunk {
                content: content.to_string(),
                metadata,
                format_info: Some(FormatInfo {
                    headers: None,
                    delimiter: None,
                    encoding: Some("utf-8".to_string()),
                    schema: None,
                }),
            });
        } else {
            // Large JSON, chunk by size with attempt to preserve structure
            for (start, chunk_content) in self.chunk_json_content(content, chunk_size).iter().enumerate() {
                let metadata = DataChunkMetadata {
                    base: ChunkMetadata {
                        id: chunk_id,
                        source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                        page: None,
                        char_offset: start * chunk_size,
                        length: chunk_content.len(),
                        frame: chunk_id as u32,
                        extra: HashMap::new(),
                    },
                    file_type: DataFileType::Json,
                    strategy: ChunkingStrategy::BySize(chunk_size),
                    line_range: None,
                    columns: None,
                    row_count: None,
                    code_metadata: None,
                    data_stats: None,
                };

                chunks.push(DataChunk {
                    content: chunk_content.clone(),
                    metadata,
                    format_info: Some(FormatInfo {
                        headers: None,
                        delimiter: None,
                        encoding: Some("utf-8".to_string()),
                        schema: None,
                    }),
                });

                chunk_id += 1;
            }
        }

        Ok(chunks)
    }

    /// Chunk JSON content while trying to preserve structure
    fn chunk_json_content(&self, content: &str, chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut brace_count = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for ch in content.chars() {
            current_chunk.push(ch);

            if !escape_next {
                match ch {
                    '"' if !escape_next => in_string = !in_string,
                    '{' | '[' if !in_string => brace_count += 1,
                    '}' | ']' if !in_string => brace_count -= 1,
                    '\\' if in_string => escape_next = true,
                    _ => {}
                }
            } else {
                escape_next = false;
            }

            // If we've reached chunk size and we're at a balanced point
            if current_chunk.len() >= chunk_size && brace_count == 0 && !in_string {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }
        }

        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }

    /// Process code files (Rust, JavaScript, Python, etc.)
    async fn process_code(&self, file_path: &str, file_type: DataFileType) -> Result<Vec<DataChunk>> {
        let content = std::fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Use logical chunking for code (by functions, classes, etc.)
        let chunks = self.chunk_code_by_logic(&content, &lines, file_type.clone())?;

        let mut data_chunks = Vec::new();
        for (chunk_id, (chunk_content, line_range, code_metadata)) in chunks.into_iter().enumerate() {
            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: chunk_id,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0,
                    length: chunk_content.len(),
                    frame: chunk_id as u32,
                    extra: HashMap::new(),
                },
                file_type: file_type.clone(),
                strategy: ChunkingStrategy::ByLogicalUnits,
                line_range: Some(line_range),
                columns: None,
                row_count: None,
                code_metadata: Some(code_metadata),
                data_stats: None,
            };

            data_chunks.push(DataChunk {
                content: chunk_content,
                metadata,
                format_info: Some(FormatInfo {
                    headers: None,
                    delimiter: None,
                    encoding: Some("utf-8".to_string()),
                    schema: None,
                }),
            });
        }

        Ok(data_chunks)
    }

    /// Chunk code by logical units (functions, classes, etc.)
    fn chunk_code_by_logic(&self, content: &str, lines: &[&str], file_type: DataFileType) -> Result<Vec<(String, (usize, usize), CodeMetadata)>> {
        match file_type {
            DataFileType::Rust => self.chunk_rust_code(content, lines),
            DataFileType::Python => self.chunk_python_code(content, lines),
            DataFileType::JavaScript | DataFileType::TypeScript => self.chunk_js_code(content, lines),
            _ => self.chunk_generic_code(content, lines, file_type),
        }
    }

    /// Chunk Rust code by functions and impl blocks
    fn chunk_rust_code(&self, content: &str, lines: &[&str]) -> Result<Vec<(String, (usize, usize), CodeMetadata)>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_start = 0;
        let mut brace_count = 0;
        let mut in_function = false;
        let mut functions = Vec::new();
        let mut imports = Vec::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Track imports
            if trimmed.starts_with("use ") || trimmed.starts_with("extern ") {
                imports.push(trimmed.to_string());
            }

            // Track function definitions
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") ||
               trimmed.starts_with("async fn ") || trimmed.starts_with("pub async fn ") {
                if let Some(fn_name) = self.extract_rust_function_name(trimmed) {
                    functions.push(fn_name);
                }
                in_function = true;
                if current_chunk.is_empty() {
                    current_start = line_idx;
                }
            }

            // Track braces
            brace_count += line.matches('{').count() as i32;
            brace_count -= line.matches('}').count() as i32;

            current_chunk.push(line.to_string());

            // End of logical unit
            if in_function && brace_count == 0 && !trimmed.is_empty() {
                let chunk_content = current_chunk.join("\n");
                let complexity = self.calculate_rust_complexity(&chunk_content);

                let code_metadata = CodeMetadata {
                    language: "rust".to_string(),
                    functions: functions.clone(),
                    classes: Vec::new(), // Rust doesn't have classes
                    imports: imports.clone(),
                    comment_ratio: self.calculate_comment_ratio(&chunk_content),
                    complexity: Some(complexity),
                };

                chunks.push((chunk_content, (current_start, line_idx + 1), code_metadata));

                current_chunk.clear();
                functions.clear();
                in_function = false;
            }

            // If chunk gets too large, split it
            if current_chunk.len() > 100 {
                let chunk_content = current_chunk.join("\n");
                let complexity = self.calculate_rust_complexity(&chunk_content);

                let code_metadata = CodeMetadata {
                    language: "rust".to_string(),
                    functions: functions.clone(),
                    classes: Vec::new(),
                    imports: imports.clone(),
                    comment_ratio: self.calculate_comment_ratio(&chunk_content),
                    complexity: Some(complexity),
                };

                chunks.push((chunk_content, (current_start, line_idx + 1), code_metadata));

                current_chunk.clear();
                functions.clear();
                current_start = line_idx + 1;
                in_function = false;
            }
        }

        // Handle remaining content
        if !current_chunk.is_empty() {
            let chunk_content = current_chunk.join("\n");
            let complexity = self.calculate_rust_complexity(&chunk_content);

            let code_metadata = CodeMetadata {
                language: "rust".to_string(),
                functions,
                classes: Vec::new(),
                imports,
                comment_ratio: self.calculate_comment_ratio(&chunk_content),
                complexity: Some(complexity),
            };

            chunks.push((chunk_content, (current_start, lines.len()), code_metadata));
        }

        Ok(chunks)
    }

    /// Extract function name from Rust function definition
    fn extract_rust_function_name(&self, line: &str) -> Option<String> {
        let line = line.trim();
        if let Some(fn_pos) = line.find("fn ") {
            let after_fn = &line[fn_pos + 3..];
            if let Some(paren_pos) = after_fn.find('(') {
                let name = after_fn[..paren_pos].trim();
                return Some(name.to_string());
            }
        }
        None
    }

    /// Calculate Rust code complexity
    fn calculate_rust_complexity(&self, content: &str) -> ComplexityMetrics {
        let lines: Vec<&str> = content.lines().collect();
        let loc = lines.iter().filter(|line| !line.trim().is_empty()).count() as u32;
        let comments = lines.iter().filter(|line| line.trim().starts_with("//")).count() as u32;

        // Simple cyclomatic complexity calculation
        let mut complexity = 1; // Base complexity
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.contains("if ") || trimmed.contains("else if ") ||
               trimmed.contains("while ") || trimmed.contains("for ") ||
               trimmed.contains("match ") || trimmed.contains("loop ") {
                complexity += 1;
            }
        }

        // Calculate max nesting depth
        let mut max_depth = 0;
        let mut current_depth = 0;
        for line in &lines {
            current_depth += line.matches('{').count() as u32;
            max_depth = max_depth.max(current_depth);
            current_depth = current_depth.saturating_sub(line.matches('}').count() as u32);
        }

        ComplexityMetrics {
            cyclomatic: complexity,
            loc,
            comments,
            max_depth,
        }
    }

    /// Calculate comment ratio in code
    fn calculate_comment_ratio(&self, content: &str) -> f64 {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len() as f64;
        if total_lines == 0.0 {
            return 0.0;
        }

        let comment_lines = lines.iter().filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("//") || trimmed.starts_with("/*") ||
            trimmed.starts_with("*") || trimmed.starts_with("#")
        }).count() as f64;

        comment_lines / total_lines
    }

    /// Chunk Python code by functions and classes
    fn chunk_python_code(&self, content: &str, lines: &[&str]) -> Result<Vec<(String, (usize, usize), CodeMetadata)>> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_start = 0;
        let mut indent_level = 0;
        let mut functions = Vec::new();
        let mut classes = Vec::new();
        let mut imports = Vec::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Track imports
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                imports.push(trimmed.to_string());
            }

            // Track function and class definitions
            if trimmed.starts_with("def ") {
                if let Some(fn_name) = self.extract_python_function_name(trimmed) {
                    functions.push(fn_name);
                }
                if current_chunk.is_empty() {
                    current_start = line_idx;
                }
            } else if trimmed.starts_with("class ") {
                if let Some(class_name) = self.extract_python_class_name(trimmed) {
                    classes.push(class_name);
                }
                if current_chunk.is_empty() {
                    current_start = line_idx;
                }
            }

            current_chunk.push(line.to_string());

            // Check if we're at the end of a logical unit (function/class)
            let current_indent = line.len() - line.trim_start().len();
            if line_idx > 0 && current_indent == 0 && !trimmed.is_empty() &&
               !trimmed.starts_with("def ") && !trimmed.starts_with("class ") {

                if !current_chunk.is_empty() {
                    let chunk_content = current_chunk.join("\n");
                    let complexity = self.calculate_python_complexity(&chunk_content);

                    let code_metadata = CodeMetadata {
                        language: "python".to_string(),
                        functions: functions.clone(),
                        classes: classes.clone(),
                        imports: imports.clone(),
                        comment_ratio: self.calculate_comment_ratio(&chunk_content),
                        complexity: Some(complexity),
                    };

                    chunks.push((chunk_content, (current_start, line_idx), code_metadata));

                    current_chunk.clear();
                    functions.clear();
                    classes.clear();
                    current_start = line_idx;
                }
            }

            // If chunk gets too large, split it
            if current_chunk.len() > 100 {
                let chunk_content = current_chunk.join("\n");
                let complexity = self.calculate_python_complexity(&chunk_content);

                let code_metadata = CodeMetadata {
                    language: "python".to_string(),
                    functions: functions.clone(),
                    classes: classes.clone(),
                    imports: imports.clone(),
                    comment_ratio: self.calculate_comment_ratio(&chunk_content),
                    complexity: Some(complexity),
                };

                chunks.push((chunk_content, (current_start, line_idx + 1), code_metadata));

                current_chunk.clear();
                functions.clear();
                classes.clear();
                current_start = line_idx + 1;
            }
        }

        // Handle remaining content
        if !current_chunk.is_empty() {
            let chunk_content = current_chunk.join("\n");
            let complexity = self.calculate_python_complexity(&chunk_content);

            let code_metadata = CodeMetadata {
                language: "python".to_string(),
                functions,
                classes,
                imports,
                comment_ratio: self.calculate_comment_ratio(&chunk_content),
                complexity: Some(complexity),
            };

            chunks.push((chunk_content, (current_start, lines.len()), code_metadata));
        }

        Ok(chunks)
    }

    /// Extract function name from Python function definition
    fn extract_python_function_name(&self, line: &str) -> Option<String> {
        if let Some(def_pos) = line.find("def ") {
            let after_def = &line[def_pos + 4..];
            if let Some(paren_pos) = after_def.find('(') {
                let name = after_def[..paren_pos].trim();
                return Some(name.to_string());
            }
        }
        None
    }

    /// Extract class name from Python class definition
    fn extract_python_class_name(&self, line: &str) -> Option<String> {
        if let Some(class_pos) = line.find("class ") {
            let after_class = &line[class_pos + 6..];
            let end_pos = after_class.find('(').or_else(|| after_class.find(':')).unwrap_or(after_class.len());
            let name = after_class[..end_pos].trim();
            return Some(name.to_string());
        }
        None
    }

    /// Calculate Python code complexity
    fn calculate_python_complexity(&self, content: &str) -> ComplexityMetrics {
        let lines: Vec<&str> = content.lines().collect();
        let loc = lines.iter().filter(|line| !line.trim().is_empty()).count() as u32;
        let comments = lines.iter().filter(|line| line.trim().starts_with("#")).count() as u32;

        // Simple cyclomatic complexity calculation
        let mut complexity = 1;
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.starts_with("if ") || trimmed.starts_with("elif ") ||
               trimmed.starts_with("while ") || trimmed.starts_with("for ") ||
               trimmed.starts_with("try:") || trimmed.starts_with("except ") {
                complexity += 1;
            }
        }

        // Calculate max indentation depth as proxy for nesting
        let max_depth = lines.iter().map(|line| {
            (line.len() - line.trim_start().len()) / 4 // Assuming 4-space indentation
        }).max().unwrap_or(0) as u32;

        ComplexityMetrics {
            cyclomatic: complexity,
            loc,
            comments,
            max_depth,
        }
    }

    /// Chunk JavaScript/TypeScript code
    fn chunk_js_code(&self, content: &str, lines: &[&str]) -> Result<Vec<(String, (usize, usize), CodeMetadata)>> {
        // Similar to Rust but with JS-specific patterns
        self.chunk_generic_code(content, lines, DataFileType::JavaScript)
    }

    /// Generic code chunking for other languages
    fn chunk_generic_code(&self, content: &str, lines: &[&str], file_type: DataFileType) -> Result<Vec<(String, (usize, usize), CodeMetadata)>> {
        let lines_per_chunk = std::cmp::max(self.config.text.chunk_size / 50, 20); // Estimate 50 chars per line
        let mut chunks = Vec::new();

        for (chunk_idx, chunk_lines) in lines.chunks(lines_per_chunk).enumerate() {
            let chunk_content = chunk_lines.join("\n");
            let start_line = chunk_idx * lines_per_chunk;
            let end_line = start_line + chunk_lines.len();

            let code_metadata = CodeMetadata {
                language: file_type.syntax_language().to_string(),
                functions: Vec::new(), // Could be enhanced with language-specific parsing
                classes: Vec::new(),
                imports: Vec::new(),
                comment_ratio: self.calculate_comment_ratio(&chunk_content),
                complexity: Some(ComplexityMetrics {
                    cyclomatic: 1,
                    loc: chunk_lines.iter().filter(|line| !line.trim().is_empty()).count() as u32,
                    comments: chunk_lines.iter().filter(|line| {
                        let trimmed = line.trim();
                        trimmed.starts_with("//") || trimmed.starts_with("#") ||
                        trimmed.starts_with("/*") || trimmed.starts_with("<!--")
                    }).count() as u32,
                    max_depth: 0,
                }),
            };

            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: chunk_idx,
                    source: None,
                    page: None,
                    char_offset: 0,
                    length: chunk_content.len(),
                    frame: chunk_idx as u32,
                    extra: HashMap::new(),
                },
                file_type: file_type.clone(),
                strategy: ChunkingStrategy::ByLines(lines_per_chunk),
                line_range: Some((start_line, end_line)),
                columns: None,
                row_count: None,
                code_metadata: Some(code_metadata),
                data_stats: None,
            };

            chunks.push((chunk_content, (start_line, end_line), metadata.code_metadata.unwrap()));
        }

        Ok(chunks)
    }

    /// Process log files
    async fn process_log(&self, file_path: &str) -> Result<Vec<DataChunk>> {
        let content = std::fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Group log entries by time periods or logical units
        let entries_per_chunk = std::cmp::max(self.config.text.chunk_size / 100, 50); // Estimate 100 chars per log line
        let mut chunks = Vec::new();

        for (chunk_idx, chunk_lines) in lines.chunks(entries_per_chunk).enumerate() {
            let chunk_content = chunk_lines.join("\n");
            let start_line = chunk_idx * entries_per_chunk;
            let end_line = start_line + chunk_lines.len();

            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: chunk_idx,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0,
                    length: chunk_content.len(),
                    frame: chunk_idx as u32,
                    extra: HashMap::new(),
                },
                file_type: DataFileType::Log,
                strategy: ChunkingStrategy::ByLines(entries_per_chunk),
                line_range: Some((start_line, end_line)),
                columns: None,
                row_count: Some(chunk_lines.len()),
                code_metadata: None,
                data_stats: None,
            };

            chunks.push(DataChunk {
                content: chunk_content,
                metadata,
                format_info: Some(FormatInfo {
                    headers: None,
                    delimiter: None,
                    encoding: Some("utf-8".to_string()),
                    schema: None,
                }),
            });
        }

        Ok(chunks)
    }

    /// Process structured text files (YAML, TOML, XML, etc.)
    async fn process_structured_text(&self, file_path: &str, file_type: DataFileType) -> Result<Vec<DataChunk>> {
        let content = std::fs::read_to_string(file_path)?;

        // For structured text, we'll chunk by logical sections or size
        let chunk_size = self.config.text.chunk_size;
        let mut chunks = Vec::new();

        if content.len() <= chunk_size {
            // Small file, single chunk
            let metadata = DataChunkMetadata {
                base: ChunkMetadata {
                    id: 0,
                    source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                    page: None,
                    char_offset: 0,
                    length: content.len(),
                    frame: 0,
                    extra: HashMap::new(),
                },
                file_type,
                strategy: ChunkingStrategy::BySize(content.len()),
                line_range: None,
                columns: None,
                row_count: None,
                code_metadata: None,
                data_stats: None,
            };

            chunks.push(DataChunk {
                content,
                metadata,
                format_info: Some(FormatInfo {
                    headers: None,
                    delimiter: None,
                    encoding: Some("utf-8".to_string()),
                    schema: None,
                }),
            });
        } else {
            // Large file, chunk by size
            for (chunk_idx, chunk_start) in (0..content.len()).step_by(chunk_size).enumerate() {
                let chunk_end = std::cmp::min(chunk_start + chunk_size, content.len());
                let chunk_content = content[chunk_start..chunk_end].to_string();

                let metadata = DataChunkMetadata {
                    base: ChunkMetadata {
                        id: chunk_idx,
                        source: Some(Path::new(file_path).file_name().unwrap().to_string_lossy().to_string()),
                        page: None,
                        char_offset: chunk_start,
                        length: chunk_content.len(),
                        frame: chunk_idx as u32,
                        extra: HashMap::new(),
                    },
                    file_type: file_type.clone(),
                    strategy: ChunkingStrategy::BySize(chunk_size),
                    line_range: None,
                    columns: None,
                    row_count: None,
                    code_metadata: None,
                    data_stats: None,
                };

                chunks.push(DataChunk {
                    content: chunk_content,
                    metadata,
                    format_info: Some(FormatInfo {
                        headers: None,
                        delimiter: None,
                        encoding: Some("utf-8".to_string()),
                        schema: None,
                    }),
                });
            }
        }

        Ok(chunks)
    }
}
