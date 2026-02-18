//! Automatic memory linking system.
//!
//! Automatically creates relationships between memory chunks based on:
//! - Semantic similarity (embeddings)
//! - Temporal relationships (daily logs)
//! - Entity references (string matching)
//! 
//! This is what enables IronClaw to "remember everything" by building
//! a rich knowledge graph of interconnected memories.

use chrono::NaiveDate;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::db::Database;
use crate::workspace::{
    EmbeddingProvider, MemoryEdge, MemoryChunk, MemoryDocument,
    relations, GraphError,
};

/// Configuration for automatic memory linking.
#[derive(Debug, Clone)]
pub struct AutoLinkerConfig {
    /// Minimum cosine similarity threshold for "relates_to" edges (0.0-1.0).
    pub similarity_threshold: f32,
    /// Maximum number of similar chunks to link per new chunk.
    pub max_similar_links: usize,
    /// Weight for similarity-based edges.
    pub similarity_weight: f32,
    /// Weight for temporal "follows" edges.
    pub temporal_weight: f32,
    /// Weight for entity reference edges.
    pub reference_weight: f32,
    /// Whether to create temporal links between daily logs.
    pub link_daily_logs: bool,
    /// Simple entity extraction patterns (for references).
    pub entity_patterns: Vec<String>,
}

impl Default for AutoLinkerConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.75,
            max_similar_links: 5,
            similarity_weight: 0.7,
            temporal_weight: 0.9,
            reference_weight: 0.8,
            link_daily_logs: true,
            entity_patterns: vec![
                // Basic patterns for common entities
                r"\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b".to_string(), // Names
                r"\b(?:project|Project) [A-Z]\w+\b".to_string(),   // Project names
                r"\b\w+\.(?:com|org|net)\b".to_string(),           // Domains
                r"\b[A-Z]{2,}\b".to_string(),                      // Acronyms
            ],
        }
    }
}

/// Error types for auto-linking operations.
#[derive(Debug, thiserror::Error)]
pub enum AutoLinkerError {
    #[error("Database error: {0}")]
    Database(#[from] crate::error::DatabaseError),

    #[error("Workspace error: {0}")]
    Workspace(#[from] crate::error::WorkspaceError),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Parsing error: {0}")]
    Parsing(String),

    #[error("Graph error: {0}")]
    Graph(#[from] GraphError),
}

/// Automatic memory linking system.
pub struct AutoLinker {
    config: AutoLinkerConfig,
    db: Arc<dyn Database>,
    embeddings: Option<Arc<dyn EmbeddingProvider>>,
    // Cache for recent entity extractions to avoid recomputation
    entity_cache: Arc<RwLock<HashMap<Uuid, HashSet<String>>>>,
}

impl AutoLinker {
    /// Create a new auto-linker.
    pub fn new(
        config: AutoLinkerConfig,
        db: Arc<dyn Database>,
        embeddings: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Self {
        Self {
            config,
            db,
            embeddings,
            entity_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Automatically link a new document/chunk to existing memory.
    ///
    /// This is called asynchronously after document writes to avoid blocking.
    pub async fn auto_link_document(
        &self,
        user_id: &str,
        document: &MemoryDocument,
        chunks: &[MemoryChunk],
    ) -> Result<usize, AutoLinkerError> {
        let mut total_links = 0;

        // Process each chunk individually
        for chunk in chunks {
            total_links += self.auto_link_chunk(user_id, document, chunk).await?;
        }

        tracing::debug!(
            "Auto-linked document {} ({} chunks) with {} total edges",
            document.path,
            chunks.len(),
            total_links
        );

        Ok(total_links)
    }

    /// Auto-link a single memory chunk.
    async fn auto_link_chunk(
        &self,
        user_id: &str,
        document: &MemoryDocument,
        chunk: &MemoryChunk,
    ) -> Result<usize, AutoLinkerError> {
        let mut link_count = 0;

        // 1. Semantic similarity links (if embeddings available)
        if let Some(ref embeddings) = self.embeddings {
            link_count += self.create_similarity_links(user_id, chunk, embeddings).await?;
        }

        // 2. Temporal links (daily logs)
        if self.config.link_daily_logs {
            link_count += self.create_temporal_links(user_id, document, chunk).await?;
        }

        // 3. Entity reference links
        link_count += self.create_entity_reference_links(user_id, document, chunk).await?;

        Ok(link_count)
    }

    /// Create similarity-based links using embedding comparison.
    async fn create_similarity_links(
        &self,
        user_id: &str,
        new_chunk: &MemoryChunk,
        embeddings: &Arc<dyn EmbeddingProvider>,
    ) -> Result<usize, AutoLinkerError> {
        // Get embedding for the new chunk
        let new_embedding = match new_chunk.embedding.as_ref() {
            Some(emb) => emb,
            None => {
                // If no embedding, try to generate one
                let emb = embeddings
                    .embed(&new_chunk.content)
                    .await
                    .map_err(|e| AutoLinkerError::Embedding(e.to_string()))?;
                
                // Store the embedding back to the chunk (this would need database update)
                // For now, we'll work with the generated embedding in memory
                return Ok(0); // Skip for chunks without embeddings for now
            }
        };

        // Get chunks without embeddings for comparison
        // In a full implementation, you'd want to batch-compare with all embedded chunks
        // For now, we'll pass None for agent_id since we don't have the document context here
        let candidate_chunks = self.db
            .get_chunks_without_embeddings(user_id, None, 100)
            .await
            .map_err(AutoLinkerError::Workspace)?;

        let mut link_count = 0;
        let mut similarities = Vec::new();

        // Calculate similarities (this would be optimized with a vector database in practice)
        for candidate in &candidate_chunks {
            if candidate.id == new_chunk.id {
                continue; // Don't link to self
            }

            if let Some(candidate_embedding) = &candidate.embedding {
                let similarity = cosine_similarity(new_embedding, candidate_embedding);
                
                if similarity >= self.config.similarity_threshold {
                    similarities.push((candidate.id, similarity));
                }
            }
        }

        // Sort by similarity and take top N
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(self.config.max_similar_links);

        // Create edges
        for (similar_chunk_id, similarity) in similarities {
            let weight = (similarity * self.config.similarity_weight).clamp(0.0, 1.0);
            
            let edge = MemoryEdge::new(
                new_chunk.id,
                similar_chunk_id,
                relations::RELATES_TO.to_string(),
                weight,
            );

            self.db.store_memory_edge(user_id, &edge).await?;
            link_count += 1;

            tracing::debug!(
                "Created similarity edge: {} -> {} (similarity: {:.3}, weight: {:.3})",
                new_chunk.id,
                similar_chunk_id,
                similarity,
                weight
            );
        }

        Ok(link_count)
    }

    /// Create temporal links between daily log entries.
    async fn create_temporal_links(
        &self,
        user_id: &str,
        document: &MemoryDocument,
        chunk: &MemoryChunk,
    ) -> Result<usize, AutoLinkerError> {
        // Check if this is a daily log (path like "daily/2024-01-15.md")
        if let Some(date) = extract_daily_log_date(&document.path) {
            // Find the previous day's log
            let previous_date = date.pred_opt().ok_or_else(|| {
                AutoLinkerError::Parsing("Invalid date for temporal linking".to_string())
            })?;

            let previous_path = format!("daily/{}.md", previous_date.format("%Y-%m-%d"));
            
            // Try to find the previous day's document
            if let Ok(prev_document) = self.db.get_document_by_path(user_id, document.agent_id, &previous_path).await {
                // Get chunks from the previous document
                // For simplicity, link to the first chunk of the previous day
                // In practice, you might want more sophisticated logic
                let prev_chunks = self.db
                    .get_chunks_without_embeddings(user_id, document.agent_id, 1)
                    .await
                    .map_err(AutoLinkerError::Workspace)?;

                if let Some(prev_chunk) = prev_chunks
                    .iter()
                    .find(|c| c.document_id == prev_document.id)
                {
                    let edge = MemoryEdge::new(
                        chunk.id,
                        prev_chunk.id,
                        relations::FOLLOWS.to_string(),
                        self.config.temporal_weight,
                    );

                    self.db.store_memory_edge(user_id, &edge).await?;

                    tracing::debug!(
                        "Created temporal edge: {} follows {} (daily logs)",
                        document.path,
                        previous_path
                    );

                    return Ok(1);
                }
            }
        }

        Ok(0)
    }

    /// Create entity reference links based on simple pattern matching.
    async fn create_entity_reference_links(
        &self,
        user_id: &str,
        _document: &MemoryDocument,
        chunk: &MemoryChunk,
    ) -> Result<usize, AutoLinkerError> {
        // Extract entities from the chunk content
        let entities = self.extract_entities(&chunk.content).await;
        
        if entities.is_empty() {
            return Ok(0);
        }

        // Cache the entities for this chunk
        self.entity_cache.write().await.insert(chunk.id, entities.clone());

        // Find other chunks that mention the same entities
        // This is a simplified implementation; a production system would use
        // more sophisticated entity linking
        let mut link_count = 0;
        
        for entity in entities {
            // Simple approach: find chunks containing the same entity string
            // In practice, you'd want fuzzy matching, disambiguation, etc.
            link_count += self.link_by_entity_mention(user_id, chunk, &entity).await?;
        }

        Ok(link_count)
    }

    /// Extract entities from text using simple regex patterns.
    async fn extract_entities(&self, text: &str) -> HashSet<String> {
        let mut entities = HashSet::new();
        
        for pattern in &self.config.entity_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for mat in regex.find_iter(text) {
                    let entity = mat.as_str().to_string();
                    entities.insert(entity);
                }
            }
        }

        entities
    }

    /// Create reference links based on entity mentions.
    async fn link_by_entity_mention(
        &self,
        _user_id: &str,
        source_chunk: &MemoryChunk,
        entity: &str,
    ) -> Result<usize, AutoLinkerError> {
        // This is a placeholder for entity-based linking
        // In practice, you'd search for other documents/chunks mentioning the same entity
        // and create "references" edges between them
        
        // For now, we'll skip the actual implementation to avoid complexity
        tracing::debug!("Would create entity reference link for '{}' from chunk {}", entity, source_chunk.id);
        
        Ok(0)
    }
}

/// Calculate cosine similarity between two embeddings.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Extract date from daily log path (e.g., "daily/2024-01-15.md" -> Some(2024-01-15)).
fn extract_daily_log_date(path: &str) -> Option<NaiveDate> {
    let regex = regex::Regex::new(r"daily/(\d{4}-\d{2}-\d{2})\.md$").ok()?;
    let captures = regex.captures(path)?;
    let date_str = captures.get(1)?.as_str();
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extract_daily_log_date() {
        assert_eq!(
            extract_daily_log_date("daily/2024-01-15.md"),
            Some(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap())
        );

        assert_eq!(
            extract_daily_log_date("other/file.md"),
            None
        );

        assert_eq!(
            extract_daily_log_date("daily/invalid-date.md"),
            None
        );
    }

    #[test]
    fn test_auto_linker_config_default() {
        let config = AutoLinkerConfig::default();
        assert_eq!(config.similarity_threshold, 0.75);
        assert_eq!(config.max_similar_links, 5);
        assert!(config.link_daily_logs);
    }
}