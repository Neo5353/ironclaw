//! Memory graph system for relationship tracking.
//!
//! The memory graph enables IronClaw to understand and remember relationships
//! between different pieces of information. This creates a web of connected
//! knowledge that allows for more sophisticated recall and reasoning.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// An edge connecting two memory chunks with a labeled relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    /// Unique identifier for this edge.
    pub id: Uuid,
    /// Source memory chunk ID.
    pub source_id: Uuid,
    /// Target memory chunk ID.
    pub target_id: Uuid,
    /// Relationship type (e.g., "references", "follows", "caused_by").
    pub relation: String,
    /// Relationship strength/confidence (0.0 to 1.0).
    pub weight: f32,
    /// Optional metadata as JSON.
    pub metadata: Option<serde_json::Value>,
    /// When this edge was created.
    pub created_at: DateTime<Utc>,
}

impl MemoryEdge {
    /// Create a new memory edge.
    pub fn new(
        source_id: Uuid,
        target_id: Uuid,
        relation: String,
        weight: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_id,
            target_id,
            relation,
            weight: weight.clamp(0.0, 1.0),
            metadata: None,
            created_at: Utc::now(),
        }
    }

    /// Create a new edge with metadata.
    pub fn with_metadata(
        source_id: Uuid,
        target_id: Uuid,
        relation: String,
        weight: f32,
        metadata: serde_json::Value,
    ) -> Self {
        let mut edge = Self::new(source_id, target_id, relation, weight);
        edge.metadata = Some(metadata);
        edge
    }
}

/// Relationship types for memory edges.
pub mod relations {
    /// One chunk references another (e.g., mentions an entity).
    pub const REFERENCES: &str = "references";
    /// One event/document follows another chronologically.
    pub const FOLLOWS: &str = "follows";
    /// One event was caused by another.
    pub const CAUSED_BY: &str = "caused_by";
    /// General semantic relationship.
    pub const RELATES_TO: &str = "relates_to";
    /// One concept is part of another.
    pub const PART_OF: &str = "part_of";
    /// One version supersedes another.
    pub const SUPERSEDES: &str = "supersedes";
    /// One fact contradicts another.
    pub const CONTRADICTS: &str = "contradicts";
}

/// Trait for managing memory graph operations.
#[async_trait]
pub trait MemoryGraph: Send + Sync {
    /// Add an edge between two memory chunks.
    async fn add_edge(
        &self,
        source: Uuid,
        target: Uuid,
        relation: String,
        weight: f32,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid, GraphError>;

    /// Get all edges originating from a source chunk.
    async fn get_edges_from(&self, source_id: Uuid) -> Result<Vec<MemoryEdge>, GraphError>;

    /// Get all edges pointing to a target chunk.
    async fn get_edges_to(&self, target_id: Uuid) -> Result<Vec<MemoryEdge>, GraphError>;

    /// Find related chunks via graph traversal.
    ///
    /// Performs BFS/DFS traversal starting from the given chunk,
    /// returning related chunks with their cumulative relationship strength.
    /// Weight decays with distance: weight * decay^depth.
    async fn get_related(
        &self,
        id: Uuid,
        max_depth: usize,
        limit: usize,
        decay: f32,
    ) -> Result<Vec<(Uuid, f32)>, GraphError>;

    /// Remove an edge by ID.
    async fn remove_edge(&self, id: Uuid) -> Result<(), GraphError>;

    /// Get all edges between two specific chunks.
    async fn get_edges_between(
        &self,
        source: Uuid,
        target: Uuid,
    ) -> Result<Vec<MemoryEdge>, GraphError>;

    /// Get edge statistics for debugging.
    async fn get_stats(&self) -> Result<GraphStats, GraphError>;
}

/// Graph operation errors.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Edge not found: {0}")]
    EdgeNotFound(Uuid),

    #[error("Invalid weight: {0} (must be 0.0-1.0)")]
    InvalidWeight(f32),

    #[error("Circular reference detected")]
    CircularReference,

    #[error("Graph traversal depth exceeded")]
    DepthExceeded,
}

/// Graph statistics for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of edges.
    pub edge_count: usize,
    /// Number of unique source nodes.
    pub source_nodes: usize,
    /// Number of unique target nodes.
    pub target_nodes: usize,
    /// Most common relation types.
    pub top_relations: Vec<(String, usize)>,
    /// Average edge weight.
    pub avg_weight: f32,
}

/// In-memory implementation of MemoryGraph for testing and development.
pub struct InMemoryGraph {
    edges: tokio::sync::RwLock<HashMap<Uuid, MemoryEdge>>,
    // Indexes for fast lookups
    by_source: tokio::sync::RwLock<HashMap<Uuid, HashSet<Uuid>>>,
    by_target: tokio::sync::RwLock<HashMap<Uuid, HashSet<Uuid>>>,
}

impl InMemoryGraph {
    /// Create a new in-memory graph.
    pub fn new() -> Self {
        Self {
            edges: tokio::sync::RwLock::new(HashMap::new()),
            by_source: tokio::sync::RwLock::new(HashMap::new()),
            by_target: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Perform BFS traversal to find related chunks.
    async fn bfs_related(
        &self,
        start_id: Uuid,
        max_depth: usize,
        limit: usize,
        decay: f32,
    ) -> Result<Vec<(Uuid, f32)>, GraphError> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut weights = HashMap::new();

        // Start with the initial node
        queue.push_back((start_id, 0, 1.0)); // (node_id, depth, weight)
        visited.insert(start_id);

        let edges = self.edges.read().await;

        while let Some((current_id, depth, current_weight)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Find outgoing edges from current node
            if let Some(edge_ids) = self.by_source.read().await.get(&current_id) {
                for edge_id in edge_ids {
                    if let Some(edge) = edges.get(edge_id) {
                        let target = edge.target_id;
                        
                        // Skip if already visited
                        if visited.contains(&target) {
                            continue;
                        }
                        
                        let new_weight = current_weight * edge.weight * decay.powi(depth as i32);
                        
                        // Update or set weight for this target
                        weights.entry(target)
                            .and_modify(|w| *w = f32::max(*w, new_weight))
                            .or_insert(new_weight);
                        
                        visited.insert(target);
                        queue.push_back((target, depth + 1, new_weight));
                    }
                }
            }
        }

        // Convert to sorted vector, excluding the start node
        let mut results: Vec<(Uuid, f32)> = weights
            .into_iter()
            .filter(|(id, _)| *id != start_id)
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        
        Ok(results)
    }
}

#[async_trait]
impl MemoryGraph for InMemoryGraph {
    async fn add_edge(
        &self,
        source: Uuid,
        target: Uuid,
        relation: String,
        weight: f32,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid, GraphError> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(GraphError::InvalidWeight(weight));
        }

        let edge = if let Some(meta) = metadata {
            MemoryEdge::with_metadata(source, target, relation, weight, meta)
        } else {
            MemoryEdge::new(source, target, relation, weight)
        };

        let edge_id = edge.id;

        // Update all data structures
        self.edges.write().await.insert(edge_id, edge);
        self.by_source.write().await
            .entry(source)
            .or_insert_with(HashSet::new)
            .insert(edge_id);
        self.by_target.write().await
            .entry(target)
            .or_insert_with(HashSet::new)
            .insert(edge_id);

        Ok(edge_id)
    }

    async fn get_edges_from(&self, source_id: Uuid) -> Result<Vec<MemoryEdge>, GraphError> {
        let edges = self.edges.read().await;
        let by_source = self.by_source.read().await;

        let edge_ids = by_source.get(&source_id).cloned().unwrap_or_default();
        let mut results = Vec::new();

        for edge_id in edge_ids {
            if let Some(edge) = edges.get(&edge_id) {
                results.push(edge.clone());
            }
        }

        Ok(results)
    }

    async fn get_edges_to(&self, target_id: Uuid) -> Result<Vec<MemoryEdge>, GraphError> {
        let edges = self.edges.read().await;
        let by_target = self.by_target.read().await;

        let edge_ids = by_target.get(&target_id).cloned().unwrap_or_default();
        let mut results = Vec::new();

        for edge_id in edge_ids {
            if let Some(edge) = edges.get(&edge_id) {
                results.push(edge.clone());
            }
        }

        Ok(results)
    }

    async fn get_related(
        &self,
        id: Uuid,
        max_depth: usize,
        limit: usize,
        decay: f32,
    ) -> Result<Vec<(Uuid, f32)>, GraphError> {
        if max_depth == 0 || limit == 0 {
            return Ok(Vec::new());
        }

        self.bfs_related(id, max_depth, limit, decay).await
    }

    async fn remove_edge(&self, id: Uuid) -> Result<(), GraphError> {
        let mut edges = self.edges.write().await;
        let edge = edges.remove(&id).ok_or(GraphError::EdgeNotFound(id))?;

        // Remove from indexes
        let mut by_source = self.by_source.write().await;
        let mut by_target = self.by_target.write().await;

        if let Some(source_edges) = by_source.get_mut(&edge.source_id) {
            source_edges.remove(&id);
            if source_edges.is_empty() {
                by_source.remove(&edge.source_id);
            }
        }

        if let Some(target_edges) = by_target.get_mut(&edge.target_id) {
            target_edges.remove(&id);
            if target_edges.is_empty() {
                by_target.remove(&edge.target_id);
            }
        }

        Ok(())
    }

    async fn get_edges_between(
        &self,
        source: Uuid,
        target: Uuid,
    ) -> Result<Vec<MemoryEdge>, GraphError> {
        let edges_from_source = self.get_edges_from(source).await?;
        let results = edges_from_source
            .into_iter()
            .filter(|edge| edge.target_id == target)
            .collect();
        
        Ok(results)
    }

    async fn get_stats(&self) -> Result<GraphStats, GraphError> {
        let edges = self.edges.read().await;
        let by_source = self.by_source.read().await;
        let by_target = self.by_target.read().await;

        let edge_count = edges.len();
        let source_nodes = by_source.len();
        let target_nodes = by_target.len();

        // Count relations
        let mut relation_counts: HashMap<String, usize> = HashMap::new();
        let mut total_weight = 0.0;

        for edge in edges.values() {
            *relation_counts.entry(edge.relation.clone()).or_default() += 1;
            total_weight += edge.weight;
        }

        // Sort relations by count
        let mut top_relations: Vec<(String, usize)> = relation_counts.into_iter().collect();
        top_relations.sort_by(|a, b| b.1.cmp(&a.1));
        top_relations.truncate(5); // Top 5 relations

        let avg_weight = if edge_count > 0 { 
            total_weight / edge_count as f32 
        } else { 
            0.0 
        };

        Ok(GraphStats {
            edge_count,
            source_nodes,
            target_nodes,
            top_relations,
            avg_weight,
        })
    }
}

impl Default for InMemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_retrieve_edge() {
        let graph = InMemoryGraph::new();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();

        let edge_id = graph
            .add_edge(source, target, relations::REFERENCES.to_string(), 0.8, None)
            .await
            .unwrap();

        let edges_from_source = graph.get_edges_from(source).await.unwrap();
        assert_eq!(edges_from_source.len(), 1);
        assert_eq!(edges_from_source[0].id, edge_id);
        assert_eq!(edges_from_source[0].relation, relations::REFERENCES);
        assert_eq!(edges_from_source[0].weight, 0.8);

        let edges_to_target = graph.get_edges_to(target).await.unwrap();
        assert_eq!(edges_to_target.len(), 1);
        assert_eq!(edges_to_target[0].id, edge_id);
    }

    #[tokio::test]
    async fn test_graph_traversal() {
        let graph = InMemoryGraph::new();
        
        // Create a small graph: A -> B -> C
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        graph.add_edge(a, b, relations::FOLLOWS.to_string(), 0.9, None).await.unwrap();
        graph.add_edge(b, c, relations::FOLLOWS.to_string(), 0.8, None).await.unwrap();

        let related = graph.get_related(a, 3, 10, 0.9).await.unwrap();
        
        assert_eq!(related.len(), 2);
        // B should have higher weight than C (direct connection)
        assert!(related[0].1 > related[1].1);
    }

    #[tokio::test]
    async fn test_remove_edge() {
        let graph = InMemoryGraph::new();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();

        let edge_id = graph
            .add_edge(source, target, relations::RELATES_TO.to_string(), 0.5, None)
            .await
            .unwrap();

        // Verify edge exists
        let edges = graph.get_edges_from(source).await.unwrap();
        assert_eq!(edges.len(), 1);

        // Remove edge
        graph.remove_edge(edge_id).await.unwrap();

        // Verify edge is gone
        let edges = graph.get_edges_from(source).await.unwrap();
        assert_eq!(edges.len(), 0);
    }

    #[tokio::test]
    async fn test_graph_stats() {
        let graph = InMemoryGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        graph.add_edge(a, b, relations::REFERENCES.to_string(), 0.8, None).await.unwrap();
        graph.add_edge(b, c, relations::REFERENCES.to_string(), 0.6, None).await.unwrap();
        graph.add_edge(a, c, relations::FOLLOWS.to_string(), 0.7, None).await.unwrap();

        let stats = graph.get_stats().await.unwrap();
        assert_eq!(stats.edge_count, 3);
        assert_eq!(stats.source_nodes, 2); // a, b
        assert_eq!(stats.target_nodes, 2); // b, c
        
        // references should be the top relation (2 occurrences)
        assert_eq!(stats.top_relations[0].0, relations::REFERENCES);
        assert_eq!(stats.top_relations[0].1, 2);
        
        // Average weight should be (0.8 + 0.6 + 0.7) / 3 = 0.7
        assert!((stats.avg_weight - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_memory_edge_creation() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        
        let edge = MemoryEdge::new(source, target, relations::CAUSED_BY.to_string(), 0.95);
        
        assert_eq!(edge.source_id, source);
        assert_eq!(edge.target_id, target);
        assert_eq!(edge.relation, relations::CAUSED_BY);
        assert_eq!(edge.weight, 0.95);
        assert!(edge.metadata.is_none());
    }

    #[test]
    fn test_edge_weight_clamping() {
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        
        let edge_high = MemoryEdge::new(source, target, "test".to_string(), 1.5);
        assert_eq!(edge_high.weight, 1.0);
        
        let edge_low = MemoryEdge::new(source, target, "test".to_string(), -0.5);
        assert_eq!(edge_low.weight, 0.0);
    }
}