//! Local embeddings provider using FastEmbed.
//!
//! FastEmbed provides local embeddings with no API dependency.
//! Supports BGE-small and other models with automatic caching.

use async_trait::async_trait;
use std::sync::{Arc, Mutex, OnceLock};
use fastembed::{EmbeddingModel, TextEmbedding, InitOptions};

use crate::workspace::embeddings::{EmbeddingError, EmbeddingProvider};

/// FastEmbed local embeddings provider.
pub struct FastEmbedProvider {
    dimension: usize,
    model_name: String,
    max_length: usize,
}

/// Global instance of the embedding model (expensive to initialize).
static EMBEDDING_MODEL: OnceLock<Arc<Mutex<TextEmbedding>>> = OnceLock::new();

impl FastEmbedProvider {
    /// Create a new FastEmbed provider with BGE-small model.
    pub fn new() -> Self {
        Self {
            dimension: 384,
            model_name: "BAAI/bge-small-en-v1.5".to_string(),
            max_length: 512,
        }
    }

    /// Create with custom model (if supported).
    pub fn with_model(model: &str) -> Result<Self, EmbeddingError> {
        let (dimension, max_length) = match model {
            "BAAI/bge-small-en-v1.5" => (384, 512),
            _ => return Err(EmbeddingError::InvalidResponse(format!(
                "Unsupported model: {}. Supported models: BAAI/bge-small-en-v1.5", 
                model
            ))),
        };

        Ok(Self {
            dimension,
            model_name: model.to_string(),
            max_length,
        })
    }

    /// Initialize the model (called lazily on first use).
    fn get_or_init_model(&self) -> Result<Arc<Mutex<TextEmbedding>>, EmbeddingError> {
        let model_arc = EMBEDDING_MODEL.get_or_init(|| {
            let model = match self.model_name.as_str() {
                "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
                _ => EmbeddingModel::BGESmallENV15, // Default fallback
            };

            // Create InitOptions using the builder pattern (methods take ownership)
            let init_options = InitOptions::new(model)
                .with_show_download_progress(true)
                .with_max_length(self.max_length);

            match TextEmbedding::try_new(init_options) {
                Ok(embedding) => Arc::new(Mutex::new(embedding)),
                Err(e) => {
                    tracing::error!("Failed to initialize FastEmbed: {}", e);
                    panic!("FastEmbed initialization failed: {}", e)
                }
            }
        });
        
        Ok(Arc::clone(model_arc))
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn max_input_length(&self) -> usize {
        // Character-based approximation (tokens are typically 3-4 chars)
        self.max_length * 4
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.len() > self.max_input_length() {
            return Err(EmbeddingError::TextTooLong {
                length: text.len(),
                max: self.max_input_length(),
            });
        }

        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("No embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Get the model instance 
        let model_arc = self.get_or_init_model()?;
        
        // Clone texts for the blocking task
        let texts: Vec<String> = texts.to_vec();
        
        // Run embedding in a blocking task to avoid blocking the async runtime
        let embeddings: Result<Vec<Vec<f32>>, EmbeddingError> = tokio::task::spawn_blocking(move || {
            // Lock the mutex and get mutable access to the model
            let mut model = model_arc.lock().map_err(|e| {
                EmbeddingError::HttpError(format!("Failed to lock model: {}", e))
            })?;
            
            // Use the TextEmbedding::embed method
            let result = model.embed(texts, None).map_err(|e| {
                EmbeddingError::HttpError(format!("FastEmbed error: {}", e))
            })?;
            
            Ok(result)
        })
        .await
        .map_err(|e| EmbeddingError::HttpError(format!("Task join error: {}", e)))?;

        embeddings
    }
}

impl Default for FastEmbedProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fastembed_provider() {
        let provider = FastEmbedProvider::new();
        
        assert_eq!(provider.dimension(), 384);
        assert_eq!(provider.model_name(), "BAAI/bge-small-en-v1.5");
        
        // Test embedding generation
        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding.len(), 384);
        
        // Check that embedding is normalized (approximately unit vector)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_fastembed_batch() {
        let provider = FastEmbedProvider::new();
        
        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = provider.embed_batch(&texts).await.unwrap();
        
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
        
        // Different texts should produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[test]
    fn test_custom_model() {
        let provider = FastEmbedProvider::with_model("BAAI/bge-small-en-v1.5").unwrap();
        assert_eq!(provider.model_name(), "BAAI/bge-small-en-v1.5");
        assert_eq!(provider.dimension(), 384);
    }

    #[test]
    fn test_unsupported_model() {
        let result = FastEmbedProvider::with_model("unsupported/model");
        assert!(result.is_err());
    }
}