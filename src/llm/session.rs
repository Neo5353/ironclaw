//! Session management stub (NEAR AI removed).
//!
//! This module provides stub implementations for compatibility with existing code
//! that expects SessionManager to exist. For local providers, no authentication
//! session is needed.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::LlmError;

/// Stub session data for compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_token: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub auth_provider: Option<String>,
}

/// Configuration for session management (stub).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub auth_base_url: String,
    pub session_path: PathBuf,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            auth_base_url: "".to_string(),
            session_path: PathBuf::from("/tmp/stub-session.json"),
        }
    }
}

/// Stub session manager for compatibility.
#[derive(Debug)]
pub struct SessionManager {
    _config: SessionConfig,
}

impl SessionManager {
    /// Create a new session manager (stub).
    pub fn new(config: SessionConfig) -> Self {
        Self { _config: config }
    }

    /// Create a new session manager asynchronously (stub).
    pub async fn new_async(config: SessionConfig) -> Self {
        Self { _config: config }
    }

    /// Check if we have a valid token (stub - always returns false).
    pub async fn has_token(&self) -> bool {
        false
    }

    /// Ensure we're authenticated (stub - always succeeds).
    pub async fn ensure_authenticated(&self) -> Result<(), LlmError> {
        // For local providers, no authentication needed
        Ok(())
    }

    /// Get session data (stub - returns empty data).
    pub async fn session_data(&self) -> Option<SessionData> {
        None
    }

    /// Get the auth token (stub - returns None).
    pub async fn auth_token(&self) -> Option<String> {
        None
    }

    /// Attach store (stub - does nothing).
    pub async fn attach_store(&self, _db: std::sync::Arc<dyn crate::db::Database>, _user_id: &str) {
        // Stub implementation - does nothing for local providers
    }
}

/// Create a session manager (stub).
pub async fn create_session_manager(config: SessionConfig) -> Arc<SessionManager> {
    Arc::new(SessionManager::new(config))
}

/// Default session path (stub).
pub fn default_session_path() -> PathBuf {
    PathBuf::from("/tmp/stub-session.json")
}