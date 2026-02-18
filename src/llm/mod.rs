//! LLM integration for the agent.
//!
//! Supports multiple backends:
//! - **NEAR AI** (default): Session-based or API key auth via NEAR AI proxy
//! - **OpenAI**: Direct API access with your own key
//! - **Anthropic**: Direct API access with your own key
//! - **Ollama**: Local model inference
//! - **OpenAI-compatible**: Any endpoint that speaks the OpenAI API

pub mod circuit_breaker;
pub mod costs;
pub mod failover;
mod provider;
mod reasoning;
pub mod response_cache;
mod retry;
mod rig_adapter;
pub mod session;

pub use circuit_breaker::{CircuitBreakerConfig, CircuitBreakerProvider};
pub use failover::{CooldownConfig, FailoverProvider};
pub use provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    Role, ToolCall, ToolCompletionRequest, ToolCompletionResponse, ToolDefinition, ToolResult,
};
pub use reasoning::{
    ActionPlan, Reasoning, ReasoningContext, RespondOutput, RespondResult, TokenUsage,
    ToolSelection,
};
pub use response_cache::{CachedProvider, ResponseCacheConfig};
pub use rig_adapter::RigAdapter;
pub use session::{SessionConfig, SessionManager, create_session_manager};

use std::sync::Arc;

use rig::client::CompletionClient;
use secrecy::ExposeSecret;

use crate::config::{LlmBackend, LlmConfig};
use crate::error::LlmError;

/// Create an LLM provider based on configuration.
///
/// - All backends: Use rig-core adapter with provider-specific clients
/// - SessionManager parameter is maintained for compatibility but not used by local providers
pub fn create_llm_provider(
    config: &LlmConfig,
    _session: Option<Arc<SessionManager>>,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    match config.backend {
        LlmBackend::OpenAi => create_openai_provider(config),
        LlmBackend::Anthropic => create_anthropic_provider(config),
        LlmBackend::Ollama => create_ollama_provider(config),
        LlmBackend::OpenAiCompatible => create_openai_compatible_provider(config),
        LlmBackend::Tinfoil => create_tinfoil_provider(config),
    }
}

// create_llm_provider_with_config removed - was NEAR AI specific

fn create_openai_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let oai = config.openai.as_ref().ok_or_else(|| LlmError::AuthFailed {
        provider: "openai".to_string(),
    })?;

    use rig::providers::openai;

    let client: openai::Client =
        openai::Client::new(oai.api_key.expose_secret()).map_err(|e| LlmError::RequestFailed {
            provider: "openai".to_string(),
            reason: format!("Failed to create OpenAI client: {}", e),
        })?;

    let model = client.completion_model(&oai.model);
    tracing::info!("Using OpenAI direct API (model: {})", oai.model);
    Ok(Arc::new(RigAdapter::new(model, &oai.model)))
}

fn create_anthropic_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let anth = config
        .anthropic
        .as_ref()
        .ok_or_else(|| LlmError::AuthFailed {
            provider: "anthropic".to_string(),
        })?;

    use rig::providers::anthropic;

    let client: anthropic::Client =
        anthropic::Client::new(anth.api_key.expose_secret()).map_err(|e| {
            LlmError::RequestFailed {
                provider: "anthropic".to_string(),
                reason: format!("Failed to create Anthropic client: {}", e),
            }
        })?;

    let model = client.completion_model(&anth.model);
    tracing::info!("Using Anthropic direct API (model: {})", anth.model);
    Ok(Arc::new(RigAdapter::new(model, &anth.model)))
}

fn create_ollama_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let oll = config.ollama.as_ref().ok_or_else(|| LlmError::AuthFailed {
        provider: "ollama".to_string(),
    })?;

    use rig::client::Nothing;
    use rig::providers::ollama;

    let client: ollama::Client = ollama::Client::builder()
        .base_url(&oll.base_url)
        .api_key(Nothing)
        .build()
        .map_err(|e| LlmError::RequestFailed {
            provider: "ollama".to_string(),
            reason: format!("Failed to create Ollama client: {}", e),
        })?;

    let model = client.completion_model(&oll.model);
    tracing::info!(
        "Using Ollama (base_url: {}, model: {})",
        oll.base_url,
        oll.model
    );
    Ok(Arc::new(RigAdapter::new(model, &oll.model)))
}

const TINFOIL_BASE_URL: &str = "https://inference.tinfoil.sh/v1";

fn create_tinfoil_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let tf = config
        .tinfoil
        .as_ref()
        .ok_or_else(|| LlmError::AuthFailed {
            provider: "tinfoil".to_string(),
        })?;

    use rig::providers::openai;

    let client: openai::Client = openai::Client::builder()
        .base_url(TINFOIL_BASE_URL)
        .api_key(tf.api_key.expose_secret())
        .build()
        .map_err(|e| LlmError::RequestFailed {
            provider: "tinfoil".to_string(),
            reason: format!("Failed to create Tinfoil client: {}", e),
        })?;

    // Tinfoil currently only supports the Chat Completions API and not the newer Responses API,
    // so we must explicitly select the completions API here (unlike other OpenAI-compatible providers).
    let client = client.completions_api();
    let model = client.completion_model(&tf.model);
    tracing::info!("Using Tinfoil private inference (model: {})", tf.model);
    Ok(Arc::new(RigAdapter::new(model, &tf.model)))
}

fn create_openai_compatible_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let compat = config
        .openai_compatible
        .as_ref()
        .ok_or_else(|| LlmError::AuthFailed {
            provider: "openai_compatible".to_string(),
        })?;

    use rig::providers::openai;

    let api_key = compat
        .api_key
        .as_ref()
        .map(|k| k.expose_secret().to_string())
        .unwrap_or_else(|| "no-key".to_string());

    let client: openai::Client = openai::Client::builder()
        .base_url(&compat.base_url)
        .api_key(api_key)
        .build()
        .map_err(|e| LlmError::RequestFailed {
            provider: "openai_compatible".to_string(),
            reason: format!("Failed to create OpenAI-compatible client: {}", e),
        })?;

    // OpenAI-compatible providers (e.g. OpenRouter) are most reliable on Chat Completions.
    // This avoids Responses-API-specific assumptions such as required tool call IDs.
    let model = client.completions_api().completion_model(&compat.model);
    tracing::info!(
        "Using OpenAI-compatible endpoint via Chat Completions API (base_url: {}, model: {})",
        compat.base_url,
        compat.model
    );
    Ok(Arc::new(RigAdapter::new(model, &compat.model)))
}

/// Create a cheap/fast LLM provider for lightweight tasks (heartbeat, routing, evaluation).
///
/// Currently not implemented for local providers - returns None.
/// For local providers, the cost difference between models is minimal.
pub fn create_cheap_llm_provider(
    _config: &LlmConfig,
    _session: Option<Arc<SessionManager>>,
) -> Result<Option<Arc<dyn LlmProvider>>, LlmError> {
    // No cheap model concept for local providers - use main provider instead
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmBackend;

    fn test_llm_config() -> LlmConfig {
        LlmConfig {
            backend: LlmBackend::Ollama,
            openai: None,
            anthropic: None,
            ollama: Some(crate::config::OllamaConfig {
                base_url: "http://localhost:11434".to_string(),
                model: "llama3.1".to_string(),
            }),
            openai_compatible: None,
            tinfoil: None,
        }
    }

    #[test]
    fn test_create_cheap_llm_provider_returns_none_when_not_configured() {
        let config = test_llm_config();
        let session = Some(Arc::new(SessionManager::new(SessionConfig::default())));

        let result = create_cheap_llm_provider(&config, session);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_create_cheap_llm_provider_returns_none_for_local_providers() {
        let config = test_llm_config();
        let session = Some(Arc::new(SessionManager::new(SessionConfig::default())));

        let result = create_cheap_llm_provider(&config, session);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_create_cheap_llm_provider_returns_none_for_all_backends() {
        let mut config = test_llm_config();
        config.backend = LlmBackend::OpenAi;

        let session = Some(Arc::new(SessionManager::new(SessionConfig::default())));
        let result = create_cheap_llm_provider(&config, session);

        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
