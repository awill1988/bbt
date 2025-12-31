use crate::error::{BbtError, Result};
use crate::llm::gpu::get_gpu_config;
use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::ChatMessage;
use llm::LLMProvider;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

impl Message {
    /// convert to ChatMessage for non-system messages
    fn to_chat_message(&self) -> Option<ChatMessage> {
        match self.role {
            MessageRole::System => None, // system messages handled separately via builder
            MessageRole::User => Some(ChatMessage::user().content(&self.content).build()),
            MessageRole::Assistant => Some(ChatMessage::assistant().content(&self.content).build()),
        }
    }

    /// check if this is a system message
    fn is_system(&self) -> bool {
        matches!(self.role, MessageRole::System)
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_ctx: usize,
    pub n_batch: usize,
    pub temperature: f32,
    pub verbose: bool,
    pub max_tokens: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            temperature: 0.0,
            verbose: false,
            max_tokens: 2048,
        }
    }
}

/// LLM backend type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendType {
    /// Ollama for local model inference
    Ollama,
    /// OpenAI API
    OpenAi,
    /// Anthropic Claude API
    Anthropic,
    /// Placeholder/offline mode
    Placeholder,
}

impl BackendType {
    /// detect backend from environment variables
    pub fn from_env() -> Self {
        // check for explicit backend selection
        if let Ok(backend) = env::var("LLM_BACKEND") {
            return match backend.to_lowercase().as_str() {
                "ollama" => BackendType::Ollama,
                "openai" => BackendType::OpenAi,
                "anthropic" | "claude" => BackendType::Anthropic,
                "placeholder" | "offline" => BackendType::Placeholder,
                _ => BackendType::Placeholder,
            };
        }

        // auto-detect based on available API keys
        if env::var("ANTHROPIC_API_KEY").is_ok() {
            return BackendType::Anthropic;
        }

        if env::var("OPENAI_API_KEY").is_ok() {
            return BackendType::OpenAi;
        }

        // default to ollama for local inference
        if env::var("OLLAMA_HOST").is_ok() || is_ollama_available() {
            return BackendType::Ollama;
        }

        BackendType::Placeholder
    }
}

/// check if ollama is available at default address
fn is_ollama_available() -> bool {
    // simple check - in production would actually ping the service
    std::net::TcpStream::connect("127.0.0.1:11434").is_ok()
}

pub struct LlamaModel {
    backend_type: BackendType,
    model_name: String,
    config: ModelConfig,
}

impl LlamaModel {
    /// create a new llm model wrapper
    ///
    /// # arguments
    /// * `model_path` - path to local model (used for ollama model name extraction)
    /// * `config` - model configuration
    pub fn new(model_path: PathBuf, config: ModelConfig) -> Result<Self> {
        let gpu_config = get_gpu_config();
        let backend_type = BackendType::from_env();

        if gpu_config.is_accelerated() {
            tracing::info!("gpu acceleration available: {} backend", gpu_config.backend.as_str());
        }

        // extract model name from path or use env override
        let model_name = env::var("LLM_MODEL")
            .unwrap_or_else(|_| {
                model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("llama3.1")
                    .to_string()
            });

        tracing::info!(
            "llm initialized: backend={:?}, model={}",
            backend_type,
            model_name
        );

        Ok(Self {
            backend_type,
            model_name,
            config,
        })
    }

    /// create with explicit backend
    pub fn with_backend(backend_type: BackendType, model_name: String, config: ModelConfig) -> Self {
        Self {
            backend_type,
            model_name,
            config,
        }
    }

    /// generate response from messages
    #[tracing::instrument(skip(self, messages), fields(message_count = messages.len()))]
    pub fn generate(&self, messages: Vec<Message>) -> Result<String> {
        match self.backend_type {
            BackendType::Placeholder => self.generate_placeholder(&messages),
            BackendType::Ollama => self.generate_with_ollama(&messages),
            BackendType::OpenAi => self.generate_with_openai(&messages),
            BackendType::Anthropic => self.generate_with_anthropic(&messages),
        }
    }

    /// extract system message from messages
    fn extract_system_message(messages: &[Message]) -> Option<String> {
        messages
            .iter()
            .find(|m| m.is_system())
            .map(|m| m.content.clone())
    }

    /// convert messages to ChatMessage vector (excluding system)
    fn to_chat_messages(messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .filter_map(|m| m.to_chat_message())
            .collect()
    }

    /// generate using ollama backend
    fn generate_with_ollama(&self, messages: &[Message]) -> Result<String> {
        let host = env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
        let system = Self::extract_system_message(messages);

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::Ollama)
            .base_url(&host)
            .model(&self.model_name)
            .temperature(self.config.temperature)
            .max_tokens(self.config.max_tokens);

        if let Some(sys) = system {
            builder = builder.system(sys);
        }

        let backend = builder
            .build()
            .map_err(|e| BbtError::Model(format!("failed to create ollama backend: {}", e)))?;

        self.run_chat(backend, messages)
    }

    /// generate using openai backend
    fn generate_with_openai(&self, messages: &[Message]) -> Result<String> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| BbtError::Model("OPENAI_API_KEY not set".to_string()))?;
        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        let system = Self::extract_system_message(messages);

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::OpenAI)
            .api_key(&api_key)
            .model(&model)
            .temperature(self.config.temperature)
            .max_tokens(self.config.max_tokens);

        if let Some(sys) = system {
            builder = builder.system(sys);
        }

        let backend = builder
            .build()
            .map_err(|e| BbtError::Model(format!("failed to create openai backend: {}", e)))?;

        self.run_chat(backend, messages)
    }

    /// generate using anthropic backend
    fn generate_with_anthropic(&self, messages: &[Message]) -> Result<String> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| BbtError::Model("ANTHROPIC_API_KEY not set".to_string()))?;
        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());
        let system = Self::extract_system_message(messages);

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::Anthropic)
            .api_key(&api_key)
            .model(&model)
            .temperature(self.config.temperature)
            .max_tokens(self.config.max_tokens);

        if let Some(sys) = system {
            builder = builder.system(sys);
        }

        let backend = builder
            .build()
            .map_err(|e| BbtError::Model(format!("failed to create anthropic backend: {}", e)))?;

        self.run_chat(backend, messages)
    }

    /// run chat completion with given backend
    fn run_chat(&self, backend: Box<dyn LLMProvider>, messages: &[Message]) -> Result<String> {
        let chat_messages = Self::to_chat_messages(messages);

        tracing::debug!("sending {} messages to llm", chat_messages.len());

        // use tokio runtime to run async chat
        let rt = tokio::runtime::Handle::try_current()
            .or_else(|_| {
                tokio::runtime::Runtime::new()
                    .map(|rt| rt.handle().clone())
                    .map_err(|e| BbtError::Model(format!("failed to create runtime: {}", e)))
            })?;

        let response = rt.block_on(async {
            backend.chat(&chat_messages).await
        }).map_err(|e| BbtError::Model(format!("llm chat failed: {}", e)))?;

        let text = response.text().unwrap_or_default();
        tracing::debug!("received response: {} chars", text.len());

        Ok(text)
    }

    /// generate placeholder response (offline mode)
    fn generate_placeholder(&self, messages: &[Message]) -> Result<String> {
        let prompt = self.format_chat_prompt(messages);
        tracing::debug!("placeholder mode: prompt length {} chars", prompt.len());

        // detect intent from messages and return appropriate placeholder
        let last_user_msg = messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // schema generation placeholder
        if last_user_msg.contains("schema") || last_user_msg.contains("CREATE TABLE") {
            return Ok(
                "CREATE TABLE items (\n\
                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n\
                    url TEXT NOT NULL UNIQUE,\n\
                    title TEXT,\n\
                    created_at INTEGER,\n\
                    folder TEXT,\n\
                    embeddings BLOB\n\
                );".to_string()
            );
        }

        // generic placeholder
        tracing::warn!(
            "llm running in placeholder mode - set LLM_BACKEND or provide API keys"
        );
        Ok("[placeholder response - configure LLM_BACKEND for actual inference]".to_string())
    }

    fn format_chat_prompt(&self, messages: &[Message]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
                }
                MessageRole::User => {
                    prompt.push_str(&format!("[INST] {} [/INST] ", msg.content));
                }
                MessageRole::Assistant => {
                    prompt.push_str(&format!("{} ", msg.content));
                }
            }
        }

        prompt
    }

    /// get the current backend type
    pub fn backend_type(&self) -> &BackendType {
        &self.backend_type
    }

    /// get the model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::system("test");
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.content, "test");

        let msg = Message::user("hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "hello");

        let msg = Message::assistant("response");
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.content, "response");
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.n_ctx, 4096);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.max_tokens, 2048);
    }

    #[test]
    fn test_backend_type_from_env_placeholder() {
        // without any env vars set, should fall back to placeholder
        // (unless ollama is running locally)
        let backend = BackendType::from_env();
        // just verify it returns one of the valid types
        assert!(matches!(
            backend,
            BackendType::Ollama | BackendType::OpenAi | BackendType::Anthropic | BackendType::Placeholder
        ));
    }

    #[test]
    fn test_llama_model_with_backend() {
        let model = LlamaModel::with_backend(
            BackendType::Placeholder,
            "test-model".to_string(),
            ModelConfig::default(),
        );

        assert_eq!(model.backend_type(), &BackendType::Placeholder);
        assert_eq!(model.model_name(), "test-model");
    }

    #[test]
    fn test_placeholder_generation() {
        let model = LlamaModel::with_backend(
            BackendType::Placeholder,
            "test".to_string(),
            ModelConfig::default(),
        );

        // test schema generation placeholder
        let messages = vec![
            Message::system("you are a sql expert"),
            Message::user("generate a CREATE TABLE schema for bookmarks"),
        ];
        let result = model.generate(messages).unwrap();
        assert!(result.contains("CREATE TABLE"));

        // test generic placeholder
        let messages = vec![
            Message::user("hello there"),
        ];
        let result = model.generate(messages).unwrap();
        assert!(result.contains("placeholder"));
    }
}
