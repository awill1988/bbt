use crate::error::Result;
use crate::llm::gpu::get_gpu_config;
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

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_ctx: usize,
    pub n_batch: usize,
    pub temperature: f32,
    pub verbose: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            temperature: 0.0,
            verbose: false,
        }
    }
}

pub struct LlamaModel {
    _model_path: PathBuf,
    _config: ModelConfig,
}

impl LlamaModel {
    pub fn new(model_path: PathBuf, config: ModelConfig) -> Result<Self> {
        let gpu_config = get_gpu_config();

        if gpu_config.is_accelerated() {
            tracing::info!("gpu acceleration enabled: {} backend", gpu_config.backend.as_str());
        }

        tracing::info!("model initialized for {}", model_path.display());

        Ok(Self {
            _model_path: model_path,
            _config: config,
        })
    }

    #[tracing::instrument(skip(self, messages), fields(message_count = messages.len()))]
    pub fn generate(&self, messages: Vec<Message>) -> Result<String> {
        // for now, return a placeholder since the llm crate API has changed significantly
        // in production, this would use llama-cpp or another inference backend
        let prompt = self.format_chat_prompt(&messages);

        tracing::debug!("formatted prompt length: {} chars", prompt.len());

        // placeholder response for schema generation
        let output = format!(
            "CREATE TABLE bookmarks (\n\
             id INTEGER PRIMARY KEY AUTOINCREMENT,\n\
             url TEXT NOT NULL UNIQUE,\n\
             title TEXT,\n\
             created_at INTEGER,\n\
             folder TEXT,\n\
             embeddings BLOB\n\
             );"
        );

        tracing::debug!("generated {} chars", output.len());

        Ok(output.trim().to_string())
    }

    fn format_chat_prompt(&self, messages: &[Message]) -> String {
        // simple chat template for llama-2-chat format
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::system("test");
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.content, "test");
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.n_ctx, 4096);
        assert_eq!(config.temperature, 0.0);
    }
}
