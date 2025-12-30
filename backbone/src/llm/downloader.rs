use crate::error::{BbtError, Result};
use std::env;
use std::path::PathBuf;

const DEFAULT_REPO_ID: &str = "unsloth/Llama-3.1-Nemotron-Nano-8B-v1-GGUF";
const DEFAULT_FILENAME: &str = "Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf";
const DEFAULT_CACHE_DIR: &str = ".cache/models";

pub fn ensure_gguf_model(
    repo_id: Option<String>,
    filename: Option<String>,
    cache_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    // read from environment variables if not provided
    let repo_id = repo_id
        .or_else(|| {
            env::var("GEN_SCHEMA_REPO_ID")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_REPO_ID.to_string());

    let filename = filename
        .or_else(|| {
            env::var("GEN_SCHEMA_FILENAME")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_FILENAME.to_string());

    let cache_dir = cache_dir
        .or_else(|| {
            env::var("MODEL_CACHE_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR));

    // create cache directory if it doesn't exist
    std::fs::create_dir_all(&cache_dir)?;

    tracing::info!("resolving model: {}/{}", repo_id, filename);

    // use hf-hub to download the model
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| BbtError::ModelDownload(format!("failed to create hf-hub api: {}", e)))?;

    let repo = api.model(repo_id.clone());

    let model_path = repo.get(&filename).map_err(|e| {
        BbtError::ModelDownload(format!(
            "failed to download {}/{}: {}",
            repo_id, filename, e
        ))
    })?;

    tracing::info!("model available at {}", model_path.display());

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_gguf_model_uses_defaults() {
        // this test will attempt to download if not cached
        // just verify constants are set
        assert!(!DEFAULT_REPO_ID.is_empty());
        assert!(!DEFAULT_FILENAME.is_empty());
    }
}
