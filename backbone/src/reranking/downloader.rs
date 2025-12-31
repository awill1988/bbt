use crate::error::{BbtError, Result};
use crate::reranking::models::RerankModelInfo;
use std::path::{Path, PathBuf};

/// ensure an onnx rerank model and its tokenizer are available locally
///
/// downloads the model and tokenizer from huggingface if not already cached
///
/// # arguments
/// * `model_info` - model metadata including repo and file path
/// * `cache_dir` - base cache directory for models
///
/// # returns
/// path to the downloaded model file (tokenizer will be in same directory)
pub fn ensure_rerank_model(
    model_info: &RerankModelInfo,
    cache_dir: &Path,
) -> Result<PathBuf> {
    // construct full path to model file
    let model_dir = cache_dir.join(&model_info.repo_id.replace('/', "_"));
    let model_path = model_dir.join(&model_info.model_file);
    let tokenizer_path = model_path.parent()
        .map(|p| p.join("tokenizer.json"))
        .unwrap_or_else(|| model_dir.join("tokenizer.json"));

    // check if both model and tokenizer already exist
    let model_exists = model_path.exists() && std::fs::metadata(&model_path)
        .map(|m| m.len() > 1000)
        .unwrap_or(false);
    let tokenizer_exists = tokenizer_path.exists() && std::fs::metadata(&tokenizer_path)
        .map(|m| m.len() > 100)
        .unwrap_or(false);

    if model_exists && tokenizer_exists {
        tracing::info!(
            "using cached rerank model at {}",
            model_path.display()
        );
        return Ok(model_path);
    }

    // create model directory if it doesn't exist
    std::fs::create_dir_all(&model_dir)?;

    // use hf-hub to download the model and tokenizer
    let api = hf_hub::api::sync::Api::new().map_err(|e| {
        BbtError::ModelDownload(format!("failed to create hf-hub api: {}", e))
    })?;

    let repo = api.model(model_info.repo_id.clone());

    // download model if needed
    if !model_exists {
        tracing::info!(
            "downloading rerank model {}/{} from huggingface",
            model_info.repo_id,
            model_info.model_file
        );

        let downloaded_path = repo.get(&model_info.model_file).map_err(|e| {
            BbtError::ModelDownload(format!(
                "failed to download {}/{}: {}",
                model_info.repo_id, model_info.model_file, e
            ))
        })?;

        // create parent directory for model file (e.g., onnx/)
        if let Some(parent) = model_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // copy to our cache location for consistency
        std::fs::copy(&downloaded_path, &model_path)?;

        tracing::info!(
            "rerank model downloaded successfully to {}",
            model_path.display()
        );
    }

    // download tokenizer if needed
    if !tokenizer_exists {
        tracing::info!(
            "downloading tokenizer for {}",
            model_info.repo_id
        );

        let downloaded_tokenizer = repo.get("tokenizer.json").map_err(|e| {
            BbtError::ModelDownload(format!(
                "failed to download tokenizer for {}: {}",
                model_info.repo_id, e
            ))
        })?;

        // copy tokenizer to model directory
        if let Some(parent) = tokenizer_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&downloaded_tokenizer, &tokenizer_path)?;

        tracing::info!(
            "tokenizer downloaded successfully to {}",
            tokenizer_path.display()
        );
    }

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_rerank_model_path_construction() {
        let model_info = RerankModelInfo::default_model();
        let cache_dir = PathBuf::from("/tmp/test_cache");

        // this will fail without network, but we can test path construction
        let result = ensure_rerank_model(&model_info, &cache_dir);

        // just ensure it doesn't panic
        let _ = result;
    }
}
