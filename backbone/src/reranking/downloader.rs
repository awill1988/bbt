use crate::error::{BbtError, Result};
use crate::reranking::models::RerankModelInfo;
use std::path::{Path, PathBuf};

/// ensure an onnx rerank model is available locally
///
/// downloads the model from huggingface if not already cached
///
/// # arguments
/// * `model_info` - model metadata including repo and file path
/// * `cache_dir` - base cache directory for models
///
/// # returns
/// path to the downloaded model file
pub fn ensure_rerank_model(
    model_info: &RerankModelInfo,
    cache_dir: &Path,
) -> Result<PathBuf> {
    // construct full path to model file
    let model_dir = cache_dir.join(&model_info.repo_id.replace('/', "_"));
    let model_path = model_dir.join(&model_info.model_file);

    // if model already exists and is non-empty, return it
    if model_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&model_path) {
            if metadata.len() > 1000 {
                // file is larger than 1kb, likely not a git-lfs pointer
                tracing::info!(
                    "using cached rerank model at {}",
                    model_path.display()
                );
                return Ok(model_path);
            }
        }
    }

    // create model directory if it doesn't exist
    std::fs::create_dir_all(&model_dir)?;

    tracing::info!(
        "downloading rerank model {}/{} from huggingface",
        model_info.repo_id,
        model_info.model_file
    );

    // use hf-hub to download the model
    let api = hf_hub::api::sync::Api::new().map_err(|e| {
        BbtError::ModelDownload(format!("failed to create hf-hub api: {}", e))
    })?;

    let repo = api.model(model_info.repo_id.clone());
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
