use crate::embedding::models::EmbeddingModelInfo;
use super::ExecutionProvider;
use std::env;
use std::process::Command;

/// GPU memory information in MiB
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
}

/// GPU capabilities including memory and recommended configuration
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub provider: ExecutionProvider,
    pub total_memory_mb: Option<u64>,
    pub available_memory_mb: Option<u64>,
    pub recommended_workers: usize,
    pub memory_per_worker_mb: Option<u64>,
}

/// Worker configuration for embedding workers
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub workers: usize,
    pub memory_limit_bytes: Option<u64>,
    pub queue_size: usize,
}

impl WorkerConfig {
    /// Fallback configuration (current behavior: 1 worker, 2GB limit)
    pub fn fallback() -> Self {
        WorkerConfig {
            workers: 1,
            memory_limit_bytes: Some(2 * 1024 * 1024 * 1024), // 2 GiB
            queue_size: 4,
        }
    }
}

/// Query CUDA GPU VRAM using nvidia-smi
fn query_cuda_vram() -> Result<GpuMemoryInfo, String> {
    // try standard path first
    let paths = vec![
        "/usr/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi", // wsl2 path
    ];

    let mut last_error = String::new();

    for nvidia_smi_path in paths {
        if !std::path::Path::new(nvidia_smi_path).exists() {
            continue;
        }

        let output = Command::new(nvidia_smi_path)
            .args(&[
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let line = stdout.lines().next().ok_or_else(|| {
                    "nvidia-smi returned no output".to_string()
                })?;

                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() != 2 {
                    return Err(format!("unexpected nvidia-smi output format: {}", line));
                }

                let total_mb = parts[0]
                    .parse::<u64>()
                    .map_err(|e| format!("failed to parse total memory: {}", e))?;
                let available_mb = parts[1]
                    .parse::<u64>()
                    .map_err(|e| format!("failed to parse free memory: {}", e))?;

                tracing::info!(
                    total_mb,
                    available_mb,
                    "detected cuda gpu vram via {}",
                    nvidia_smi_path
                );

                return Ok(GpuMemoryInfo {
                    total_mb,
                    available_mb,
                });
            }
            Ok(output) => {
                last_error = format!(
                    "nvidia-smi at {} failed: {}",
                    nvidia_smi_path,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => {
                last_error = format!("failed to execute nvidia-smi at {}: {}", nvidia_smi_path, e);
            }
        }
    }

    Err(format!(
        "nvidia-smi not found or failed: {}",
        last_error
    ))
}

/// Query CoreML unified memory on macOS
#[cfg(target_os = "macos")]
fn query_coreml_memory() -> Result<GpuMemoryInfo, String> {
    // get total system memory
    let total_output = Command::new("sysctl")
        .args(&["hw.memsize"])
        .output()
        .map_err(|e| format!("failed to execute sysctl for hw.memsize: {}", e))?;

    if !total_output.status.success() {
        return Err(format!(
            "sysctl hw.memsize failed: {}",
            String::from_utf8_lossy(&total_output.stderr)
        ));
    }

    let total_str = String::from_utf8_lossy(&total_output.stdout);
    let total_bytes = total_str
        .split(':')
        .nth(1)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .ok_or_else(|| format!("failed to parse hw.memsize: {}", total_str))?;

    let total_mb = total_bytes / (1024 * 1024);

    // get page size and free count
    let page_size_output = Command::new("sysctl")
        .args(&["vm.pagesize"])
        .output()
        .map_err(|e| format!("failed to execute sysctl for vm.pagesize: {}", e))?;

    let page_size_str = String::from_utf8_lossy(&page_size_output.stdout);
    let page_size = page_size_str
        .split(':')
        .nth(1)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .ok_or_else(|| format!("failed to parse vm.pagesize: {}", page_size_str))?;

    let free_count_output = Command::new("sysctl")
        .args(&["vm.page_free_count"])
        .output()
        .map_err(|e| format!("failed to execute sysctl for vm.page_free_count: {}", e))?;

    let free_count_str = String::from_utf8_lossy(&free_count_output.stdout);
    let free_count = free_count_str
        .split(':')
        .nth(1)
        .and_then(|s| s.trim().parse::<u64>().ok())
        .ok_or_else(|| format!("failed to parse vm.page_free_count: {}", free_count_str))?;

    // available memory = free pages × page size × 0.7 (conservative estimate for unified memory)
    let available_bytes = (free_count * page_size) as f64 * 0.7;
    let available_mb = (available_bytes / (1024.0 * 1024.0)) as u64;

    tracing::info!(
        total_mb,
        available_mb,
        "detected coreml unified memory"
    );

    Ok(GpuMemoryInfo {
        total_mb,
        available_mb,
    })
}

#[cfg(not(target_os = "macos"))]
fn query_coreml_memory() -> Result<GpuMemoryInfo, String> {
    Err("coreml is only available on macos".to_string())
}

/// Estimate memory requirements per worker in MiB
fn estimate_worker_memory(model_info: &EmbeddingModelInfo, batch_size: usize) -> u64 {
    // base model loading overhead (1 GiB minimum)
    let base_mb = 1024u64;

    // per-batch tensor memory
    // formula: batch_size × max_seq_len × (hidden_size + 24) × 4 bytes
    // the +24 accounts for input_ids, attention_mask, token_type_ids (3 × 8 bytes each = 24)
    let hidden_size = model_info.dimensions;
    let max_seq_len = model_info.max_seq_len;
    let tensor_bytes = (batch_size * max_seq_len * (hidden_size + 24) * 4) as u64;
    let tensor_mb = tensor_bytes / (1024 * 1024);

    // safety margin (20% overhead)
    let total_mb = ((base_mb + tensor_mb) as f64 * 1.2) as u64;

    tracing::debug!(
        base_mb,
        tensor_mb,
        total_mb,
        "estimated per-worker memory requirements"
    );

    total_mb
}

/// Detect GPU capabilities for given execution provider
fn detect_gpu_capabilities(provider: ExecutionProvider) -> GpuCapabilities {
    match provider {
        ExecutionProvider::Cuda => match query_cuda_vram() {
            Ok(info) => GpuCapabilities {
                provider,
                total_memory_mb: Some(info.total_mb),
                available_memory_mb: Some(info.available_mb),
                recommended_workers: 0, // calculated later
                memory_per_worker_mb: None, // calculated later
            },
            Err(e) => {
                tracing::warn!("failed to query cuda vram: {}, using fallback", e);
                GpuCapabilities::fallback(provider)
            }
        },
        ExecutionProvider::Coreml => match query_coreml_memory() {
            Ok(info) => GpuCapabilities {
                provider,
                total_memory_mb: Some(info.total_mb),
                available_memory_mb: Some(info.available_mb),
                recommended_workers: 0, // calculated later
                memory_per_worker_mb: None, // calculated later
            },
            Err(e) => {
                tracing::warn!("failed to query coreml memory: {}, using fallback", e);
                GpuCapabilities::fallback(provider)
            }
        },
        ExecutionProvider::Cpu => GpuCapabilities {
            provider,
            total_memory_mb: None,
            available_memory_mb: None,
            recommended_workers: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            memory_per_worker_mb: None,
        },
    }
}

impl GpuCapabilities {
    fn fallback(provider: ExecutionProvider) -> Self {
        GpuCapabilities {
            provider,
            total_memory_mb: None,
            available_memory_mb: None,
            recommended_workers: 1,
            memory_per_worker_mb: Some(2048), // 2 GiB
        }
    }
}

/// Calculate optimal worker configuration based on GPU capabilities
pub fn calculate_worker_config(
    provider: ExecutionProvider,
    model_info: &EmbeddingModelInfo,
    batch_size: usize,
    max_cpu_workers: usize,
) -> WorkerConfig {
    // cpu mode: use available cores
    if provider == ExecutionProvider::Cpu {
        return WorkerConfig {
            workers: max_cpu_workers.max(1),
            memory_limit_bytes: None,
            queue_size: max_cpu_workers.saturating_mul(4).max(1),
        };
    }

    // detect gpu capabilities
    let caps = detect_gpu_capabilities(provider);

    // if no memory info available, use fallback
    let available_mb = match caps.available_memory_mb {
        Some(mb) => mb,
        None => {
            tracing::warn!(
                "gpu memory detection failed, using fallback config: 1 worker, 2 GiB limit"
            );
            return WorkerConfig::fallback();
        }
    };

    // estimate memory per worker
    let estimated_worker_memory_mb = estimate_worker_memory(model_info, batch_size);

    // calculate workers (leave 25% vram headroom)
    let usable_memory_mb = (available_mb as f64 * 0.75) as u64;
    let max_workers = (usable_memory_mb / estimated_worker_memory_mb)
        .max(1)
        .min(max_cpu_workers as u64) as usize;

    // calculate per-worker memory limit
    let memory_limit_mb = usable_memory_mb / max_workers as u64;
    let memory_limit_bytes = memory_limit_mb * 1024 * 1024;

    // queue size = workers × 4
    let queue_size = max_workers.saturating_mul(4);

    // apply environment variable overrides
    let workers = env::var("EMBEDDING_MAX_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(max_workers);

    let memory_limit_bytes = env::var("EMBEDDING_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(memory_limit_bytes);

    tracing::info!(
        provider = %provider.as_str(),
        total_vram_mb = caps.total_memory_mb,
        available_vram_mb = available_mb,
        usable_vram_mb = usable_memory_mb,
        estimated_worker_mb = estimated_worker_memory_mb,
        calculated_workers = max_workers,
        final_workers = workers,
        memory_limit_mb = memory_limit_bytes / (1024 * 1024),
        queue_size,
        "calculated gpu-aware worker configuration"
    );

    WorkerConfig {
        workers,
        memory_limit_bytes: Some(memory_limit_bytes),
        queue_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_worker_memory() {
        let model_info = EmbeddingModelInfo::default_model();
        let batch_size = 32;
        let estimated = estimate_worker_memory(&model_info, batch_size);

        // jina model: 768 dims, 8192 max seq len
        // expected: ~2-3 GB per worker
        assert!(estimated >= 2000);
        assert!(estimated <= 4000);
    }

    #[test]
    fn test_worker_config_fallback() {
        let config = WorkerConfig::fallback();
        assert_eq!(config.workers, 1);
        assert_eq!(config.memory_limit_bytes, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(config.queue_size, 4);
    }

    #[test]
    fn test_calculate_worker_config_cpu() {
        let model_info = EmbeddingModelInfo::default_model();
        let config = calculate_worker_config(ExecutionProvider::Cpu, &model_info, 32, 8);

        // cpu mode should use available cores
        assert!(config.workers > 0);
        assert_eq!(config.memory_limit_bytes, None);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_query_cuda_vram() {
        // skip if nvidia-smi not available
        if !std::path::Path::new("/usr/bin/nvidia-smi").exists()
            && !std::path::Path::new("/usr/lib/wsl/lib/nvidia-smi").exists()
        {
            return;
        }

        let result = query_cuda_vram();
        if let Ok(info) = result {
            assert!(info.total_mb > 0);
            assert!(info.available_mb <= info.total_mb);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_calculate_worker_config_cuda() {
        let model_info = EmbeddingModelInfo::default_model();
        let config = calculate_worker_config(ExecutionProvider::Cuda, &model_info, 32, 8);

        // should have at least 1 worker
        assert!(config.workers >= 1);
        // should have memory limit set
        assert!(config.memory_limit_bytes.is_some());
        // queue size should be workers × 4
        assert_eq!(config.queue_size, config.workers * 4);
    }
}
