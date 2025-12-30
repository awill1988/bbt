use std::env;

#[cfg(target_os = "linux")]
use std::path::Path;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuBackend {
    Metal,
    Cuda,
    Vulkan,
    Cpu,
}

impl GpuBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuBackend::Metal => "metal",
            GpuBackend::Cuda => "cuda",
            GpuBackend::Vulkan => "vulkan",
            GpuBackend::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub n_gpu_layers: i32,
    pub backend: GpuBackend,
    pub available: bool,
}

impl GpuConfig {
    pub fn is_accelerated(&self) -> bool {
        self.available && self.n_gpu_layers != 0
    }
}

#[cfg(target_os = "linux")]
fn is_wsl() -> bool {
    if let Ok(release) = std::fs::read_to_string("/proc/version") {
        let release_lower = release.to_lowercase();
        return release_lower.contains("microsoft") || release_lower.contains("wsl");
    }
    false
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn has_vulkan() -> bool {
    Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn has_nvidia_gpu() -> bool {
    if is_wsl() {
        let wsl_lib_dir = Path::new("/usr/lib/wsl/lib");
        if wsl_lib_dir.join("libcuda.so.1").exists()
            || wsl_lib_dir.join("nvidia-smi").exists()
        {
            return true;
        }
    }

    let dev_dir = Path::new("/dev");
    if dev_dir.join("nvidia0").exists() || dev_dir.join("nvidiactl").exists() {
        return true;
    }

    if let Ok(output) = Command::new("nvidia-smi").arg("-L").output() {
        return output.status.success();
    }

    false
}

pub fn detect_gpu_backend() -> GpuBackend {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        GpuBackend::Metal
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        #[cfg(target_os = "linux")]
        {
            if has_nvidia_gpu() {
                return GpuBackend::Cuda;
            }
        }

        if has_vulkan() {
            return GpuBackend::Vulkan;
        }

        GpuBackend::Cpu
    }
}

pub fn get_gpu_config() -> GpuConfig {
    // check for force CPU override
    let force_cpu = read_bool_env(&["FORCE_CPU"]);

    if force_cpu {
        tracing::info!("gpu acceleration disabled (cpu mode forced)");
        return GpuConfig {
            n_gpu_layers: 0,
            backend: GpuBackend::Cpu,
            available: false,
        };
    }

    let backend = detect_gpu_backend();

    if backend == GpuBackend::Cpu {
        tracing::info!("no gpu acceleration available, using cpu");
        return GpuConfig {
            n_gpu_layers: 0,
            backend: GpuBackend::Cpu,
            available: false,
        };
    }

    // get layer count from env or default to -1 (all layers)
    let n_gpu_layers = env::var("GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(-1);

    let layers_str = if n_gpu_layers == -1 {
        "all".to_string()
    } else {
        n_gpu_layers.to_string()
    };

    tracing::info!(
        "gpu acceleration enabled: backend={}, layers={}",
        backend.as_str(),
        layers_str
    );

    GpuConfig {
        n_gpu_layers,
        backend,
        available: true,
    }
}

fn read_bool_env(keys: &[&str]) -> bool {
    for key in keys {
        if let Ok(value) = env::var(key) {
            let value = value.to_lowercase();
            return value == "1" || value == "true" || value == "yes";
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gpu_backend() {
        let backend = detect_gpu_backend();
        // backend should be one of the valid types
        assert!(matches!(
            backend,
            GpuBackend::Metal | GpuBackend::Cuda | GpuBackend::Vulkan | GpuBackend::Cpu
        ));
    }

    #[test]
    fn test_gpu_config_is_accelerated() {
        let config = GpuConfig {
            n_gpu_layers: -1,
            backend: GpuBackend::Cuda,
            available: true,
        };
        assert!(config.is_accelerated());

        let cpu_config = GpuConfig {
            n_gpu_layers: 0,
            backend: GpuBackend::Cpu,
            available: false,
        };
        assert!(!cpu_config.is_accelerated());
    }
}
