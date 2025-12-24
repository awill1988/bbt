use std::env;
use std::path::Path;
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

fn is_wsl() -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(release) = std::fs::read_to_string("/proc/version") {
            let release_lower = release.to_lowercase();
            return release_lower.contains("microsoft") || release_lower.contains("wsl");
        }
    }
    false
}

fn has_vulkan() -> bool {
    Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn has_nvidia_gpu() -> bool {
    #[cfg(not(target_os = "linux"))]
    {
        return false;
    }

    #[cfg(target_os = "linux")]
    {
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
}

pub fn detect_gpu_backend() -> GpuBackend {
    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        {
            return GpuBackend::Metal;
        }
    }

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

pub fn get_gpu_config() -> GpuConfig {
    // check for force CPU override
    let force_cpu = env::var("BOOKMARKS_FORCE_CPU")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false);

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
    let n_gpu_layers = env::var("BOOKMARKS_GPU_LAYERS")
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
