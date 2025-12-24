pub mod gpu;
pub mod downloader;
pub mod model;

pub use gpu::{detect_gpu_backend, get_gpu_config, GpuBackend, GpuConfig};
pub use downloader::ensure_gguf_model;
pub use model::{LlamaModel, ModelConfig, Message, MessageRole};
