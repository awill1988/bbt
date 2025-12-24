use thiserror::Error;

#[derive(Error, Debug)]
pub enum BbtError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("model error: {0}")]
    Model(String),

    #[error("schema inference error: {0}")]
    Schema(String),

    #[error("ddl extraction failed: {0}")]
    DdlExtraction(String),

    #[error("model download failed: {0}")]
    ModelDownload(String),

    #[error("gpu detection failed: {0}")]
    GpuDetection(String),

    #[error("tracing initialization failed: {0}")]
    Tracing(String),
}

pub type Result<T> = std::result::Result<T, BbtError>;
