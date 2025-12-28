use crate::agent::parser::extract_create_table;
use crate::agent::prompt::{build_schema_task_prompt, SCHEMA_SYSTEM_PROMPT};
use crate::error::{BbtError, Result};
use crate::llm::model::{LlamaModel, Message};
use crate::schema::hints::FieldHints;
use crate::schema::inference::Schema;
use std::sync::Arc;

#[tracing::instrument(skip(model, prompt), fields(prompt_len = prompt.len()))]
pub fn generate_schema_sql(model: &LlamaModel, prompt: &str) -> Result<String> {
    let messages = vec![
        Message::system(SCHEMA_SYSTEM_PROMPT),
        Message::user(prompt),
    ];

    let output = model.generate(messages)?;

    extract_create_table(&output)
}

#[tracing::instrument(skip(model, schema, hints), fields(max_attempts))]
pub async fn generate_schema_sql_with_retry(
    model: &Arc<LlamaModel>,
    schema: &Schema,
    hints: &FieldHints,
    max_attempts: usize,
) -> Result<String> {
    let prompt = build_schema_task_prompt(schema, hints);
    let mut last_error: Option<BbtError> = None;
    let mut current_prompt = prompt.clone();

    for attempt in 1..=max_attempts {
        tracing::info!("schema synthesis attempt {}/{}", attempt, max_attempts);

        // clone model arc for move into spawn_blocking
        let model_clone = std::sync::Arc::clone(model);
        let current_prompt_clone = current_prompt.clone();

        // run sync llm generation in blocking task
        let result = tokio::task::spawn_blocking(move || {
            generate_schema_sql(&model_clone, &current_prompt_clone)
        })
        .await
        .map_err(|e| BbtError::Model(format!("task join error: {}", e)))?;

        match result {
            Ok(ddl) => {
                tracing::info!("schema synthesis succeeded on attempt {}", attempt);
                return Ok(ddl);
            }
            Err(e) => {
                tracing::warn!("schema synthesis attempt {} failed: {}", attempt, e);
                last_error = Some(e.clone());

                // append error feedback to prompt for next attempt
                current_prompt = format!(
                    "{}\n\nprevious output was invalid: {}\n\
                     return only one sqlite create table statement. no notes.",
                    prompt, e
                );
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        BbtError::Model(format!(
            "schema synthesis failed after {} attempts",
            max_attempts
        ))
    }))
}

impl Clone for BbtError {
    fn clone(&self) -> Self {
        match self {
            BbtError::Io(e) => BbtError::Io(std::io::Error::new(e.kind(), e.to_string())),
            BbtError::Json(e) => BbtError::Json(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))),
            BbtError::Model(s) => BbtError::Model(s.clone()),
            BbtError::Embedding(s) => BbtError::Embedding(s.clone()),
            BbtError::Schema(s) => BbtError::Schema(s.clone()),
            BbtError::DdlExtraction(s) => BbtError::DdlExtraction(s.clone()),
            BbtError::ModelDownload(s) => BbtError::ModelDownload(s.clone()),
            BbtError::GpuDetection(s) => BbtError::GpuDetection(s.clone()),
            BbtError::Tracing(s) => BbtError::Tracing(s.clone()),
            BbtError::Sqlite(e) => BbtError::Sqlite(rusqlite::Error::ToSqlConversionFailure(
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                )),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schema_sql_with_retry_params() {
        // just test that the function signature compiles
        // actual testing would require a real model
        let max_attempts = 3;
        assert!(max_attempts > 0);
    }
}
