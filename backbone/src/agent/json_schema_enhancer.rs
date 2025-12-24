use crate::error::{BbtError, Result};
use crate::llm::model::{LlamaModel, Message};
use crate::schema::hints::FieldHints;
use serde_json::Value;
use std::sync::Arc;

const ENHANCEMENT_SYSTEM_PROMPT: &str = "add concise description fields to json schema properties. \
     descriptions should be 5-15 words explaining field purpose. \
     use field hints to inform descriptions. \
     output only valid json with no markdown formatting.";

/// enhance json schema with llm-generated descriptions
#[tracing::instrument(skip(model, json_schema), fields(llm.model = "schema_enhancer"))]
pub async fn enhance_json_schema(
    model: &Arc<LlamaModel>,
    json_schema: &Value,
    hints: &FieldHints,
) -> Result<Value> {
    tracing::info!("enhancing json schema with llm-generated descriptions");

    let hint_context = format!(
        "timestamp fields: {}\nfolder fields: {}",
        if hints.timestamps.is_empty() {
            "none".to_string()
        } else {
            hints.timestamps.join(", ")
        },
        if hints.folders.is_empty() {
            "none".to_string()
        } else {
            hints.folders.join(", ")
        }
    );

    let schema_text = serde_json::to_string_pretty(json_schema)?;

    let prompt = format!(
        "add description fields to this json schema. \
         use the field hints to inform descriptions.\n\n\
         hints:\n{}\n\n\
         schema:\n{}",
        hint_context, schema_text
    );

    // clone model arc for move into spawn_blocking
    let model_clone = Arc::clone(model);

    // run sync llm generation in blocking task
    let output = tokio::task::spawn_blocking(move || {
        let messages = vec![
            Message::system(ENHANCEMENT_SYSTEM_PROMPT),
            Message::user(prompt),
        ];

        model_clone.generate(messages)
    })
    .await
    .map_err(|e| BbtError::Model(format!("task join error: {}", e)))??;

    // parse and validate enhanced schema
    let enhanced: Value = serde_json::from_str(&output).map_err(|e| {
        BbtError::Schema(format!(
            "llm returned invalid json for schema enhancement: {}",
            e
        ))
    })?;

    tracing::info!("json schema enhancement complete");
    Ok(enhanced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhancement_prompt_format() {
        // verify prompt formatting works
        let hints = FieldHints {
            timestamps: vec!["created_at".to_string()],
            folders: vec!["path".to_string()],
        };

        let hint_text = format!(
            "timestamp fields: {}\nfolder fields: {}",
            hints.timestamps.join(", "),
            hints.folders.join(", ")
        );

        assert!(hint_text.contains("created_at"));
        assert!(hint_text.contains("path"));
    }
}
