use crate::schema::{FieldHints, Schema};

pub const SCHEMA_SYSTEM_PROMPT: &str =
    "you generate deterministic sqlite ddl for bookmarks. \
     output only a single create table statement. \
     no markdown, no commentary, no backticks, no multiple tables.";

pub fn build_schema_task_prompt(schema: &Schema, hints: &FieldHints) -> String {
    let schema_text = serde_json::to_string_pretty(schema)
        .unwrap_or_else(|_| "{}".to_string());

    let mut hint_lines = Vec::new();

    if !hints.timestamps.is_empty() {
        hint_lines.push(format!(
            "timestamp candidates: {}",
            hints.timestamps.join(", ")
        ));
    }

    if !hints.folders.is_empty() {
        hint_lines.push(format!(
            "folder/tag candidates: {}",
            hints.folders.join(", ")
        ));
    }

    let hints_text = if hint_lines.is_empty() {
        "no explicit timestamp or folder hints found".to_string()
    } else {
        hint_lines.join("\n")
    };

    format!(
        "generate a single sqlite create table statement for bookmarks.\n\
         - include url and title columns.\n\
         - include timestamp columns if present in the export.\n\
         - include folder/path metadata if present.\n\
         - include an embeddings vector column.\n\
         - keep types and constraints reasonable and deterministic.\n\
         - output only the create table statement.\n\n\
         json schema:\n{}\n\nhints:\n{}",
        schema_text, hints_text
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::inference::infer_json_schema;
    use crate::schema::hints::derive_field_hints;
    use serde_json::json;

    #[test]
    fn test_build_schema_task_prompt() {
        let payload = json!({
            "url": "https://example.com",
            "title": "Example",
            "created_at": "2023-01-01"
        });

        let schema = infer_json_schema(&payload);
        let hints = derive_field_hints(&schema);
        let prompt = build_schema_task_prompt(&schema, &hints);

        assert!(prompt.contains("json schema:"));
        assert!(prompt.contains("hints:"));
        assert!(prompt.contains("created_at"));
    }
}
