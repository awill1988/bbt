use super::inference::{Schema, SchemaType};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHints {
    pub timestamps: Vec<String>,
    pub folders: Vec<String>,
}

pub fn derive_field_hints(schema: &Schema) -> FieldHints {
    let mut timestamp_keys = HashSet::new();
    let mut folder_keys = HashSet::new();

    fn walk(
        schema: &Schema,
        path: &[String],
        timestamp_keys: &mut HashSet<String>,
        folder_keys: &mut HashSet<String>,
    ) {
        match &schema.schema_type {
            SchemaType::Single(t) if t == "object" => {
                if let Some(properties) = &schema.properties {
                    for (key, value) in properties {
                        let lowered = key.to_lowercase();

                        // check for timestamp fields
                        if lowered.contains("date")
                            || lowered.contains("time")
                            || lowered.contains("updated")
                            || lowered.contains("created")
                        {
                            let mut full_path = path.to_vec();
                            full_path.push(key.clone());
                            timestamp_keys.insert(full_path.join("."));
                        }

                        // check for folder fields
                        if lowered.contains("folder")
                            || lowered.contains("path")
                            || lowered.contains("group")
                            || lowered.contains("category")
                            || lowered.contains("tag")
                        {
                            let mut full_path = path.to_vec();
                            full_path.push(key.clone());
                            folder_keys.insert(full_path.join("."));
                        }

                        let mut new_path = path.to_vec();
                        new_path.push(key.clone());
                        walk(value, &new_path, timestamp_keys, folder_keys);
                    }
                }
            }
            SchemaType::Single(t) if t == "array" => {
                if let Some(items) = &schema.items {
                    let mut new_path = path.to_vec();
                    new_path.push("[]".to_string());
                    walk(items, &new_path, timestamp_keys, folder_keys);
                }
            }
            _ => {}
        }
    }

    walk(&schema, &[], &mut timestamp_keys, &mut folder_keys);

    let mut timestamps: Vec<String> = timestamp_keys.into_iter().collect();
    timestamps.sort();

    let mut folders: Vec<String> = folder_keys.into_iter().collect();
    folders.sort();

    FieldHints {
        timestamps,
        folders,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::inference::infer_json_schema;
    use serde_json::json;

    #[test]
    fn test_derive_timestamp_hints() {
        let payload = json!({
            "created_at": "2023-01-01",
            "updated_at": "2023-01-02",
            "name": "test"
        });

        let schema = infer_json_schema(&payload);
        let hints = derive_field_hints(&schema);

        assert!(hints.timestamps.contains(&"created_at".to_string()));
        assert!(hints.timestamps.contains(&"updated_at".to_string()));
        assert_eq!(hints.timestamps.len(), 2);
    }

    #[test]
    fn test_derive_folder_hints() {
        let payload = json!({
            "folder": "bookmarks",
            "path": "/home/user",
            "category": "tech",
            "title": "test"
        });

        let schema = infer_json_schema(&payload);
        let hints = derive_field_hints(&schema);

        assert!(hints.folders.contains(&"folder".to_string()));
        assert!(hints.folders.contains(&"path".to_string()));
        assert!(hints.folders.contains(&"category".to_string()));
        assert_eq!(hints.folders.len(), 3);
    }
}
