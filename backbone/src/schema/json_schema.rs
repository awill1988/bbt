use crate::error::Result;
use crate::schema::inference::{Schema, SchemaType};
use serde_json::{json, Value};

/// convert internal schema representation to json schema draft 2020-12
#[tracing::instrument(skip(schema), fields(title = ?title))]
pub fn to_json_schema(schema: &Schema, title: Option<&str>) -> Result<Value> {
    tracing::debug!("converting schema to json schema format");

    let mut result = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
    });

    if let Some(t) = title {
        result["title"] = json!(t);
        result["$id"] = json!(format!("https://example.com/{}.schema.json", t));
    }

    // convert schema type
    result["type"] = convert_schema_type(&schema.schema_type);

    // convert object properties
    if let Some(properties) = &schema.properties {
        let mut props = serde_json::Map::new();
        for (key, value) in properties {
            props.insert(key.clone(), to_json_schema(value, None)?);
        }
        result["properties"] = json!(props);
    }

    // convert required fields
    if let Some(required) = &schema.required {
        if !required.is_empty() {
            result["required"] = json!(required);
        }
    }

    // convert array items
    if let Some(items) = &schema.items {
        result["items"] = to_json_schema(items, None)?;
    }

    // add examples if present
    if let Some(examples) = &schema.examples {
        if !examples.is_empty() {
            result["examples"] = json!(examples);
        }
    }

    Ok(result)
}

/// convert internal schema type to json schema type representation
fn convert_schema_type(schema_type: &SchemaType) -> Value {
    match schema_type {
        SchemaType::Single(s) => json!(s),
        SchemaType::Multiple(types) => json!(types),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_simple_string_schema() {
        let schema = Schema {
            schema_type: SchemaType::Single("string".to_string()),
            properties: None,
            required: None,
            items: None,
            examples: Some(vec![json!("hello")]),
        };

        let result = to_json_schema(&schema, None).unwrap();

        assert_eq!(result["$schema"], "https://json-schema.org/draft/2020-12/schema");
        assert_eq!(result["type"], "string");
        assert_eq!(result["examples"], json!(["hello"]));
    }

    #[test]
    fn test_object_schema_with_properties() {
        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            Schema {
                schema_type: SchemaType::Single("string".to_string()),
                properties: None,
                required: None,
                items: None,
                examples: None,
            },
        );
        properties.insert(
            "age".to_string(),
            Schema {
                schema_type: SchemaType::Single("number".to_string()),
                properties: None,
                required: None,
                items: None,
                examples: None,
            },
        );

        let schema = Schema {
            schema_type: SchemaType::Single("object".to_string()),
            properties: Some(properties),
            required: Some(vec!["name".to_string()]),
            items: None,
            examples: None,
        };

        let result = to_json_schema(&schema, Some("person")).unwrap();

        assert_eq!(result["title"], "person");
        assert_eq!(result["type"], "object");
        assert_eq!(result["properties"]["name"]["type"], "string");
        assert_eq!(result["properties"]["age"]["type"], "number");
        assert_eq!(result["required"], json!(["name"]));
    }

    #[test]
    fn test_array_schema() {
        let schema = Schema {
            schema_type: SchemaType::Single("array".to_string()),
            properties: None,
            required: None,
            items: Some(Box::new(Schema {
                schema_type: SchemaType::Single("number".to_string()),
                properties: None,
                required: None,
                items: None,
                examples: None,
            })),
            examples: None,
        };

        let result = to_json_schema(&schema, None).unwrap();

        assert_eq!(result["type"], "array");
        assert_eq!(result["items"]["type"], "number");
    }

    #[test]
    fn test_multiple_types() {
        let schema = Schema {
            schema_type: SchemaType::Multiple(vec!["string".to_string(), "null".to_string()]),
            properties: None,
            required: None,
            items: None,
            examples: None,
        };

        let result = to_json_schema(&schema, None).unwrap();

        assert_eq!(result["type"], json!(["string", "null"]));
    }
}
