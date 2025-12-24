use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    #[serde(rename = "type")]
    pub schema_type: SchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Schema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Schema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SchemaType {
    Single(String),
    Multiple(Vec<String>),
}

fn infer_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn merge_types(lhs: SchemaType, rhs: SchemaType) -> SchemaType {
    let mut merged = HashSet::new();

    match lhs {
        SchemaType::Single(s) => { merged.insert(s); }
        SchemaType::Multiple(v) => { merged.extend(v); }
    }

    match rhs {
        SchemaType::Single(s) => { merged.insert(s); }
        SchemaType::Multiple(v) => { merged.extend(v); }
    }

    let mut types: Vec<String> = merged.into_iter().collect();
    types.sort();

    if types.len() == 1 {
        SchemaType::Single(types.into_iter().next().unwrap())
    } else {
        SchemaType::Multiple(types)
    }
}

fn merge_schema(lhs: Schema, rhs: Schema) -> Schema {
    let mut result = lhs.clone();

    result.schema_type = merge_types(lhs.schema_type, rhs.schema_type);

    // merge object properties
    if matches!(&result.schema_type, SchemaType::Single(s) if s == "object") {
        let mut props = lhs.properties.unwrap_or_default();
        for (key, rhs_schema) in rhs.properties.unwrap_or_default() {
            if let Some(lhs_schema) = props.remove(&key) {
                props.insert(key, merge_schema(lhs_schema, rhs_schema));
            } else {
                props.insert(key, rhs_schema);
            }
        }

        let mut required = HashSet::new();
        if let Some(r) = lhs.required {
            required.extend(r);
        }
        if let Some(r) = rhs.required {
            required.extend(r);
        }

        result.properties = Some(props);
        if !required.is_empty() {
            let mut req_vec: Vec<String> = required.into_iter().collect();
            req_vec.sort();
            result.required = Some(req_vec);
        }
    }

    // merge array items
    if matches!(&result.schema_type, SchemaType::Single(s) if s == "array") {
        result.items = match (lhs.items, rhs.items) {
            (Some(l), Some(r)) => Some(Box::new(merge_schema(*l, *r))),
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        };
    }

    // merge examples
    let mut examples = Vec::new();
    if let Some(e) = lhs.examples {
        examples.extend(e);
    }
    if let Some(e) = rhs.examples {
        examples.extend(e);
    }
    if !examples.is_empty() {
        examples.truncate(3);
        result.examples = Some(examples);
    }

    result
}

pub fn infer_json_schema(payload: &Value) -> Schema {
    fn visit(node: &Value) -> Schema {
        let node_type = infer_type(node);

        match node {
            Value::Object(obj) => {
                let mut properties = HashMap::new();
                let mut required_keys = HashSet::new();

                for (key, value) in obj {
                    let value_schema = visit(value);
                    if let Some(existing) = properties.remove(key) {
                        properties.insert(key.clone(), merge_schema(existing, value_schema));
                    } else {
                        properties.insert(key.clone(), value_schema);
                    }

                    if !value.is_null() {
                        required_keys.insert(key.clone());
                    }
                }

                let mut schema = Schema {
                    schema_type: SchemaType::Single(node_type.to_string()),
                    properties: Some(properties),
                    required: None,
                    items: None,
                    examples: None,
                };

                if !required_keys.is_empty() {
                    let mut req: Vec<String> = required_keys.into_iter().collect();
                    req.sort();
                    schema.required = Some(req);
                }

                schema
            }
            Value::Array(arr) => {
                let mut items_schema: Option<Schema> = None;

                for value in arr {
                    let value_schema = visit(value);
                    items_schema = Some(match items_schema {
                        None => value_schema,
                        Some(existing) => merge_schema(existing, value_schema),
                    });
                }

                Schema {
                    schema_type: SchemaType::Single(node_type.to_string()),
                    properties: None,
                    required: None,
                    items: items_schema.map(Box::new),
                    examples: None,
                }
            }
            _ => {
                let mut schema = Schema {
                    schema_type: SchemaType::Single(node_type.to_string()),
                    properties: None,
                    required: None,
                    items: None,
                    examples: None,
                };

                if !node.is_null() {
                    schema.examples = Some(vec![node.clone()]);
                }

                schema
            }
        }
    }

    visit(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_infer_simple_object() {
        let payload = json!({
            "name": "test",
            "age": 42
        });

        let schema = infer_json_schema(&payload);
        assert!(matches!(schema.schema_type, SchemaType::Single(ref s) if s == "object"));
        assert!(schema.properties.is_some());
    }

    #[test]
    fn test_infer_array() {
        let payload = json!([
            {"name": "alice"},
            {"name": "bob", "age": 30}
        ]);

        let schema = infer_json_schema(&payload);
        assert!(matches!(schema.schema_type, SchemaType::Single(ref s) if s == "array"));
        assert!(schema.items.is_some());
    }

    #[test]
    fn test_merge_types() {
        let t1 = SchemaType::Single("string".to_string());
        let t2 = SchemaType::Single("number".to_string());
        let merged = merge_types(t1, t2);

        match merged {
            SchemaType::Multiple(types) => {
                assert_eq!(types, vec!["number", "string"]);
            }
            _ => panic!("expected multiple types"),
        }
    }
}
