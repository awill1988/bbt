use crate::error::{BbtError, Result};
use regex::Regex;
use once_cell::sync::Lazy;

static CREATE_TABLE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)create\s+table\b").unwrap()
});

static FENCE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```(?:sql)?\n(.*?)```").unwrap()
});

pub fn extract_create_table(sql_text: &str) -> Result<String> {
    let text = sql_text.trim();

    if text.is_empty() {
        return Err(BbtError::DdlExtraction(
            "model returned empty output".to_string()
        ));
    }

    // strip markdown fences if present
    let text = if let Some(captures) = FENCE_REGEX.captures(text) {
        captures.get(1).map(|m| m.as_str()).unwrap_or(text)
    } else {
        text
    };

    // find CREATE TABLE statement
    let mat = CREATE_TABLE_REGEX.find(text).ok_or_else(|| {
        BbtError::DdlExtraction(
            "model output did not contain create table".to_string()
        )
    })?;

    // extract from CREATE TABLE onwards
    let ddl = &text[mat.start()..];

    // take only the first statement (up to double newline)
    let ddl = ddl.split("\n\n")
        .next()
        .unwrap_or(ddl)
        .trim();

    Ok(ddl.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_create_table_plain() {
        let input = "CREATE TABLE bookmarks (id INTEGER PRIMARY KEY);";
        let result = extract_create_table(input).unwrap();
        assert!(result.starts_with("CREATE TABLE"));
    }

    #[test]
    fn test_extract_create_table_with_fence() {
        let input = "```sql\nCREATE TABLE bookmarks (id INTEGER PRIMARY KEY);\n```";
        let result = extract_create_table(input).unwrap();
        assert!(result.starts_with("CREATE TABLE"));
    }

    #[test]
    fn test_extract_create_table_case_insensitive() {
        let input = "create table bookmarks (id integer primary key);";
        let result = extract_create_table(input).unwrap();
        assert!(result.to_lowercase().contains("create table"));
    }

    #[test]
    fn test_extract_create_table_empty_fails() {
        let input = "";
        let result = extract_create_table(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_create_table_missing_fails() {
        let input = "SELECT * FROM bookmarks;";
        let result = extract_create_table(input);
        assert!(result.is_err());
    }
}
