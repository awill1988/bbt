/// Tokenize text into alphanumeric tokens (for BM25)
///
/// This is a simple alphanumeric tokenizer that:
/// - Splits on non-alphanumeric characters
/// - Converts to lowercase
/// - Filters out empty tokens
///
/// # Arguments
/// * `text` - The text to tokenize
///
/// # Returns
/// Vector of lowercase alphanumeric tokens
pub fn tokenize_text(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Count tokens in text
pub fn count_tokens_simple(text: &str) -> usize {
    tokenize_text(text).len()
}

/// Tokenize text preserving case
pub fn tokenize_text_preserve_case(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_text_basic() {
        let text = "Hello World";
        let tokens = tokenize_text(text);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_text_punctuation() {
        let text = "Hello, world! How are you?";
        let tokens = tokenize_text(text);
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn test_tokenize_text_numbers() {
        let text = "Testing 123 with numbers456";
        let tokens = tokenize_text(text);
        assert_eq!(tokens, vec!["testing", "123", "with", "numbers456"]);
    }

    #[test]
    fn test_tokenize_text_mixed() {
        let text = "user@example.com, hello-world_test";
        let tokens = tokenize_text(text);
        assert_eq!(
            tokens,
            vec!["user", "example", "com", "hello", "world", "test"]
        );
    }

    #[test]
    fn test_tokenize_text_empty() {
        let text = "";
        let tokens = tokenize_text(text);
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_tokenize_text_only_punctuation() {
        let text = "!@#$%^&*()";
        let tokens = tokenize_text(text);
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_count_tokens_simple() {
        assert_eq!(count_tokens_simple("hello world"), 2);
        assert_eq!(count_tokens_simple("one, two, three!"), 3);
        assert_eq!(count_tokens_simple(""), 0);
    }

    #[test]
    fn test_tokenize_preserve_case() {
        let text = "Hello World";
        let tokens = tokenize_text_preserve_case(text);
        assert_eq!(tokens, vec!["Hello", "World"]);
    }

    #[test]
    fn test_tokenize_unicode() {
        let text = "Café naïve résumé";
        let tokens = tokenize_text(text);
        assert_eq!(tokens, vec!["café", "naïve", "résumé"]);
    }
}
