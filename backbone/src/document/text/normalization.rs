use unicode_normalization::UnicodeNormalization;

/// Normalize text by:
/// - Applying unicode NFC normalization
/// - Collapsing multiple whitespace characters into single spaces
/// - Trimming leading/trailing whitespace
/// - Removing zero-width characters
///
/// # Arguments
/// * `text` - The text to normalize
///
/// # Returns
/// Normalized text string
pub fn normalize_text(text: &str) -> String {
    // apply unicode NFC normalization (canonical decomposition followed by composition)
    let normalized: String = text.nfc().collect();

    // remove zero-width characters
    let without_zero_width: String = normalized
        .chars()
        .filter(|c| {
            !matches!(
                *c,
                '\u{200B}' | // zero-width space
                '\u{200C}' | // zero-width non-joiner
                '\u{200D}' | // zero-width joiner
                '\u{FEFF}'   // zero-width no-break space (BOM)
            )
        })
        .collect();

    // collapse multiple whitespace into single spaces
    let mut result = String::new();
    let mut prev_was_whitespace = false;

    for ch in without_zero_width.chars() {
        if ch.is_whitespace() {
            if !prev_was_whitespace {
                result.push(' ');
            }
            prev_was_whitespace = true;
        } else {
            result.push(ch);
            prev_was_whitespace = false;
        }
    }

    // trim leading and trailing whitespace
    result.trim().to_string()
}

/// Remove extra whitespace from text
/// - Collapse multiple spaces into single spaces
/// - Remove leading/trailing whitespace
pub fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove non-printable characters from text
pub fn remove_non_printable(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || c.is_whitespace())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text_basic() {
        let input = "  hello   world  ";
        let output = normalize_text(input);
        assert_eq!(output, "hello world");
    }

    #[test]
    fn test_normalize_text_multiple_spaces() {
        let input = "hello     world";
        let output = normalize_text(input);
        assert_eq!(output, "hello world");
    }

    #[test]
    fn test_normalize_text_tabs_and_newlines() {
        let input = "hello\t\nworld\n\n\tthere";
        let output = normalize_text(input);
        assert_eq!(output, "hello world there");
    }

    #[test]
    fn test_normalize_text_unicode_nfc() {
        // é can be represented as:
        // - single character: U+00E9 (composed)
        // - two characters: e + ́ (U+0065 + U+0301, decomposed)
        let composed = "\u{00E9}"; // é (single char)
        let decomposed = "\u{0065}\u{0301}"; // e + combining acute accent

        let normalized_composed = normalize_text(composed);
        let normalized_decomposed = normalize_text(decomposed);

        // both should normalize to the same NFC form
        assert_eq!(normalized_composed, normalized_decomposed);
        assert_eq!(normalized_composed, "é");
    }

    #[test]
    fn test_normalize_text_zero_width() {
        let input = "hello\u{200B}world\u{FEFF}test";
        let output = normalize_text(input);
        assert_eq!(output, "helloworldtest");
    }

    #[test]
    fn test_normalize_text_empty() {
        assert_eq!(normalize_text(""), "");
        assert_eq!(normalize_text("   "), "");
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(collapse_whitespace("  hello   world  "), "hello world");
        assert_eq!(collapse_whitespace("hello"), "hello");
        assert_eq!(collapse_whitespace(""), "");
    }

    #[test]
    fn test_remove_non_printable() {
        // control characters (except whitespace)
        let input = "hello\x00\x01world\x1F";
        let output = remove_non_printable(input);
        assert_eq!(output, "helloworld");

        // keep whitespace control characters
        let input2 = "hello\nworld\tthere";
        let output2 = remove_non_printable(input2);
        assert_eq!(output2, "hello\nworld\tthere");
    }
}
