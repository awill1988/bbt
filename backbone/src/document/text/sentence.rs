use unicode_segmentation::UnicodeSegmentation;

/// Split text into sentences using unicode sentence boundary detection
///
/// Uses UAX #29 (Unicode Text Segmentation) for sentence boundaries
///
/// # Arguments
/// * `text` - The text to split
///
/// # Returns
/// Vector of sentence strings
pub fn split_sentences(text: &str) -> Vec<String> {
    text.unicode_sentences()
        .map(|s| s.to_string())
        .collect()
}

/// Split text into sentences and filter out empty ones
pub fn split_sentences_filtered(text: &str) -> Vec<String> {
    text.unicode_sentences()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Count sentences in text
pub fn count_sentences(text: &str) -> usize {
    text.unicode_sentences().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. How are you? I am fine.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world. ");
        assert_eq!(sentences[1], "How are you? ");
        assert_eq!(sentences[2], "I am fine.");
    }

    #[test]
    fn test_split_sentences_abbreviations() {
        let text = "Dr. Smith went to the store. He bought milk.";
        let sentences = split_sentences(text);
        // unicode segmentation handles abbreviations
        assert!(sentences.len() >= 2);
    }

    #[test]
    fn test_split_sentences_empty() {
        let text = "";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 0);
    }

    #[test]
    fn test_split_sentences_single() {
        let text = "This is a single sentence";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "This is a single sentence");
    }

    #[test]
    fn test_split_sentences_filtered() {
        let text = "  Hello.  \n\n  World!  ";
        let sentences = split_sentences_filtered(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello.");
        assert_eq!(sentences[1], "World!");
    }

    #[test]
    fn test_count_sentences() {
        assert_eq!(count_sentences("Hello. World."), 2);
        assert_eq!(count_sentences("Single sentence"), 1);
        assert_eq!(count_sentences(""), 0);
    }

    #[test]
    fn test_split_sentences_exclamation_question() {
        let text = "What?! Really! Yes.";
        let sentences = split_sentences(text);
        assert!(sentences.len() >= 2);
    }

    #[test]
    fn test_split_sentences_unicode() {
        let text = "Hola. ¿Cómo estás? Bien.";
        let sentences = split_sentences(text);
        assert!(sentences.len() >= 2);
    }
}
