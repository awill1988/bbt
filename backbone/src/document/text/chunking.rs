use crate::error::Result;

/// Chunking strategy for text
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Sliding window with overlap
    SlidingWindow,
    /// Section-aware chunking (respects document structure)
    SectionAware,
}

/// Chunk text using sliding window with overlap
///
/// # Arguments
/// * `text` - The text to chunk
/// * `chunk_size` - Maximum tokens per chunk
/// * `overlap` - Number of tokens to overlap between chunks
///
/// # Returns
/// Vector of text chunks
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Result<Vec<String>> {
    if chunk_size == 0 {
        return Err(crate::error::BbtError::Schema(
            "chunk_size must be greater than 0".to_string(),
        ));
    }

    if overlap >= chunk_size {
        return Err(crate::error::BbtError::Schema(
            "overlap must be less than chunk_size".to_string(),
        ));
    }

    // split text into words (simple whitespace-based tokenization)
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return Ok(vec![]);
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);

        // move start forward by (chunk_size - overlap)
        let step = chunk_size.saturating_sub(overlap);
        if step == 0 {
            break; // prevent infinite loop
        }
        start += step;

        // break if we've consumed all words
        if end == words.len() {
            break;
        }
    }

    Ok(chunks)
}

/// Count tokens in text (simple whitespace-based)
pub fn count_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Chunk text with a target token count using a tokenizer-aware approach
///
/// This is a placeholder for future integration with the tokenizers crate
/// for more accurate token counting
pub fn chunk_text_with_tokenizer(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<String>> {
    // for now, use simple word-based chunking
    // todo: integrate tokenizers crate for accurate token counting
    chunk_text(text, chunk_size, overlap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_simple() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, 3, 1).unwrap();

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], "one two three");
        assert_eq!(chunks[1], "three four five");
        assert_eq!(chunks[2], "five six seven");
        assert_eq!(chunks[3], "seven eight nine ten");
    }

    #[test]
    fn test_chunk_text_no_overlap() {
        let text = "one two three four five six";
        let chunks = chunk_text(text, 2, 0).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "one two");
        assert_eq!(chunks[1], "three four");
        assert_eq!(chunks[2], "five six");
    }

    #[test]
    fn test_chunk_text_empty() {
        let text = "";
        let chunks = chunk_text(text, 10, 2).unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_chunk_text_single_chunk() {
        let text = "hello world";
        let chunks = chunk_text(text, 100, 10).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_text_invalid_params() {
        let text = "hello world";

        // chunk_size = 0
        assert!(chunk_text(text, 0, 0).is_err());

        // overlap >= chunk_size
        assert!(chunk_text(text, 10, 10).is_err());
        assert!(chunk_text(text, 10, 11).is_err());
    }

    #[test]
    fn test_count_tokens() {
        assert_eq!(count_tokens("hello world"), 2);
        assert_eq!(count_tokens("one two three"), 3);
        assert_eq!(count_tokens(""), 0);
        assert_eq!(count_tokens("   spaced   out   "), 2);
    }

    #[test]
    fn test_chunk_text_large_overlap() {
        let text = "one two three four five";
        let chunks = chunk_text(text, 3, 2).unwrap();

        // with chunk_size=3 and overlap=2, step=1
        assert_eq!(chunks[0], "one two three");
        assert_eq!(chunks[1], "two three four");
        assert_eq!(chunks[2], "three four five");
    }
}
