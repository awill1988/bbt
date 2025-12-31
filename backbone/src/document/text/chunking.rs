use crate::error::{BbtError, Result};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Chunking strategy for text
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Sliding window with overlap
    SlidingWindow,
    /// Section-aware chunking (respects document structure)
    SectionAware,
}

/// Chunk text using sliding window with overlap (word-based)
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
        return Err(BbtError::Schema(
            "chunk_size must be greater than 0".to_string(),
        ));
    }

    if overlap >= chunk_size {
        return Err(BbtError::Schema(
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

/// Count tokens using a tokenizer
pub fn count_tokens_with_tokenizer(text: &str, tokenizer: &Tokenizer) -> Result<usize> {
    let encoding = tokenizer.encode(text, false)
        .map_err(|e| BbtError::Schema(format!("tokenization failed: {}", e)))?;
    Ok(encoding.len())
}

/// Tokenizer-aware text chunker
///
/// Uses a huggingface tokenizer to accurately count tokens and chunk text
/// while respecting token boundaries and overlap requirements.
pub struct TokenizerChunker {
    tokenizer: Arc<Tokenizer>,
    chunk_size: usize,
    overlap: usize,
}

impl TokenizerChunker {
    /// Create a new tokenizer-aware chunker
    ///
    /// # Arguments
    /// * `tokenizer` - huggingface tokenizer instance
    /// * `chunk_size` - maximum tokens per chunk
    /// * `overlap` - number of tokens to overlap between chunks
    pub fn new(tokenizer: Arc<Tokenizer>, chunk_size: usize, overlap: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(BbtError::Schema(
                "chunk_size must be greater than 0".to_string(),
            ));
        }

        if overlap >= chunk_size {
            return Err(BbtError::Schema(
                "overlap must be less than chunk_size".to_string(),
            ));
        }

        Ok(Self {
            tokenizer,
            chunk_size,
            overlap,
        })
    }

    /// Create from a tokenizer file path
    pub fn from_file(path: &str, chunk_size: usize, overlap: usize) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| BbtError::Model(format!("failed to load tokenizer: {}", e)))?;
        Self::new(Arc::new(tokenizer), chunk_size, overlap)
    }

    /// Count tokens in text
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        count_tokens_with_tokenizer(text, &self.tokenizer)
    }

    /// Chunk text using the tokenizer for accurate token counting
    ///
    /// This method splits text into chunks that each contain at most
    /// `chunk_size` tokens, with `overlap` tokens shared between adjacent chunks.
    pub fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // encode the full text to get all tokens
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| BbtError::Schema(format!("tokenization failed: {}", e)))?;

        let tokens = encoding.get_ids();
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        let offsets = encoding.get_offsets();

        // build chunks using token indices
        let mut chunks = Vec::new();
        let mut start_token = 0;
        let step = self.chunk_size.saturating_sub(self.overlap);

        while start_token < tokens.len() {
            let end_token = (start_token + self.chunk_size).min(tokens.len());

            // get character offsets for this chunk
            let start_char = offsets[start_token].0;
            let end_char = offsets[end_token - 1].1;

            // extract the text for this chunk
            let chunk_text = &text[start_char..end_char];
            chunks.push(chunk_text.to_string());

            if step == 0 || end_token == tokens.len() {
                break;
            }
            start_token += step;
        }

        Ok(chunks)
    }

    /// Chunk text with sentence awareness
    ///
    /// Attempts to split at sentence boundaries when possible,
    /// while still respecting the token limit.
    pub fn chunk_sentences(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // split into sentences (simple heuristic)
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.is_empty() {
            // no sentence boundaries found, fall back to regular chunking
            return self.chunk(text);
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0;

        for sentence in sentences {
            let sentence_text = sentence.trim();
            if sentence_text.is_empty() {
                continue;
            }

            // count tokens in this sentence
            let sentence_tokens = self.count_tokens(sentence_text)?;

            // if adding this sentence would exceed the limit, start a new chunk
            if current_tokens + sentence_tokens > self.chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());

                // include overlap tokens from the end of the previous chunk
                if self.overlap > 0 {
                    // get the last few words for overlap
                    let words: Vec<&str> = chunks.last().unwrap().split_whitespace().collect();
                    let overlap_start = words.len().saturating_sub(self.overlap / 2);
                    current_chunk = words[overlap_start..].join(" ") + " ";
                    current_tokens = self.count_tokens(&current_chunk)?;
                } else {
                    current_chunk = String::new();
                    current_tokens = 0;
                }
            }

            // if the sentence itself is too long, use regular chunking for it
            if sentence_tokens > self.chunk_size {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk = String::new();
                    current_tokens = 0;
                }

                // chunk the long sentence
                let sub_chunks = self.chunk(sentence_text)?;
                chunks.extend(sub_chunks);
                continue;
            }

            // add sentence to current chunk
            if !current_chunk.is_empty() && !current_chunk.ends_with(' ') {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence_text);
            current_chunk.push('.');
            current_tokens += sentence_tokens + 1; // +1 for the period
        }

        // add final chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        Ok(chunks)
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get overlap size
    pub fn overlap(&self) -> usize {
        self.overlap
    }
}

/// Chunk text with a target token count using a tokenizer-aware approach
///
/// # Arguments
/// * `text` - the text to chunk
/// * `chunk_size` - maximum tokens per chunk
/// * `overlap` - number of tokens to overlap between chunks
/// * `tokenizer` - optional tokenizer for accurate token counting
///
/// If no tokenizer is provided, falls back to simple word-based chunking.
pub fn chunk_text_with_tokenizer(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<String>> {
    // fall back to word-based chunking
    // for full tokenizer support, use TokenizerChunker directly
    chunk_text(text, chunk_size, overlap)
}

/// Chunk text using a specific tokenizer
pub fn chunk_text_with_tokenizer_instance(
    text: &str,
    chunk_size: usize,
    overlap: usize,
    tokenizer: &Tokenizer,
) -> Result<Vec<String>> {
    let chunker = TokenizerChunker::new(Arc::new(tokenizer.clone()), chunk_size, overlap)?;
    chunker.chunk(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_simple() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, 3, 1).unwrap();

        // step = 3 - 1 = 2, so we advance by 2 words each iteration
        // chunks: [0..3], [2..5], [4..7], [6..9], [8..10]
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0], "one two three");
        assert_eq!(chunks[1], "three four five");
        assert_eq!(chunks[2], "five six seven");
        assert_eq!(chunks[3], "seven eight nine");
        assert_eq!(chunks[4], "nine ten");
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
