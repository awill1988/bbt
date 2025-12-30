use crate::error::Result;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// document statistics for bm25 scoring
#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
pub struct DocumentStats {
    pub doc_id: String,
    pub doc_length: usize,
}

/// in-memory bm25 inverted index
#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
pub struct Bm25Index {
    /// inverted index: term -> [(doc_id, term_freq)]
    term_index: HashMap<String, Vec<(String, u32)>>,
    /// document stats: doc_id -> stats
    doc_stats: HashMap<String, DocumentStats>,
    /// total number of documents
    num_docs: usize,
    /// average document length
    avg_doc_length: f32,
}

impl Bm25Index {
    /// create a new empty bm25 index
    pub fn new() -> Self {
        Self {
            term_index: HashMap::new(),
            doc_stats: HashMap::new(),
            num_docs: 0,
            avg_doc_length: 0.0,
        }
    }

    /// add a document to the index
    pub fn add_document(&mut self, doc_id: &str, text: &str) -> Result<()> {
        // tokenize text
        let tokens = tokenize(text);
        let doc_length = tokens.len();

        // count term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        // update inverted index
        for (term, freq) in term_freqs {
            self.term_index
                .entry(term)
                .or_insert_with(Vec::new)
                .push((doc_id.to_string(), freq));
        }

        // store document stats
        self.doc_stats.insert(
            doc_id.to_string(),
            DocumentStats {
                doc_id: doc_id.to_string(),
                doc_length,
            },
        );

        // update global stats
        self.num_docs += 1;
        self.recalculate_avg_length();

        Ok(())
    }

    /// remove a document from the index
    pub fn remove_document(&mut self, doc_id: &str) -> Result<()> {
        // remove from doc stats
        if self.doc_stats.remove(doc_id).is_none() {
            return Ok(()); // document not in index
        }

        // remove from inverted index
        for (_term, postings) in self.term_index.iter_mut() {
            postings.retain(|(id, _freq)| id != doc_id);
        }

        // remove empty term entries
        self.term_index.retain(|_term, postings| !postings.is_empty());

        // update global stats
        self.num_docs = self.num_docs.saturating_sub(1);
        self.recalculate_avg_length();

        Ok(())
    }

    /// get document stats
    pub fn get_document(&self, doc_id: &str) -> Option<&DocumentStats> {
        self.doc_stats.get(doc_id)
    }

    /// get number of documents in index
    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// get average document length
    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }

    /// get term document frequency
    pub fn term_doc_freq(&self, term: &str) -> usize {
        self.term_index.get(term).map(|p| p.len()).unwrap_or(0)
    }

    /// get term frequency in document
    pub fn term_freq(&self, term: &str, doc_id: &str) -> u32 {
        self.term_index
            .get(term)
            .and_then(|postings| {
                postings
                    .iter()
                    .find(|(id, _)| id == doc_id)
                    .map(|(_, freq)| *freq)
            })
            .unwrap_or(0)
    }

    /// get document length
    pub fn doc_length(&self, doc_id: &str) -> usize {
        self.doc_stats
            .get(doc_id)
            .map(|stats| stats.doc_length)
            .unwrap_or(0)
    }

    /// recalculate average document length
    fn recalculate_avg_length(&mut self) {
        if self.num_docs == 0 {
            self.avg_doc_length = 0.0;
            return;
        }

        let total_length: usize = self.doc_stats.values().map(|s| s.doc_length).sum();
        self.avg_doc_length = total_length as f32 / self.num_docs as f32;
    }

    /// get all document ids
    pub fn doc_ids(&self) -> Vec<String> {
        self.doc_stats.keys().cloned().collect()
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

/// tokenize text into terms
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| s.len() > 1) // filter single characters
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(!tokens.contains(&"a".to_string())); // single char filtered
    }

    #[test]
    fn test_add_document() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "hello world").unwrap();

        assert_eq!(index.num_docs(), 1);
        assert_eq!(index.term_doc_freq("hello"), 1);
        assert_eq!(index.term_freq("hello", "doc1"), 1);
    }

    #[test]
    fn test_remove_document() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "hello world").unwrap();
        index.add_document("doc2", "hello rust").unwrap();

        assert_eq!(index.num_docs(), 2);

        index.remove_document("doc1").unwrap();
        assert_eq!(index.num_docs(), 1);
        assert_eq!(index.term_doc_freq("world"), 0);
        assert_eq!(index.term_doc_freq("hello"), 1);
    }

    #[test]
    fn test_avg_doc_length() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "hello world").unwrap(); // 2 tokens
        index.add_document("doc2", "hello rust programming").unwrap(); // 3 tokens

        assert!((index.avg_doc_length() - 2.5).abs() < 0.01);
    }
}
