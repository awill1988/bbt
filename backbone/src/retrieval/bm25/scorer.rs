use crate::retrieval::bm25::index::Bm25Index;

/// bm25 scoring parameters
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// search result with score
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub doc_id: String,
    pub score: f32,
    pub text: String,
}

/// bm25 scorer
pub struct Bm25Scorer {
    params: Bm25Params,
}

impl Bm25Scorer {
    /// create a new bm25 scorer with default parameters
    pub fn new() -> Self {
        Self {
            params: Bm25Params::default(),
        }
    }

    /// create a new bm25 scorer with custom parameters
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self {
            params: Bm25Params { k1, b },
        }
    }

    /// search the index and return scored results
    ///
    /// # arguments
    /// * `index` - bm25 index
    /// * `query` - query text
    /// * `top_k` - number of results to return
    ///
    /// # returns
    /// scored results sorted by score (descending)
    pub fn search(&self, index: &Bm25Index, query: &str, top_k: usize) -> Vec<ScoredResult> {
        // tokenize query
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // score all documents
        let mut scored_docs: Vec<ScoredResult> = index
            .doc_ids()
            .into_iter()
            .map(|doc_id| {
                let score = self.score_document(index, &query_terms, &doc_id);
                let text = index
                    .get_document(&doc_id)
                    .map(|d| d.chunk_text.clone())
                    .unwrap_or_default();

                ScoredResult {
                    doc_id,
                    score,
                    text,
                }
            })
            .filter(|r| r.score > 0.0) // filter non-matching documents
            .collect();

        // sort by score (descending)
        scored_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // return top k
        scored_docs.truncate(top_k);
        scored_docs
    }

    /// score a single document against query terms
    fn score_document(&self, index: &Bm25Index, query_terms: &[String], doc_id: &str) -> f32 {
        let num_docs = index.num_docs() as f32;
        let avg_doc_length = index.avg_doc_length();
        let doc_length = index.doc_length(doc_id) as f32;

        let mut score = 0.0;

        for term in query_terms {
            let df = index.term_doc_freq(term) as f32; // document frequency
            let tf = index.term_freq(term, doc_id) as f32; // term frequency in doc

            if tf == 0.0 {
                continue;
            }

            // calculate idf
            let idf = ((num_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

            // calculate bm25 score component for this term
            let numerator = tf * (self.params.k1 + 1.0);
            let denominator = tf
                + self.params.k1
                    * (1.0 - self.params.b + self.params.b * (doc_length / avg_doc_length));

            score += idf * (numerator / denominator);
        }

        score
    }
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self::new()
    }
}

/// tokenize text (same as index tokenization)
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| s.len() > 1)
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_scoring() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "machine learning optimization").unwrap();
        index.add_document("doc2", "deep learning neural networks").unwrap();
        index.add_document("doc3", "machine learning algorithms").unwrap();

        let scorer = Bm25Scorer::new();
        let results = scorer.search(&index, "machine learning", 10);

        // should return 2 documents containing "machine learning"
        assert_eq!(results.len(), 2);

        // both should have same terms, scores should be > 0
        assert!(results[0].score > 0.0);
        assert!(results[1].score > 0.0);
    }

    #[test]
    fn test_no_matches() {
        let mut index = Bm25Index::new();
        index.add_document("doc1", "hello world").unwrap();

        let scorer = Bm25Scorer::new();
        let results = scorer.search(&index, "nonexistent query", 10);

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_top_k_limit() {
        let mut index = Bm25Index::new();
        for i in 0..10 {
            index.add_document(&format!("doc{}", i), "test document").unwrap();
        }

        let scorer = Bm25Scorer::new();
        let results = scorer.search(&index, "test", 5);

        assert_eq!(results.len(), 5);
    }
}
