use crate::error::Result;
use crate::retrieval::bm25::ScoredResult;

/// fusion strategy for combining multiple retrieval results
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// weighted sum of normalized scores
    WeightedSum { vector_weight: f32, bm25_weight: f32 },
    /// reciprocal rank fusion
    ReciprocalRankFusion { k: usize },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::WeightedSum {
            vector_weight: 0.7,
            bm25_weight: 0.3,
        }
    }
}

/// result with combined score from multiple sources
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub doc_id: String,
    pub score: f32,
    pub text: String,
    pub vector_score: Option<f32>,
    pub bm25_score: Option<f32>,
}

/// fuse vector and bm25 results using specified strategy
///
/// # arguments
/// * `vector_results` - results from vector search with (doc_id, score, text)
/// * `bm25_results` - results from bm25 search
/// * `strategy` - fusion strategy to use
/// * `top_k` - number of results to return
pub fn fuse_results(
    vector_results: &[(String, f32, String)],
    bm25_results: &[ScoredResult],
    strategy: FusionStrategy,
    top_k: usize,
) -> Result<Vec<FusedResult>> {
    match strategy {
        FusionStrategy::WeightedSum {
            vector_weight,
            bm25_weight,
        } => weighted_sum_fusion(vector_results, bm25_results, vector_weight, bm25_weight, top_k),
        FusionStrategy::ReciprocalRankFusion { k } => {
            reciprocal_rank_fusion(vector_results, bm25_results, k, top_k)
        }
    }
}

/// weighted sum fusion with min-max normalization
fn weighted_sum_fusion(
    vector_results: &[(String, f32, String)],
    bm25_results: &[ScoredResult],
    vector_weight: f32,
    bm25_weight: f32,
    top_k: usize,
) -> Result<Vec<FusedResult>> {
    // normalize vector scores to [0, 1]
    let vector_scores = normalize_scores(vector_results.iter().map(|(_, s, _)| *s).collect());
    let vector_map: std::collections::HashMap<String, (f32, f32, String)> = vector_results
        .iter()
        .zip(vector_scores.iter())
        .map(|((id, orig_score, text), norm_score)| {
            (id.clone(), (*orig_score, *norm_score, text.clone()))
        })
        .collect();

    // normalize bm25 scores to [0, 1]
    let bm25_scores = normalize_scores(bm25_results.iter().map(|r| r.score).collect());
    let bm25_map: std::collections::HashMap<String, (f32, f32)> = bm25_results
        .iter()
        .zip(bm25_scores.iter())
        .map(|(r, norm_score)| (r.doc_id.clone(), (r.score, *norm_score)))
        .collect();

    // combine all document ids
    let mut all_doc_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    all_doc_ids.extend(vector_map.keys().cloned());
    all_doc_ids.extend(bm25_map.keys().cloned());

    // calculate weighted sum for each document
    let mut fused: Vec<FusedResult> = all_doc_ids
        .into_iter()
        .map(|doc_id| {
            let (vector_orig, vector_norm, text) = vector_map
                .get(&doc_id)
                .cloned()
                .unwrap_or((0.0, 0.0, String::new()));
            let (bm25_orig, bm25_norm) = bm25_map.get(&doc_id).cloned().unwrap_or((0.0, 0.0));

            let combined_score = vector_weight * vector_norm + bm25_weight * bm25_norm;

            FusedResult {
                doc_id,
                score: combined_score,
                text,
                vector_score: if vector_orig > 0.0 {
                    Some(vector_orig)
                } else {
                    None
                },
                bm25_score: if bm25_orig > 0.0 {
                    Some(bm25_orig)
                } else {
                    None
                },
            }
        })
        .collect();

    // sort by combined score (descending)
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // return top k
    fused.truncate(top_k);
    Ok(fused)
}

/// reciprocal rank fusion
fn reciprocal_rank_fusion(
    vector_results: &[(String, f32, String)],
    bm25_results: &[ScoredResult],
    k: usize,
    top_k: usize,
) -> Result<Vec<FusedResult>> {
    // create rank maps
    let vector_ranks: std::collections::HashMap<String, usize> = vector_results
        .iter()
        .enumerate()
        .map(|(rank, (id, _, _))| (id.clone(), rank))
        .collect();

    let bm25_ranks: std::collections::HashMap<String, usize> = bm25_results
        .iter()
        .enumerate()
        .map(|(rank, r)| (r.doc_id.clone(), rank))
        .collect();

    // store original scores for reference
    let vector_score_map: std::collections::HashMap<String, f32> = vector_results
        .iter()
        .map(|(id, score, _)| (id.clone(), *score))
        .collect();

    let bm25_score_map: std::collections::HashMap<String, f32> = bm25_results
        .iter()
        .map(|r| (r.doc_id.clone(), r.score))
        .collect();

    let text_map: std::collections::HashMap<String, String> = vector_results
        .iter()
        .map(|(id, _, text)| (id.clone(), text.clone()))
        .collect();

    // combine all document ids
    let mut all_doc_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    all_doc_ids.extend(vector_ranks.keys().cloned());
    all_doc_ids.extend(bm25_ranks.keys().cloned());

    // calculate rrf score for each document
    let mut fused: Vec<FusedResult> = all_doc_ids
        .into_iter()
        .map(|doc_id| {
            let vector_rank = vector_ranks.get(&doc_id);
            let bm25_rank = bm25_ranks.get(&doc_id);

            let mut rrf_score = 0.0;
            if let Some(&rank) = vector_rank {
                rrf_score += 1.0 / (k as f32 + rank as f32 + 1.0);
            }
            if let Some(&rank) = bm25_rank {
                rrf_score += 1.0 / (k as f32 + rank as f32 + 1.0);
            }

            FusedResult {
                doc_id: doc_id.clone(),
                score: rrf_score,
                text: text_map.get(&doc_id).cloned().unwrap_or_default(),
                vector_score: vector_score_map.get(&doc_id).copied(),
                bm25_score: bm25_score_map.get(&doc_id).copied(),
            }
        })
        .collect();

    // sort by rrf score (descending)
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // return top k
    fused.truncate(top_k);
    Ok(fused)
}

/// normalize scores to [0, 1] using min-max normalization
fn normalize_scores(scores: Vec<f32>) -> Vec<f32> {
    if scores.is_empty() {
        return scores;
    }

    let min_score = scores
        .iter()
        .cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let max_score = scores
        .iter()
        .cloned()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    let range = max_score - min_score;
    if range == 0.0 {
        // all scores same, return all 1.0
        return vec![1.0; scores.len()];
    }

    scores
        .into_iter()
        .map(|s| (s - min_score) / range)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_scores() {
        let scores = vec![0.5, 0.8, 0.2, 1.0];
        let normalized = normalize_scores(scores);

        assert!((normalized[0] - 0.375).abs() < 0.01); // (0.5 - 0.2) / (1.0 - 0.2)
        assert!((normalized[1] - 0.75).abs() < 0.01); // (0.8 - 0.2) / (1.0 - 0.2)
        assert!((normalized[2] - 0.0).abs() < 0.01); // (0.2 - 0.2) / (1.0 - 0.2)
        assert!((normalized[3] - 1.0).abs() < 0.01); // (1.0 - 0.2) / (1.0 - 0.2)
    }

    #[test]
    fn test_weighted_sum_fusion() {
        let vector_results = vec![
            ("doc1".to_string(), 0.9, "text1".to_string()),
            ("doc2".to_string(), 0.7, "text2".to_string()),
        ];

        let bm25_results = vec![
            ScoredResult {
                doc_id: "doc2".to_string(),
                score: 5.0,
                text: "text2".to_string(),
            },
            ScoredResult {
                doc_id: "doc3".to_string(),
                score: 3.0,
                text: "text3".to_string(),
            },
        ];

        let fused = weighted_sum_fusion(&vector_results, &bm25_results, 0.6, 0.4, 10).unwrap();

        // should have 3 documents
        assert_eq!(fused.len(), 3);

        // doc2 appears in both, should score highest
        assert_eq!(fused[0].doc_id, "doc2");
        assert!(fused[0].score > 0.5);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let vector_results = vec![
            ("doc1".to_string(), 0.9, "text1".to_string()),
            ("doc2".to_string(), 0.7, "text2".to_string()),
        ];

        let bm25_results = vec![
            ScoredResult {
                doc_id: "doc2".to_string(),
                score: 5.0,
                text: "text2".to_string(),
            },
            ScoredResult {
                doc_id: "doc3".to_string(),
                score: 3.0,
                text: "text3".to_string(),
            },
        ];

        let fused = reciprocal_rank_fusion(&vector_results, &bm25_results, 60, 10).unwrap();

        // should have 3 documents
        assert_eq!(fused.len(), 3);

        // doc2 appears in both (rank 1 in vector, rank 0 in bm25)
        assert_eq!(fused[0].doc_id, "doc2");
    }
}
