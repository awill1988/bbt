# Design: temporal embeddings for document analysis

This tool implements **temporal feature augmentation** for document embeddings, combining semantic text embeddings with normalized temporal features to improve clustering accuracy for time-aware organization of information.

## Approach

Text embeddings (384-dim) are concatenated with normalized timestamp features (1-dim) to create 385-dimensional vectors. Timestamps are normalized to a 0-1 range across the dataset, with missing timestamps assigned a neutral value (0.5). This allows clustering algorithms to naturally balance semantic similarity with temporal proximity.

## Why this works

Traditional text-only embeddings cluster documents purely by content similarity. Temporal features enable the model to discover natural boundaries where both content AND time period matter—critical for accurate organization and retrieval of information when document age correlates with relevance or context.

## Implementation details

### Temporal feature extraction

Timestamps are extracted from document metadata during normalization:
- Supports multiple field names: `dateAdded`, `date_added`, `timestamp`, `created`
- Firefox timestamps are in microseconds since epoch (example from original project)
- Missing timestamps are preserved as `None`

### Feature normalization

Temporal features are computed per-batch:
- Valid timestamps are converted to seconds and min-max normalized to [0, 1]
- Missing timestamps default to 0.5 (neutral value)
- Single feature dimension keeps overhead minimal

### Embedding concatenation

Text embeddings and temporal features are concatenated before storage:
- Text model generates 384-dim vectors
- Temporal feature (1-dim) appended to create 385-dim final embeddings
- Both features stored as single vector for efficient similarity search

### Visualization enhancements

Clustering and neighbor analysis now display temporal context:
- **Cluster command**: Shows date range (min/max) for each cluster
- **Retrieval command**: Displays document metadata alongside similarity scores

## Research foundations

- **Multimodal embeddings**: Combining heterogeneous features (text + time) in a shared embedding space (Baltrušaitis et al., 2019)
- **Temporal knowledge graph embeddings**: Time-aware representations for entities that change over time (Trivedi et al., 2017; Xu et al., 2020)
- **Feature concatenation for retrieval**: Simple concatenation of complementary features often outperforms complex fusion methods for downstream tasks (Kiela et al., 2018)

**References:**
- Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE TPAMI*, 41(2), 423-443.
- Trivedi, R., Dai, H., Wang, Y., & Song, L. (2017). Know-Evolve: Deep temporal reasoning for dynamic knowledge graphs. *ICML*.
- Xu, C., Nayyeri, M., Alkhoury, F., Yazdi, H. S., & Lehmann, J. (2020). Temporal knowledge graph embedding model based on additive time series decomposition. *arXiv:1911.07893*.
- Kiela, D., Grave, E., Joulin, A., & Mikolov, T. (2018). Efficient large-scale multi-modal classification. *AAAI*.

## Trade-offs

### Benefits
- Automatic temporal-semantic grouping without explicit logic
- Minimal overhead (single additional dimension)
- Handles missing timestamps gracefully
- Works with existing similarity search infrastructure

### Limitations
- Linear time representation (doesn't capture cyclical patterns like day-of-week)
- Fixed weighting between temporal and semantic features
- Requires dataset-wide normalization (can't embed single item in isolation)

## Future enhancements

- **Learned temporal weighting**: Train a model to learn optimal temporal feature weight
- **Multi-scale temporal features**: Add year, month, day-of-week as separate cyclic features
- **Configurable temporal influence**: CLI flag to adjust temporal vs semantic balance
- **Temporal decay**: Weight recent documents more heavily for recency-aware clustering