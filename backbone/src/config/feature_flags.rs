/// Feature flags for optional components
#[derive(Debug, Clone, Copy)]
pub struct FeatureFlags {
    pub enable_pdf_ocr: bool,
    pub enable_reranking: bool,
    pub enable_section_chunking: bool,
    pub enable_citation_formatting: bool,
    pub enable_context_compression: bool,
    pub enable_extractive_summary: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            enable_pdf_ocr: false,
            enable_reranking: false,
            enable_section_chunking: true,
            enable_citation_formatting: true,
            enable_context_compression: false,
            enable_extractive_summary: false,
        }
    }
}
