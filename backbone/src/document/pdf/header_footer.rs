use super::layout::{PdfDocument, TextLine};
use std::collections::HashMap;

/// Configuration for header/footer detection
#[derive(Debug, Clone)]
pub struct HeaderFooterConfig {
    /// Top margin (in points) to consider as potential header area
    pub top_margin: f32,

    /// Bottom margin (in points) to consider as potential footer area
    pub bottom_margin: f32,

    /// Minimum frequency (as fraction of total pages) for a line to be considered header/footer
    pub min_frequency: f32,

    /// Maximum number of unique lines to consider (prevents false positives)
    pub max_candidates: usize,

    /// Minimum page count for detection (skip if document is too short)
    pub min_pages: usize,
}

impl Default for HeaderFooterConfig {
    fn default() -> Self {
        Self {
            top_margin: 72.0,    // 1 inch
            bottom_margin: 72.0, // 1 inch
            min_frequency: 0.5,  // must appear on at least 50% of pages
            max_candidates: 20,
            min_pages: 3,
        }
    }
}

/// Detector for identifying and removing headers/footers
pub struct HeaderFooterDetector {
    config: HeaderFooterConfig,
}

impl HeaderFooterDetector {
    pub fn new(config: HeaderFooterConfig) -> Self {
        Self { config }
    }

    /// Detect and remove headers and footers from a document
    pub fn clean_document(&self, document: &mut PdfDocument) {
        if document.pages.len() < self.config.min_pages {
            return; // skip detection for short documents
        }

        let headers = self.detect_repeated_lines(document, true);
        let footers = self.detect_repeated_lines(document, false);

        // Remove detected headers and footers
        for page in &mut document.pages {
            page.lines.retain(|line| {
                !headers.contains(&line.text) && !footers.contains(&line.text)
            });
        }
    }

    /// Detect repeated lines in header or footer area
    fn detect_repeated_lines(&self, document: &PdfDocument, is_header: bool) -> Vec<String> {
        let mut line_counts: HashMap<String, usize> = HashMap::new();
        let total_pages = document.pages.len();

        // Count occurrences of each line in the margin area
        for page in &document.pages {
            let candidates = self.get_margin_lines(page, is_header);

            for line in candidates {
                let normalized = self.normalize_line(&line.text);
                if !normalized.is_empty() {
                    *line_counts.entry(normalized).or_insert(0) += 1;
                }
            }
        }

        // Filter lines that appear frequently enough
        let min_count = (total_pages as f32 * self.config.min_frequency).ceil() as usize;

        let mut repeated: Vec<(String, usize)> = line_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_count)
            .collect();

        // Sort by frequency (descending) and limit to max_candidates
        repeated.sort_by(|a, b| b.1.cmp(&a.1));
        repeated.truncate(self.config.max_candidates);

        repeated.into_iter().map(|(text, _)| text).collect()
    }

    /// Get lines in the margin area (header or footer)
    fn get_margin_lines<'a>(&self, page: &'a super::layout::PageContent, is_header: bool) -> Vec<&'a TextLine> {
        let page_height = page.dimensions.1;

        page.lines
            .iter()
            .filter(|line| {
                if is_header {
                    // Header: top of page
                    let distance_from_top = page_height - line.bbox.y;
                    distance_from_top <= self.config.top_margin
                } else {
                    // Footer: bottom of page
                    line.bbox.y <= self.config.bottom_margin
                }
            })
            .collect()
    }

    /// Normalize line text for comparison
    fn normalize_line(&self, text: &str) -> String {
        // Remove extra whitespace
        let mut result = String::new();
        let mut prev_was_space = true; // start as true to trim leading spaces

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                }
                prev_was_space = true;
            } else {
                result.push(ch);
                prev_was_space = false;
            }
        }

        result.trim().to_string()
    }

    /// Get statistics about detected headers/footers
    pub fn get_stats(&self, document: &PdfDocument) -> HeaderFooterStats {
        if document.pages.len() < self.config.min_pages {
            return HeaderFooterStats::default();
        }

        let headers = self.detect_repeated_lines(document, true);
        let footers = self.detect_repeated_lines(document, false);

        let mut total_removed = 0;
        for page in &document.pages {
            for line in &page.lines {
                if headers.contains(&line.text) || footers.contains(&line.text) {
                    total_removed += 1;
                }
            }
        }

        HeaderFooterStats {
            header_count: headers.len(),
            footer_count: footers.len(),
            total_lines_removed: total_removed,
            headers,
            footers,
        }
    }
}

impl Default for HeaderFooterDetector {
    fn default() -> Self {
        Self::new(HeaderFooterConfig::default())
    }
}

/// Statistics about header/footer detection
#[derive(Debug, Clone, Default)]
pub struct HeaderFooterStats {
    pub header_count: usize,
    pub footer_count: usize,
    pub total_lines_removed: usize,
    pub headers: Vec<String>,
    pub footers: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::pdf::layout::{BoundingBox, PageContent, PdfDocument, PdfMetadata};

    fn create_test_page(page_num: usize, include_header: bool, include_footer: bool) -> PageContent {
        let mut page = PageContent::new(page_num, (600.0, 800.0));

        if include_header {
            // Header at top of page (y near page height)
            page.add_line(TextLine::new(
                "Header Text".to_string(),
                10.0,
                BoundingBox::new(100.0, 750.0, 100.0, 10.0),
                false,
            ));
        }

        // Body content in middle
        page.add_line(TextLine::new(
            format!("Page {} content", page_num),
            12.0,
            BoundingBox::new(100.0, 400.0, 200.0, 12.0),
            false,
        ));

        if include_footer {
            // Footer at bottom of page (y near 0)
            page.add_line(TextLine::new(
                format!("Page {}", page_num),
                10.0,
                BoundingBox::new(100.0, 30.0, 100.0, 10.0),
                false,
            ));
        }

        page
    }

    #[test]
    fn test_detect_headers() {
        let detector = HeaderFooterDetector::default();
        let mut doc = PdfDocument::new(PdfMetadata::default());

        // Add 5 pages, all with same header
        for i in 1..=5 {
            doc.add_page(create_test_page(i, true, false));
        }

        let headers = detector.detect_repeated_lines(&doc, true);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0], "Header Text");
    }

    #[test]
    fn test_detect_footers() {
        let detector = HeaderFooterDetector::default();
        let mut doc = PdfDocument::new(PdfMetadata::default());

        // Add 5 pages with page numbers in footer
        for i in 1..=5 {
            doc.add_page(create_test_page(i, false, true));
        }

        let footers = detector.detect_repeated_lines(&doc, false);
        // Each page has different footer (page number), so none should be detected as repeated
        // unless min_frequency is low enough
        // With default min_frequency of 0.5, need to appear on 3/5 pages
        assert_eq!(footers.len(), 0);
    }

    #[test]
    fn test_clean_document() {
        let detector = HeaderFooterDetector::default();
        let mut doc = PdfDocument::new(PdfMetadata::default());

        // Add 5 pages with same header
        for i in 1..=5 {
            doc.add_page(create_test_page(i, true, false));
        }

        // Before cleaning, each page has 2 lines (header + content)
        assert_eq!(doc.pages[0].lines.len(), 2);

        detector.clean_document(&mut doc);

        // After cleaning, each page should have 1 line (content only)
        assert_eq!(doc.pages[0].lines.len(), 1);
        assert!(doc.pages[0].lines[0].text.contains("content"));
    }

    #[test]
    fn test_get_stats() {
        let detector = HeaderFooterDetector::default();
        let mut doc = PdfDocument::new(PdfMetadata::default());

        // Add 5 pages with same header
        for i in 1..=5 {
            doc.add_page(create_test_page(i, true, false));
        }

        let stats = detector.get_stats(&doc);
        assert_eq!(stats.header_count, 1);
        assert_eq!(stats.footer_count, 0);
        assert_eq!(stats.total_lines_removed, 5); // header appears on 5 pages
    }

    #[test]
    fn test_normalize_line() {
        let detector = HeaderFooterDetector::default();

        assert_eq!(
            detector.normalize_line("  Multiple   spaces  "),
            "Multiple spaces"
        );

        assert_eq!(
            detector.normalize_line("Normal text"),
            "Normal text"
        );

        assert_eq!(detector.normalize_line("   "), "");
    }

    #[test]
    fn test_skip_short_documents() {
        let detector = HeaderFooterDetector::default();
        let mut doc = PdfDocument::new(PdfMetadata::default());

        // Add only 2 pages (less than min_pages = 3)
        doc.add_page(create_test_page(1, true, false));
        doc.add_page(create_test_page(2, true, false));

        let stats = detector.get_stats(&doc);
        assert_eq!(stats.header_count, 0);
        assert_eq!(stats.total_lines_removed, 0);
    }
}
