use super::layout::{PdfDocument, TextLine};
use serde::{Deserialize, Serialize};

/// A section of a document with heading and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// The heading text
    pub heading: String,

    /// The full content of the section (including the heading)
    pub content: String,

    /// The level of the heading (1 = largest, higher = smaller)
    pub level: usize,

    /// Page number where this section starts (1-indexed)
    pub start_page: usize,

    /// Page number where this section ends (1-indexed)
    pub end_page: usize,

    /// Font size of the heading
    pub heading_font_size: f32,
}

impl Section {
    pub fn new(
        heading: String,
        content: String,
        level: usize,
        start_page: usize,
        end_page: usize,
        heading_font_size: f32,
    ) -> Self {
        Self {
            heading,
            content,
            level,
            start_page,
            end_page,
            heading_font_size,
        }
    }
}

/// Configuration for section detection
#[derive(Debug, Clone)]
pub struct SectionDetectorConfig {
    /// Font size threshold ratio (heading must be this much larger than median)
    pub heading_threshold_ratio: f32,

    /// Minimum heading font size (absolute)
    pub min_heading_font_size: f32,

    /// Maximum number of heading levels to detect
    pub max_levels: usize,

    /// Minimum length for a heading (characters)
    pub min_heading_length: usize,

    /// Maximum length for a heading (characters)
    pub max_heading_length: usize,
}

impl Default for SectionDetectorConfig {
    fn default() -> Self {
        Self {
            heading_threshold_ratio: 1.2,
            min_heading_font_size: 10.0,
            max_levels: 5,
            min_heading_length: 3,
            max_heading_length: 200,
        }
    }
}

/// Detector for identifying sections in a PDF document
pub struct SectionDetector {
    config: SectionDetectorConfig,
}

impl SectionDetector {
    pub fn new(config: SectionDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect sections in a PDF document
    pub fn detect_sections(&self, document: &PdfDocument) -> Vec<Section> {
        let median_font_size = document.median_font_size();
        let mut sections = Vec::new();

        // Collect all potential headings with their page numbers
        let mut headings: Vec<(usize, &TextLine)> = Vec::new();

        for page in &document.pages {
            for line in &page.lines {
                if self.is_heading(line, median_font_size) {
                    headings.push((page.page_number, line));
                }
            }
        }

        // If no headings found, return the entire document as one section
        if headings.is_empty() {
            let content = document.text();
            return vec![Section::new(
                "Document".to_string(),
                content,
                1,
                1,
                document.page_count(),
                median_font_size,
            )];
        }

        // Assign levels based on font size (larger font = lower level number)
        let mut font_sizes: Vec<f32> = headings.iter().map(|(_, line)| line.font_size).collect();
        font_sizes.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
        font_sizes.dedup();

        // Helper function to get level for a given font size
        let get_level = |font_size: f32| -> usize {
            font_sizes
                .iter()
                .position(|&size| (size - font_size).abs() < 0.01)
                .map(|i| (i + 1).min(self.config.max_levels))
                .unwrap_or(1)
        };

        // Build sections
        for i in 0..headings.len() {
            let (page_num, heading_line) = headings[i];
            let level = get_level(heading_line.font_size);

            // Determine end page for this section
            let end_page = if i + 1 < headings.len() {
                // Section ends at the page before the next heading of same or higher level
                let mut end = document.page_count();
                for j in (i + 1)..headings.len() {
                    let (next_page, next_line) = headings[j];
                    let next_level = get_level(next_line.font_size);
                    if next_level <= level {
                        end = next_page.saturating_sub(1).max(page_num);
                        break;
                    }
                }
                end
            } else {
                document.page_count()
            };

            // Extract content from start_page to end_page
            let mut content_lines = Vec::new();
            let mut in_section = false;

            for page in &document.pages {
                if page.page_number < page_num {
                    continue;
                }
                if page.page_number > end_page {
                    break;
                }

                for line in &page.lines {
                    // Skip the heading itself on the first page
                    if page.page_number == page_num && !in_section {
                        if line.text == heading_line.text {
                            in_section = true;
                        }
                        content_lines.push(line.text.clone());
                        continue;
                    }

                    content_lines.push(line.text.clone());
                }
            }

            let content = content_lines.join("\n");

            sections.push(Section::new(
                heading_line.text.clone(),
                content,
                level,
                page_num,
                end_page,
                heading_line.font_size,
            ));
        }

        sections
    }

    /// Check if a text line is likely a heading
    fn is_heading(&self, line: &TextLine, median_font_size: f32) -> bool {
        // Check font size threshold
        if line.font_size < self.config.min_heading_font_size {
            return false;
        }

        if line.font_size <= median_font_size * self.config.heading_threshold_ratio {
            return false;
        }

        // Check text length
        let text_len = line.text.trim().len();
        if text_len < self.config.min_heading_length || text_len > self.config.max_heading_length {
            return false;
        }

        // Check if text is all uppercase (common for headings)
        let is_uppercase = line.text.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_uppercase());

        // Check if text ends with period (headings usually don't)
        let ends_with_period = line.text.trim().ends_with('.');

        // Headings are more likely if uppercase and don't end with period
        // But we don't require these conditions
        if is_uppercase && !ends_with_period {
            return true;
        }

        // Even if not uppercase, still consider it a heading based on font size
        !ends_with_period || text_len < 50
    }
}

impl Default for SectionDetector {
    fn default() -> Self {
        Self::new(SectionDetectorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::pdf::layout::{BoundingBox, PageContent, PdfDocument, PdfMetadata};

    #[test]
    fn test_is_heading() {
        let detector = SectionDetector::default();
        let median = 12.0;

        // Large font, short text - should be heading
        let heading = TextLine::new(
            "Chapter 1".to_string(),
            16.0,
            BoundingBox::new(0.0, 0.0, 100.0, 16.0),
            false,
        );
        assert!(detector.is_heading(&heading, median));

        // Normal font - should not be heading
        let normal = TextLine::new(
            "This is normal text.".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 200.0, 12.0),
            false,
        );
        assert!(!detector.is_heading(&normal, median));

        // Large font but ends with period and too long - less likely
        let paragraph = TextLine::new(
            "This is a very long line that is probably not a heading even though it has large font.".to_string(),
            16.0,
            BoundingBox::new(0.0, 0.0, 500.0, 16.0),
            false,
        );
        assert!(!detector.is_heading(&paragraph, median));
    }

    #[test]
    fn test_detect_sections_no_headings() {
        let detector = SectionDetector::default();
        let mut doc = PdfDocument::new(PdfMetadata::default());

        let mut page = PageContent::new(1, (600.0, 800.0));
        page.add_line(TextLine::new(
            "Normal text".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 100.0, 12.0),
            false,
        ));
        doc.add_page(page);

        let sections = detector.detect_sections(&doc);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].heading, "Document");
        assert_eq!(sections[0].start_page, 1);
        assert_eq!(sections[0].end_page, 1);
    }

    #[test]
    fn test_detect_sections_with_headings() {
        // Use a lower threshold for testing
        let mut config = SectionDetectorConfig::default();
        config.heading_threshold_ratio = 1.1; // lower threshold
        config.min_heading_font_size = 14.0;
        let detector = SectionDetector::new(config);
        let mut doc = PdfDocument::new(PdfMetadata::default());

        // Page 1 with heading
        let mut page1 = PageContent::new(1, (600.0, 800.0));
        page1.add_line(TextLine::new(
            "Introduction".to_string(),
            18.0,
            BoundingBox::new(0.0, 0.0, 100.0, 18.0),
            false,
        ));
        page1.add_line(TextLine::new(
            "This is the intro.".to_string(),
            12.0,
            BoundingBox::new(0.0, 20.0, 200.0, 12.0),
            false,
        ));
        doc.add_page(page1);

        // Page 2 with another heading
        let mut page2 = PageContent::new(2, (600.0, 800.0));
        page2.add_line(TextLine::new(
            "Chapter 1".to_string(),
            18.0,
            BoundingBox::new(0.0, 0.0, 100.0, 18.0),
            false,
        ));
        page2.add_line(TextLine::new(
            "Chapter content.".to_string(),
            12.0,
            BoundingBox::new(0.0, 20.0, 200.0, 12.0),
            false,
        ));
        doc.add_page(page2);

        let sections = detector.detect_sections(&doc);
        assert_eq!(sections.len(), 2);

        assert_eq!(sections[0].heading, "Introduction");
        assert_eq!(sections[0].start_page, 1);
        assert_eq!(sections[0].end_page, 1);

        assert_eq!(sections[1].heading, "Chapter 1");
        assert_eq!(sections[1].start_page, 2);
        assert_eq!(sections[1].end_page, 2);
    }

    #[test]
    fn test_section_levels() {
        // Use a lower threshold to detect all three headings
        let mut config = SectionDetectorConfig::default();
        config.heading_threshold_ratio = 1.05; // very low threshold for testing
        config.min_heading_font_size = 10.0;
        let detector = SectionDetector::new(config);
        let mut doc = PdfDocument::new(PdfMetadata::default());

        let mut page = PageContent::new(1, (600.0, 800.0));

        // Level 1 - largest font
        page.add_line(TextLine::new(
            "Main Title".to_string(),
            24.0,
            BoundingBox::new(0.0, 0.0, 100.0, 24.0),
            false,
        ));

        // Add some body text to lower median
        page.add_line(TextLine::new(
            "Some body text here.".to_string(),
            12.0,
            BoundingBox::new(0.0, 15.0, 200.0, 12.0),
            false,
        ));

        // Level 2 - medium font
        page.add_line(TextLine::new(
            "Subtitle".to_string(),
            18.0,
            BoundingBox::new(0.0, 30.0, 100.0, 18.0),
            false,
        ));

        // Add more body text
        page.add_line(TextLine::new(
            "More body text.".to_string(),
            12.0,
            BoundingBox::new(0.0, 45.0, 200.0, 12.0),
            false,
        ));

        // Level 3 - smaller font
        page.add_line(TextLine::new(
            "Section".to_string(),
            14.0,
            BoundingBox::new(0.0, 60.0, 100.0, 14.0),
            false,
        ));

        // And more body text
        page.add_line(TextLine::new(
            "Even more body text.".to_string(),
            12.0,
            BoundingBox::new(0.0, 75.0, 200.0, 12.0),
            false,
        ));

        doc.add_page(page);

        let sections = detector.detect_sections(&doc);
        assert_eq!(sections.len(), 3);

        assert_eq!(sections[0].level, 1); // Main Title
        assert_eq!(sections[1].level, 2); // Subtitle
        assert_eq!(sections[2].level, 3); // Section
    }
}
