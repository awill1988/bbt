use serde::{Deserialize, Serialize};

/// Bounding box for a text element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Check if this box overlaps with another
    pub fn overlaps(&self, other: &BoundingBox) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

/// A single line of text with layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    /// The text content
    pub text: String,

    /// Font size in points
    pub font_size: f32,

    /// Bounding box of the text
    pub bbox: BoundingBox,

    /// Whether this text was extracted via OCR
    pub is_ocr: bool,
}

impl TextLine {
    pub fn new(text: String, font_size: f32, bbox: BoundingBox, is_ocr: bool) -> Self {
        Self {
            text,
            font_size,
            bbox,
            is_ocr,
        }
    }

    /// Check if this line is likely a heading based on font size
    pub fn is_heading(&self, median_font_size: f32, threshold_ratio: f32) -> bool {
        self.font_size > median_font_size * threshold_ratio
    }
}

/// Content of a single PDF page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageContent {
    /// Page number (1-indexed)
    pub page_number: usize,

    /// Text lines on this page
    pub lines: Vec<TextLine>,

    /// Page dimensions (width, height) in points
    pub dimensions: (f32, f32),
}

impl PageContent {
    pub fn new(page_number: usize, dimensions: (f32, f32)) -> Self {
        Self {
            page_number,
            lines: Vec::new(),
            dimensions,
        }
    }

    /// Add a text line to this page
    pub fn add_line(&mut self, line: TextLine) {
        self.lines.push(line);
    }

    /// Get all text on this page as a single string
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Calculate median font size on this page
    pub fn median_font_size(&self) -> f32 {
        if self.lines.is_empty() {
            return 12.0; // default
        }

        let mut font_sizes: Vec<f32> = self.lines.iter().map(|line| line.font_size).collect();
        font_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = font_sizes.len() / 2;
        if font_sizes.len() % 2 == 0 {
            (font_sizes[mid - 1] + font_sizes[mid]) / 2.0
        } else {
            font_sizes[mid]
        }
    }

    /// Count total characters on page
    pub fn char_count(&self) -> usize {
        self.lines.iter().map(|line| line.text.len()).sum()
    }
}

/// PDF document metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PdfMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub subject: Option<String>,
    pub keywords: Option<String>,
    pub creator: Option<String>,
    pub producer: Option<String>,
}

/// A complete PDF document with layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfDocument {
    /// All pages in the document
    pub pages: Vec<PageContent>,

    /// Document metadata
    pub metadata: PdfMetadata,
}

impl PdfDocument {
    pub fn new(metadata: PdfMetadata) -> Self {
        Self {
            pages: Vec::new(),
            metadata,
        }
    }

    /// Add a page to the document
    pub fn add_page(&mut self, page: PageContent) {
        self.pages.push(page);
    }

    /// Get total page count
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get all text from all pages
    pub fn text(&self) -> String {
        self.pages
            .iter()
            .map(|page| page.text())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Calculate overall median font size across document
    pub fn median_font_size(&self) -> f32 {
        let mut all_font_sizes: Vec<f32> = self
            .pages
            .iter()
            .flat_map(|page| page.lines.iter().map(|line| line.font_size))
            .collect();

        if all_font_sizes.is_empty() {
            return 12.0;
        }

        all_font_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = all_font_sizes.len() / 2;
        if all_font_sizes.len() % 2 == 0 {
            (all_font_sizes[mid - 1] + all_font_sizes[mid]) / 2.0
        } else {
            all_font_sizes[mid]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_overlaps() {
        let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let box2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        let box3 = BoundingBox::new(20.0, 20.0, 10.0, 10.0);

        assert!(box1.overlaps(&box2));
        assert!(!box1.overlaps(&box3));
    }

    #[test]
    fn test_text_line_is_heading() {
        let line = TextLine::new(
            "Heading".to_string(),
            16.0,
            BoundingBox::new(0.0, 0.0, 100.0, 20.0),
            false,
        );

        assert!(line.is_heading(12.0, 1.2));
        assert!(!line.is_heading(12.0, 1.5));
    }

    #[test]
    fn test_page_content_text() {
        let mut page = PageContent::new(1, (600.0, 800.0));
        page.add_line(TextLine::new(
            "Line 1".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 100.0, 12.0),
            false,
        ));
        page.add_line(TextLine::new(
            "Line 2".to_string(),
            12.0,
            BoundingBox::new(0.0, 15.0, 100.0, 12.0),
            false,
        ));

        assert_eq!(page.text(), "Line 1\nLine 2");
        assert_eq!(page.char_count(), 12);
    }

    #[test]
    fn test_page_content_median_font_size() {
        let mut page = PageContent::new(1, (600.0, 800.0));
        page.add_line(TextLine::new(
            "Text".to_string(),
            10.0,
            BoundingBox::new(0.0, 0.0, 100.0, 10.0),
            false,
        ));
        page.add_line(TextLine::new(
            "Text".to_string(),
            12.0,
            BoundingBox::new(0.0, 15.0, 100.0, 12.0),
            false,
        ));
        page.add_line(TextLine::new(
            "Text".to_string(),
            14.0,
            BoundingBox::new(0.0, 30.0, 100.0, 14.0),
            false,
        ));

        assert_eq!(page.median_font_size(), 12.0);
    }

    #[test]
    fn test_pdf_document() {
        let mut doc = PdfDocument::new(PdfMetadata::default());

        let mut page1 = PageContent::new(1, (600.0, 800.0));
        page1.add_line(TextLine::new(
            "Page 1".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 100.0, 12.0),
            false,
        ));

        let mut page2 = PageContent::new(2, (600.0, 800.0));
        page2.add_line(TextLine::new(
            "Page 2".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 100.0, 12.0),
            false,
        ));

        doc.add_page(page1);
        doc.add_page(page2);

        assert_eq!(doc.page_count(), 2);
        assert_eq!(doc.text(), "Page 1\n\nPage 2");
    }
}
