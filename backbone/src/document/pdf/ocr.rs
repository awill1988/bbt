use super::layout::{PageContent, TextLine};
#[cfg(feature = "ocr")]
use super::layout::BoundingBox;
use crate::error::{BbtError, Result};

#[cfg(feature = "ocr")]
use tesseract::Tesseract;

/// Configuration for OCR processing
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Minimum character count threshold - pages with fewer chars trigger OCR
    pub min_chars_threshold: usize,

    /// Language code for tesseract (e.g., "eng" for English)
    pub language: String,

    /// OCR engine mode (0=legacy, 1=neural nets, 2=legacy+neural, 3=default)
    pub engine_mode: u8,

    /// Page segmentation mode (3=auto, 6=single uniform block, 7=single line, etc.)
    pub page_seg_mode: u8,

    /// Enable OCR for all pages regardless of text content
    pub force_ocr: bool,

    /// DPI for rendering PDF pages to images (default 150)
    pub render_dpi: u32,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            min_chars_threshold: 50,
            language: "eng".to_string(),
            engine_mode: 3, // default
            page_seg_mode: 3, // auto
            force_ocr: false,
            render_dpi: 150,
        }
    }
}

/// OCR processor for extracting text from images
pub struct OcrProcessor {
    config: OcrConfig,
}

impl OcrProcessor {
    /// Create a new OCR processor
    pub fn new(config: OcrConfig) -> Result<Self> {
        #[cfg(feature = "ocr")]
        {
            // verify tesseract is available by attempting initialization
            let _ = Self::create_tesseract(&config)?;
            tracing::info!(
                "ocr processor initialized with language={}, dpi={}",
                config.language,
                config.render_dpi
            );
        }

        Ok(Self { config })
    }

    #[cfg(feature = "ocr")]
    fn create_tesseract(config: &OcrConfig) -> Result<Tesseract> {
        Tesseract::new(None, Some(&config.language))
            .map_err(|e| BbtError::Model(format!("failed to initialize tesseract: {:?}", e)))
    }

    /// Check if OCR is available
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "ocr")]
        {
            Self::create_tesseract(&self.config).is_ok()
        }

        #[cfg(not(feature = "ocr"))]
        {
            false
        }
    }

    /// Check if a page needs OCR processing
    pub fn needs_ocr(&self, page: &PageContent) -> bool {
        if self.config.force_ocr {
            return true;
        }

        // Check if page has very little text
        let char_count = page.char_count();
        char_count < self.config.min_chars_threshold
    }

    /// Process a page with OCR using rendered image data
    ///
    /// # arguments
    /// * `page` - the page content to augment with OCR text
    /// * `image_data` - raw image bytes (RGB or grayscale)
    /// * `width` - image width in pixels
    /// * `height` - image height in pixels
    /// * `bytes_per_pixel` - 1 for grayscale, 3 for RGB, 4 for RGBA
    ///
    /// # returns
    /// true if OCR was performed and text was added
    #[allow(unused_variables)]
    pub fn process_page_with_image(
        &self,
        page: &mut PageContent,
        image_data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<bool> {
        if !self.is_available() {
            return Ok(false);
        }

        if !self.needs_ocr(page) {
            return Ok(false);
        }

        #[cfg(feature = "ocr")]
        {
            let ocr_lines = self.run_ocr(image_data, width, height, bytes_per_pixel, page.dimensions)?;

            if ocr_lines.is_empty() {
                return Ok(false);
            }

            tracing::debug!(
                "ocr extracted {} lines from page {}",
                ocr_lines.len(),
                page.page_number
            );

            // add ocr lines to page
            for line in ocr_lines {
                page.add_line(line);
            }

            Ok(true)
        }

        #[cfg(not(feature = "ocr"))]
        {
            Ok(false)
        }
    }

    /// Process a page with OCR if needed (legacy interface)
    ///
    /// Note: This requires the caller to have already rendered the page.
    /// For full OCR workflow, use process_page_with_image with rendered bitmap.
    #[allow(unused_variables)]
    pub fn process_page(&self, page: &mut PageContent) -> Result<bool> {
        if !self.is_available() {
            return Ok(false);
        }

        if !self.needs_ocr(page) {
            return Ok(false);
        }

        // cannot perform OCR without image data
        // caller should use process_page_with_image with rendered bitmap
        tracing::debug!(
            "page {} needs ocr but no image data provided",
            page.page_number
        );
        Ok(false)
    }

    /// Run OCR on image data and return extracted text lines
    #[cfg(feature = "ocr")]
    fn run_ocr(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        page_dimensions: (f32, f32),
    ) -> Result<Vec<TextLine>> {
        let mut tess = Self::create_tesseract(&self.config)?;

        // set image data in tesseract
        // tesseract expects raw pixel data in row-major order
        tess = tess.set_frame(
            image_data,
            width as i32,
            height as i32,
            bytes_per_pixel as i32,
            (width * bytes_per_pixel) as i32, // bytes per line
        ).map_err(|e| BbtError::Model(format!("failed to set image frame: {:?}", e)))?;

        // run recognition
        let text = tess.get_text()
            .map_err(|e| BbtError::Model(format!("tesseract recognition failed: {:?}", e)))?;

        // parse recognized text into TextLine objects
        // scale factor from image pixels to PDF points
        let scale_x = page_dimensions.0 / width as f32;
        let scale_y = page_dimensions.1 / height as f32;

        let mut lines = Vec::new();
        let line_height = 12.0; // estimated line height in points

        for (line_idx, line_text) in text.lines().enumerate() {
            let trimmed = line_text.trim();
            if trimmed.is_empty() {
                continue;
            }

            // estimate bounding box based on line position
            // note: for precise bounding boxes, would need tesseract's HOCR output
            let y_pos = page_dimensions.1 - ((line_idx as f32 + 1.0) * line_height * 1.5);
            let bbox = BoundingBox::new(
                50.0,                      // left margin
                y_pos.max(0.0),            // y position
                page_dimensions.0 - 100.0, // width
                line_height,               // height
            );

            lines.push(TextLine::new(
                trimmed.to_string(),
                line_height,
                bbox,
                true, // is_ocr = true
            ));
        }

        Ok(lines)
    }

    /// Process raw image buffer with OCR (convenience method)
    ///
    /// # arguments
    /// * `image_data` - raw image bytes
    /// * `width` - image width in pixels
    /// * `height` - image height in pixels
    /// * `bytes_per_pixel` - 1 for grayscale, 3 for RGB, 4 for RGBA
    ///
    /// # returns
    /// vector of extracted text lines
    #[allow(unused_variables)]
    pub fn process_image(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<Vec<TextLine>> {
        #[cfg(feature = "ocr")]
        {
            // use image dimensions as page dimensions (1:1 scale)
            self.run_ocr(
                image_data,
                width,
                height,
                bytes_per_pixel,
                (width as f32, height as f32),
            )
        }

        #[cfg(not(feature = "ocr"))]
        {
            Err(BbtError::Model("ocr feature not enabled".to_string()))
        }
    }

    /// Get OCR statistics for a document
    pub fn get_stats(&self, pages: &[PageContent]) -> OcrStats {
        let total_pages = pages.len();
        let mut pages_needing_ocr = 0;
        let mut pages_with_ocr = 0;

        for page in pages {
            if self.needs_ocr(page) {
                pages_needing_ocr += 1;
            }

            // Count pages that have OCR text
            if page.lines.iter().any(|line| line.is_ocr) {
                pages_with_ocr += 1;
            }
        }

        OcrStats {
            total_pages,
            pages_needing_ocr,
            pages_with_ocr,
            ocr_available: self.is_available(),
        }
    }

    /// Get the configured render DPI
    pub fn render_dpi(&self) -> u32 {
        self.config.render_dpi
    }

    /// Get the OCR configuration
    pub fn config(&self) -> &OcrConfig {
        &self.config
    }
}

impl Default for OcrProcessor {
    fn default() -> Self {
        Self::new(OcrConfig::default()).expect("failed to initialize ocr processor")
    }
}

/// Statistics about OCR processing
#[derive(Debug, Clone)]
pub struct OcrStats {
    pub total_pages: usize,
    pub pages_needing_ocr: usize,
    pub pages_with_ocr: usize,
    pub ocr_available: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::pdf::layout::{BoundingBox, PageContent, TextLine};

    #[test]
    fn test_ocr_config_default() {
        let config = OcrConfig::default();
        assert_eq!(config.min_chars_threshold, 50);
        assert_eq!(config.language, "eng");
        assert!(!config.force_ocr);
    }

    #[test]
    fn test_ocr_processor_new() {
        let config = OcrConfig::default();
        let result = OcrProcessor::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_needs_ocr() {
        let processor = OcrProcessor::default();

        // Page with very little text should need OCR
        let mut page_low_text = PageContent::new(1, (600.0, 800.0));
        page_low_text.add_line(TextLine::new(
            "Hi".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 20.0, 12.0),
            false,
        ));
        assert!(processor.needs_ocr(&page_low_text));

        // Page with sufficient text should not need OCR
        let mut page_high_text = PageContent::new(1, (600.0, 800.0));
        page_high_text.add_line(TextLine::new(
            "This is a much longer line of text that exceeds the threshold.".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 400.0, 12.0),
            false,
        ));
        assert!(!processor.needs_ocr(&page_high_text));
    }

    #[test]
    fn test_force_ocr() {
        let mut config = OcrConfig::default();
        config.force_ocr = true;
        let processor = OcrProcessor::new(config).unwrap();

        // Even page with lots of text should need OCR when force_ocr is true
        let mut page = PageContent::new(1, (600.0, 800.0));
        page.add_line(TextLine::new(
            "This is a much longer line of text that exceeds the threshold.".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 400.0, 12.0),
            false,
        ));
        assert!(processor.needs_ocr(&page));
    }

    #[test]
    fn test_get_stats() {
        let processor = OcrProcessor::default();

        let mut pages = Vec::new();

        // Page 1: low text, no OCR performed
        let mut page1 = PageContent::new(1, (600.0, 800.0));
        page1.add_line(TextLine::new(
            "Hi".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 20.0, 12.0),
            false,
        ));
        pages.push(page1);

        // Page 2: sufficient text, no OCR needed
        let mut page2 = PageContent::new(2, (600.0, 800.0));
        page2.add_line(TextLine::new(
            "This is a much longer line of text that exceeds the threshold.".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 400.0, 12.0),
            false,
        ));
        pages.push(page2);

        // Page 3: has OCR text
        let mut page3 = PageContent::new(3, (600.0, 800.0));
        page3.add_line(TextLine::new(
            "OCR text".to_string(),
            12.0,
            BoundingBox::new(0.0, 0.0, 100.0, 12.0),
            true, // is_ocr = true
        ));
        pages.push(page3);

        let stats = processor.get_stats(&pages);
        assert_eq!(stats.total_pages, 3);
        assert_eq!(stats.pages_needing_ocr, 2); // page 1 and page 3
        assert_eq!(stats.pages_with_ocr, 1); // page 3
    }
}
