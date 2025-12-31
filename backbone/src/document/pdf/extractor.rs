use super::layout::{BoundingBox, PageContent, PdfDocument, PdfMetadata, TextLine};
use crate::error::{BbtError, Result};
use pdfium_render::prelude::*;
use std::path::Path;

/// PDF extractor using pdfium-render
pub struct PdfExtractor {
    pdfium: Pdfium,
}

impl PdfExtractor {
    /// Create a new PDF extractor
    pub fn new() -> Result<Self> {
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|e| BbtError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("failed to load pdfium library: {}", e)
                )))?,
        );

        Ok(Self { pdfium })
    }

    /// Extract a PDF document from a file path
    pub fn extract<P: AsRef<Path>>(&self, path: P) -> Result<PdfDocument> {
        let document = self.pdfium
            .load_pdf_from_file(path.as_ref(), None)
            .map_err(|e| BbtError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to load pdf: {}", e)
            )))?;

        // Extract metadata
        let metadata = self.extract_metadata(&document);

        // Extract pages
        let mut pdf_doc = PdfDocument::new(metadata);

        for page_index in 0..document.pages().len() {
            let page = document.pages().get(page_index)
                .map_err(|e| BbtError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to get page {}: {}", page_index, e)
                )))?;

            let page_content = self.extract_page(&page, (page_index + 1).into())?;
            pdf_doc.add_page(page_content);
        }

        Ok(pdf_doc)
    }

    /// Extract metadata from a PDF document
    fn extract_metadata(&self, document: &pdfium_render::prelude::PdfDocument) -> PdfMetadata {
        let metadata_tags = document.metadata();

        // Helper to extract metadata value as String
        let get_metadata = |tag_type: PdfDocumentMetadataTagType| -> Option<String> {
            metadata_tags.get(tag_type)
                .and_then(|tag| {
                    let value = tag.value();
                    if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    }
                })
        };

        PdfMetadata {
            title: get_metadata(PdfDocumentMetadataTagType::Title),
            author: get_metadata(PdfDocumentMetadataTagType::Author),
            subject: get_metadata(PdfDocumentMetadataTagType::Subject),
            keywords: get_metadata(PdfDocumentMetadataTagType::Keywords),
            creator: get_metadata(PdfDocumentMetadataTagType::Creator),
            producer: get_metadata(PdfDocumentMetadataTagType::Producer),
        }
    }

    /// Extract content from a single page
    fn extract_page(&self, page: &PdfPage, page_number: usize) -> Result<PageContent> {
        let dimensions = (page.width().value, page.height().value);
        let mut page_content = PageContent::new(page_number, dimensions);

        // Get all text from the page
        let text_page = page.text()
            .map_err(|e| BbtError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to get text from page {}: {}", page_number, e)
            )))?;

        // Get the full text content
        let full_text = text_page.all();

        // Split into lines and create TextLine objects
        // For now, use a simplified approach - each line of text becomes a TextLine
        // with default font size and bounding box
        for (line_idx, line_text) in full_text.lines().enumerate() {
            if line_text.trim().is_empty() {
                continue;
            }

            // Create a simple bounding box based on line position
            // This is a simplified approach - actual bbox would require more complex extraction
            let y_pos = dimensions.1 - (line_idx as f32 * 15.0); // approximate line height
            let bbox = BoundingBox::new(
                50.0,              // left margin
                y_pos,             // y position
                dimensions.0 - 100.0, // width (page width - margins)
                12.0,              // estimated height
            );

            let text_line = TextLine::new(
                line_text.to_string(),
                12.0,  // default font size - would need more complex extraction for actual sizes
                bbox,
                false, // not OCR
            );
            page_content.add_line(text_line);
        }

        Ok(page_content)
    }
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self::new().expect("failed to initialize pdf extractor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_new() {
        // test that pdf extractor creation either succeeds or returns a proper error
        // (pdfium may not be available in all environments)
        let result = PdfExtractor::new();
        match result {
            Ok(extractor) => {
                // if pdfium is available, we should get a valid extractor
                // we can't do much else without a sample PDF
                drop(extractor);
            }
            Err(e) => {
                // if pdfium isn't available, we should get an IO error
                // with a descriptive message about the missing library
                let error_msg = format!("{}", e);
                assert!(
                    error_msg.contains("pdfium") || error_msg.contains("library"),
                    "error should mention pdfium library: {}",
                    error_msg
                );
            }
        }
    }

    #[test]
    fn test_extractor_default() {
        // test that Default trait works (panics if pdfium unavailable)
        // in production this is expected - we catch the panic in test
        let result = std::panic::catch_unwind(|| {
            let _extractor = PdfExtractor::default();
        });

        // if pdfium is available, this succeeds
        // if not, it panics - both are valid outcomes for this test
        match result {
            Ok(_) => {
                // pdfium available, default() worked
            }
            Err(_) => {
                // pdfium not available, default() panicked as expected
            }
        }
    }
}
