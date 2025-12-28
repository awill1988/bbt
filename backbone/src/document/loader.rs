use crate::document::hash::hash_file;
use crate::error::{BbtError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[cfg(feature = "pdf")]
use crate::document::pdf::PdfExtractor;

/// Supported document formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
    Pdf,
    Text,
    Markdown,
    Json,
}

impl DocumentFormat {
    /// Detect format from file extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Self> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "pdf" => Some(Self::Pdf),
                "txt" => Some(Self::Text),
                "md" | "markdown" => Some(Self::Markdown),
                "json" | "jsonc" => Some(Self::Json),
                // source code files - treat as text
                // javascript/typescript
                "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => Some(Self::Text),
                // rust
                "rs" => Some(Self::Text),
                // golang
                "go" | "mod" | "sum" => Some(Self::Text),
                // python
                "py" | "pyi" | "pyx" => Some(Self::Text),
                // ruby
                "rb" | "rake" | "gemspec" => Some(Self::Text),
                // elixir
                "ex" | "exs" => Some(Self::Text),
                // erlang
                "erl" | "hrl" => Some(Self::Text),
                // clojure
                "clj" | "cljs" | "cljc" | "edn" => Some(Self::Text),
                // scala
                "scala" | "sc" => Some(Self::Text),
                // haskell
                "hs" | "lhs" => Some(Self::Text),
                // ocaml/reason
                "ml" | "mli" | "re" | "rei" => Some(Self::Text),
                // fsharp
                "fs" | "fsi" | "fsx" => Some(Self::Text),
                // julia
                "jl" => Some(Self::Text),
                // r
                "r" | "R" | "Rmd" => Some(Self::Text),
                // dart
                "dart" => Some(Self::Text),
                // perl
                "pl" | "pm" | "perl" => Some(Self::Text),
                // lua
                "lua" => Some(Self::Text),
                // vim
                "vim" => Some(Self::Text),
                // sql
                "sql" => Some(Self::Text),
                // graphql
                "graphql" | "gql" => Some(Self::Text),
                // protobuf
                "proto" => Some(Self::Text),
                // thrift
                "thrift" => Some(Self::Text),
                // solidity
                "sol" => Some(Self::Text),
                // move (blockchain)
                "move" => Some(Self::Text),
                // zig
                "zig" => Some(Self::Text),
                // v
                "v" => Some(Self::Text),
                // d
                "d" => Some(Self::Text),
                // nim
                "nim" => Some(Self::Text),
                // crystal
                "cr" => Some(Self::Text),
                // c/c++/clang
                "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" | "inl" => Some(Self::Text),
                // assembly (note: uppercase S is preprocessed assembly, but extension matching is case-insensitive)
                "asm" | "s" => Some(Self::Text),
                // bash/shell
                "sh" | "bash" | "zsh" | "fish" => Some(Self::Text),
                // terraform/hcl
                "tf" | "tfvars" | "hcl" => Some(Self::Text),
                // ios development
                "swift" | "m" | "mm" | "xib" | "storyboard" | "plist" | "xcconfig" => Some(Self::Text),
                // android development
                "kt" | "kts" | "java" | "xml" | "gradle" | "pro" => Some(Self::Text),
                // build systems
                "cmake" | "mk" | "ninja" | "bzl" | "bazel" => Some(Self::Text),
                // visual studio / msbuild
                "vcxproj" | "sln" | "props" | "targets" | "csproj" | "vbproj" | "fsproj" => Some(Self::Text),
                // linker and compiler
                "ld" | "lds" | "pc" => Some(Self::Text),
                // template files
                "j2" | "jinja" | "jinja2" | "hbs" | "handlebars" | "mustache" |
                "tmpl" | "template" | "erb" | "ejs" | "tpl" => Some(Self::Text),
                // diagram/graph formats
                "mmd" | "mermaid" | "dot" | "gv" | "puml" | "plantuml" | "pu" | "d2" => Some(Self::Text),
                // documentation formats
                "org" | "tex" | "latex" | "adoc" | "asciidoc" | "rst" => Some(Self::Text),
                // config files
                "toml" | "yaml" | "yml" | "ini" | "conf" | "config" => Some(Self::Text),
                _ => None,
            })
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Text => "txt",
            Self::Markdown => "md",
            Self::Json => "json",
        }
    }
}

/// Metadata for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Original filename
    pub filename: String,

    /// Document format
    pub format: DocumentFormat,

    /// File size in bytes
    pub size_bytes: u64,

    /// When the document was loaded
    pub loaded_at: DateTime<Utc>,

    /// PDF-specific metadata (title, author, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pdf_metadata: Option<crate::document::pdf::PdfMetadata>,

    /// Custom metadata fields
    #[serde(default)]
    pub custom: std::collections::HashMap<String, String>,
}

/// A loaded document with content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for this document
    pub id: String,

    /// Path to the source file
    pub source_path: PathBuf,

    /// SHA256 hash of the file
    pub file_hash: String,

    /// Extracted text content
    pub content: String,

    /// Document metadata
    pub metadata: DocumentMetadata,
}

impl Document {
    /// Create a new document
    pub fn new(
        source_path: PathBuf,
        file_hash: String,
        content: String,
        metadata: DocumentMetadata,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_path,
            file_hash,
            content,
            metadata,
        }
    }

    /// Get the number of characters in the document
    pub fn char_count(&self) -> usize {
        self.content.len()
    }

    /// Get the number of lines in the document
    pub fn line_count(&self) -> usize {
        self.content.lines().count()
    }
}

/// Chunk of a document for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk
    pub id: String,

    /// ID of the parent document
    pub document_id: String,

    /// The text content of this chunk
    pub content: String,

    /// Number of tokens in this chunk (approximate)
    pub tokens: usize,

    /// Embedding vector (populated during embedding generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

impl Chunk {
    pub fn new(
        document_id: String,
        content: String,
        tokens: usize,
        metadata: ChunkMetadata,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id,
            content,
            tokens,
            embedding: None,
            metadata,
        }
    }

    /// Set the embedding for this chunk
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }
}

/// Metadata for a chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Index of this chunk in the document (0-based)
    pub chunk_index: usize,

    /// Total number of chunks in the document
    pub total_chunks: usize,

    /// Character offset in the original document
    pub char_offset: usize,

    /// Page number (for PDFs, 1-indexed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<usize>,

    /// Section heading (if detected)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_heading: Option<String>,
}

/// Document loader for multiple formats
pub struct DocumentLoader {
    #[cfg(feature = "pdf")]
    pdf_extractor: Option<PdfExtractor>,
}

impl DocumentLoader {
    /// Create a new document loader
    pub fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "pdf")]
            pdf_extractor: PdfExtractor::new().ok(),
        })
    }

    /// Load a document from a file path
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Document> {
        let path = path.as_ref();

        // Check file exists
        if !path.exists() {
            return Err(BbtError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("file not found: {}", path.display()),
            )));
        }

        // Detect format
        let format = DocumentFormat::from_path(path)
            .ok_or_else(|| BbtError::Schema(format!("unsupported file format: {}", path.display())))?;

        // Get file metadata
        let file_metadata = fs::metadata(path)?;
        let size_bytes = file_metadata.len();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Calculate file hash
        let file_hash = hash_file(path)?;

        // Load content based on format
        let (content, pdf_metadata) = match format {
            DocumentFormat::Pdf => self.load_pdf(path)?,
            DocumentFormat::Text | DocumentFormat::Markdown => (self.load_text(path)?, None),
            DocumentFormat::Json => (self.load_json(path)?, None),
        };

        let metadata = DocumentMetadata {
            filename,
            format,
            size_bytes,
            loaded_at: Utc::now(),
            pdf_metadata,
            custom: std::collections::HashMap::new(),
        };

        Ok(Document::new(
            path.to_path_buf(),
            file_hash,
            content,
            metadata,
        ))
    }

    /// Load a PDF document
    #[cfg(feature = "pdf")]
    fn load_pdf<P: AsRef<Path>>(&self, path: P) -> Result<(String, Option<crate::document::pdf::PdfMetadata>)> {
        let extractor = self.pdf_extractor.as_ref()
            .ok_or_else(|| BbtError::Model("pdf extractor not available".to_string()))?;

        let pdf_doc = extractor.extract(path)?;
        let content = pdf_doc.text();
        let metadata = pdf_doc.metadata.clone();

        Ok((content, Some(metadata)))
    }

    #[cfg(not(feature = "pdf"))]
    fn load_pdf<P: AsRef<Path>>(&self, _path: P) -> Result<(String, Option<crate::document::pdf::PdfMetadata>)> {
        Err(BbtError::Model("pdf support not enabled".to_string()))
    }

    /// Load a text file
    fn load_text<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let path = path.as_ref();

        // try to read as utf-8 first
        match fs::read_to_string(path) {
            Ok(content) => Ok(content),
            Err(e) => {
                // if utf-8 fails, try lossy conversion
                tracing::debug!("utf-8 read failed for {:?}, trying lossy conversion: {}", path, e);
                let bytes = fs::read(path)?;

                // check if file is mostly binary (>10% null bytes)
                let null_count = bytes.iter().filter(|&&b| b == 0).count();
                let null_ratio = null_count as f32 / bytes.len() as f32;
                if null_ratio > 0.1 {
                    return Err(BbtError::Schema(format!(
                        "file appears to be binary ({:.1}% null bytes)",
                        null_ratio * 100.0
                    )));
                }

                // use lossy utf-8 conversion (replaces invalid sequences with �)
                let content = String::from_utf8_lossy(&bytes).into_owned();
                tracing::debug!("loaded {:?} with lossy utf-8 conversion", path);
                Ok(content)
            }
        }
    }

    /// Load a JSON file
    fn load_json<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let path = path.as_ref();

        // read with same utf-8 handling as text files
        let content = self.load_text(path)?;

        // validate json using json5 parser (supports comments and trailing commas)
        // this allows jsonc files (like vs code configs) to be parsed
        let _: serde_json::Value = json5::from_str(&content)
            .map_err(|e| BbtError::Schema(format!("invalid json/jsonc: {}", e)))?;

        Ok(content)
    }

    /// Check if PDF support is available
    pub fn has_pdf_support(&self) -> bool {
        #[cfg(feature = "pdf")]
        {
            self.pdf_extractor.is_some()
        }

        #[cfg(not(feature = "pdf"))]
        {
            false
        }
    }
}

impl Default for DocumentLoader {
    fn default() -> Self {
        Self::new().expect("failed to initialize document loader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_detection() {
        assert_eq!(DocumentFormat::from_path("test.pdf"), Some(DocumentFormat::Pdf));
        assert_eq!(DocumentFormat::from_path("test.txt"), Some(DocumentFormat::Text));
        assert_eq!(DocumentFormat::from_path("test.md"), Some(DocumentFormat::Markdown));
        assert_eq!(DocumentFormat::from_path("test.json"), Some(DocumentFormat::Json));
        assert_eq!(DocumentFormat::from_path("test.unknown"), None);
    }

    #[test]
    fn test_load_text() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, world!").unwrap();
        writeln!(file, "This is a test.").unwrap();

        // Rename to .txt
        let txt_path = file.path().with_extension("txt");
        fs::copy(file.path(), &txt_path).unwrap();

        let loader = DocumentLoader::new().unwrap();
        let doc = loader.load(&txt_path).unwrap();

        assert_eq!(doc.metadata.format, DocumentFormat::Text);
        assert!(doc.content.contains("Hello, world!"));
        assert!(doc.content.contains("This is a test."));
        assert_eq!(doc.line_count(), 2);

        fs::remove_file(txt_path).unwrap();
    }

    #[test]
    fn test_load_json() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"key": "value", "number": 42}}"#).unwrap();

        let json_path = file.path().with_extension("json");
        fs::copy(file.path(), &json_path).unwrap();

        let loader = DocumentLoader::new().unwrap();
        let doc = loader.load(&json_path).unwrap();

        assert_eq!(doc.metadata.format, DocumentFormat::Json);
        assert!(doc.content.contains("key"));
        assert!(doc.content.contains("value"));

        fs::remove_file(json_path).unwrap();
    }

    #[test]
    fn test_load_invalid_json() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{{invalid json}}").unwrap();

        let json_path = file.path().with_extension("json");
        fs::copy(file.path(), &json_path).unwrap();

        let loader = DocumentLoader::new().unwrap();
        let result = loader.load(&json_path);

        assert!(result.is_err());
        fs::remove_file(json_path).unwrap();
    }

    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new(
            "doc-123".to_string(),
            "This is chunk content".to_string(),
            5,
            ChunkMetadata {
                chunk_index: 0,
                total_chunks: 3,
                char_offset: 0,
                page_number: Some(1),
                section_heading: Some("Introduction".to_string()),
            },
        );

        assert_eq!(chunk.document_id, "doc-123");
        assert_eq!(chunk.content, "This is chunk content");
        assert_eq!(chunk.tokens, 5);
        assert!(chunk.embedding.is_none());
        assert_eq!(chunk.metadata.chunk_index, 0);
    }

    #[test]
    fn test_chunk_set_embedding() {
        let mut chunk = Chunk::new(
            "doc-123".to_string(),
            "content".to_string(),
            1,
            ChunkMetadata {
                chunk_index: 0,
                total_chunks: 1,
                char_offset: 0,
                page_number: None,
                section_heading: None,
            },
        );

        assert!(chunk.embedding.is_none());

        chunk.set_embedding(vec![0.1, 0.2, 0.3]);
        assert!(chunk.embedding.is_some());
        assert_eq!(chunk.embedding.as_ref().unwrap().len(), 3);
    }
}
