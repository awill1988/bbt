//! Language detection from file extensions
//!
//! Provides a centralized mapping from file extensions to programming languages.
//! This module can be swapped out for a library like `hyperpolyglot` or `linguist`
//! if more sophisticated detection is needed.

use std::path::Path;

/// Programming language information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Language {
    /// Canonical language name (lowercase, no spaces)
    pub id: &'static str,
    /// Human-readable display name
    pub name: &'static str,
    /// Language category
    pub category: LanguageCategory,
}

/// Language category for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LanguageCategory {
    /// General-purpose programming languages
    Programming,
    /// Markup and documentation
    Markup,
    /// Data serialization formats
    Data,
    /// Configuration files
    Config,
    /// Shell and scripting
    Shell,
    /// Query languages
    Query,
    /// Build and infrastructure
    Build,
}

impl Language {
    const fn new(id: &'static str, name: &'static str, category: LanguageCategory) -> Self {
        Self { id, name, category }
    }

    /// Check if this is a programming language (not markup/data/config)
    pub fn is_code(&self) -> bool {
        matches!(
            self.category,
            LanguageCategory::Programming | LanguageCategory::Shell | LanguageCategory::Query
        )
    }
}

// static language definitions
mod languages {
    use super::{Language, LanguageCategory::*};

    // programming languages
    pub static RUST: Language = Language::new("rust", "Rust", Programming);
    pub static PYTHON: Language = Language::new("python", "Python", Programming);
    pub static JAVASCRIPT: Language = Language::new("javascript", "JavaScript", Programming);
    pub static TYPESCRIPT: Language = Language::new("typescript", "TypeScript", Programming);
    pub static GO: Language = Language::new("go", "Go", Programming);
    pub static JAVA: Language = Language::new("java", "Java", Programming);
    pub static KOTLIN: Language = Language::new("kotlin", "Kotlin", Programming);
    pub static SCALA: Language = Language::new("scala", "Scala", Programming);
    pub static SWIFT: Language = Language::new("swift", "Swift", Programming);
    pub static OBJECTIVE_C: Language = Language::new("objective-c", "Objective-C", Programming);
    pub static C: Language = Language::new("c", "C", Programming);
    pub static CPP: Language = Language::new("cpp", "C++", Programming);
    pub static CSHARP: Language = Language::new("csharp", "C#", Programming);
    pub static FSHARP: Language = Language::new("fsharp", "F#", Programming);
    pub static RUBY: Language = Language::new("ruby", "Ruby", Programming);
    pub static PHP: Language = Language::new("php", "PHP", Programming);
    pub static PERL: Language = Language::new("perl", "Perl", Programming);
    pub static LUA: Language = Language::new("lua", "Lua", Programming);
    pub static R: Language = Language::new("r", "R", Programming);
    pub static JULIA: Language = Language::new("julia", "Julia", Programming);
    pub static DART: Language = Language::new("dart", "Dart", Programming);
    pub static ELIXIR: Language = Language::new("elixir", "Elixir", Programming);
    pub static ERLANG: Language = Language::new("erlang", "Erlang", Programming);
    pub static HASKELL: Language = Language::new("haskell", "Haskell", Programming);
    pub static OCAML: Language = Language::new("ocaml", "OCaml", Programming);
    pub static CLOJURE: Language = Language::new("clojure", "Clojure", Programming);
    pub static ZIG: Language = Language::new("zig", "Zig", Programming);
    pub static NIM: Language = Language::new("nim", "Nim", Programming);
    pub static VLANG: Language = Language::new("v", "V", Programming);
    pub static DLANG: Language = Language::new("d", "D", Programming);
    pub static CRYSTAL: Language = Language::new("crystal", "Crystal", Programming);
    pub static SOLIDITY: Language = Language::new("solidity", "Solidity", Programming);
    pub static MOVE: Language = Language::new("move", "Move", Programming);
    pub static ASSEMBLY: Language = Language::new("assembly", "Assembly", Programming);
    pub static VIM: Language = Language::new("vim", "Vim Script", Programming);

    // shell/scripting
    pub static SHELL: Language = Language::new("shell", "Shell", Shell);
    pub static POWERSHELL: Language = Language::new("powershell", "PowerShell", Shell);

    // query languages
    pub static SQL: Language = Language::new("sql", "SQL", Query);
    pub static GRAPHQL: Language = Language::new("graphql", "GraphQL", Query);

    // markup
    pub static MARKDOWN: Language = Language::new("markdown", "Markdown", Markup);
    pub static HTML: Language = Language::new("html", "HTML", Markup);
    pub static CSS: Language = Language::new("css", "CSS", Markup);
    pub static SCSS: Language = Language::new("scss", "SCSS", Markup);
    pub static LATEX: Language = Language::new("latex", "LaTeX", Markup);
    pub static RST: Language = Language::new("rst", "reStructuredText", Markup);
    pub static ASCIIDOC: Language = Language::new("asciidoc", "AsciiDoc", Markup);
    pub static ORG: Language = Language::new("org", "Org Mode", Markup);
    pub static VUE: Language = Language::new("vue", "Vue", Markup);
    pub static SVELTE: Language = Language::new("svelte", "Svelte", Markup);
    pub static XML: Language = Language::new("xml", "XML", Markup);

    // data formats
    pub static JSON: Language = Language::new("json", "JSON", Data);
    pub static YAML: Language = Language::new("yaml", "YAML", Data);
    pub static TOML: Language = Language::new("toml", "TOML", Data);
    pub static PROTOBUF: Language = Language::new("protobuf", "Protocol Buffers", Data);
    pub static THRIFT: Language = Language::new("thrift", "Thrift", Data);

    // config/build
    pub static DOCKERFILE: Language = Language::new("dockerfile", "Dockerfile", Config);
    pub static TERRAFORM: Language = Language::new("terraform", "Terraform", Build);
    pub static NIX: Language = Language::new("nix", "Nix", Build);
    pub static CMAKE: Language = Language::new("cmake", "CMake", Build);
    pub static MAKE: Language = Language::new("make", "Makefile", Build);
    pub static GRADLE: Language = Language::new("gradle", "Gradle", Build);
    pub static BAZEL: Language = Language::new("bazel", "Bazel", Build);

    // diagrams
    pub static MERMAID: Language = Language::new("mermaid", "Mermaid", Markup);
    pub static GRAPHVIZ: Language = Language::new("graphviz", "Graphviz DOT", Markup);
    pub static PLANTUML: Language = Language::new("plantuml", "PlantUML", Markup);
    pub static D2: Language = Language::new("d2", "D2", Markup);
}

/// Detect language from a file path
///
/// Returns `Some(Language)` if the extension is recognized, `None` otherwise.
///
/// # Example
/// ```
/// use backbone::lang::from_path;
///
/// let lang = from_path("src/main.rs");
/// assert_eq!(lang.map(|l| l.id), Some("rust"));
/// ```
pub fn from_path<P: AsRef<Path>>(path: P) -> Option<&'static Language> {
    let ext = path.as_ref().extension()?.to_str()?;
    from_extension(ext)
}

/// Detect language from a file extension (without the dot)
///
/// # Example
/// ```
/// use backbone::lang::from_extension;
///
/// let lang = from_extension("rs");
/// assert_eq!(lang.map(|l| l.id), Some("rust"));
/// ```
pub fn from_extension(ext: &str) -> Option<&'static Language> {
    use languages::*;

    // normalize to lowercase for matching
    let ext_lower = ext.to_lowercase();

    match ext_lower.as_str() {
        // rust
        "rs" => Some(&RUST),

        // python
        "py" | "pyi" | "pyx" | "pyw" => Some(&PYTHON),

        // javascript
        "js" | "jsx" | "mjs" | "cjs" => Some(&JAVASCRIPT),

        // typescript
        "ts" | "tsx" | "mts" | "cts" => Some(&TYPESCRIPT),

        // go
        "go" => Some(&GO),

        // java
        "java" => Some(&JAVA),

        // kotlin
        "kt" | "kts" => Some(&KOTLIN),

        // scala
        "scala" | "sc" => Some(&SCALA),

        // swift
        "swift" => Some(&SWIFT),

        // objective-c
        "m" | "mm" => Some(&OBJECTIVE_C),

        // c
        "c" | "h" => Some(&C),

        // c++
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" | "inl" => Some(&CPP),

        // c#
        "cs" => Some(&CSHARP),

        // f#
        "fs" | "fsi" | "fsx" => Some(&FSHARP),

        // ruby
        "rb" | "rake" | "gemspec" => Some(&RUBY),

        // php
        "php" | "php3" | "php4" | "php5" | "phtml" => Some(&PHP),

        // perl
        "pl" | "pm" | "perl" => Some(&PERL),

        // lua
        "lua" => Some(&LUA),

        // r
        "r" | "rmd" => Some(&R),

        // julia
        "jl" => Some(&JULIA),

        // dart
        "dart" => Some(&DART),

        // elixir
        "ex" | "exs" => Some(&ELIXIR),

        // erlang
        "erl" | "hrl" => Some(&ERLANG),

        // haskell
        "hs" | "lhs" => Some(&HASKELL),

        // ocaml/reason
        "ml" | "mli" | "re" | "rei" => Some(&OCAML),

        // clojure
        "clj" | "cljs" | "cljc" | "edn" => Some(&CLOJURE),

        // zig
        "zig" => Some(&ZIG),

        // nim
        "nim" => Some(&NIM),

        // v
        "v" => Some(&VLANG),

        // d
        "d" => Some(&DLANG),

        // crystal
        "cr" => Some(&CRYSTAL),

        // solidity
        "sol" => Some(&SOLIDITY),

        // move
        "move" => Some(&MOVE),

        // assembly
        "asm" | "s" => Some(&ASSEMBLY),

        // vim
        "vim" => Some(&VIM),

        // shell
        "sh" | "bash" | "zsh" | "fish" => Some(&SHELL),

        // powershell
        "ps1" | "psm1" | "psd1" => Some(&POWERSHELL),

        // sql
        "sql" => Some(&SQL),

        // graphql
        "graphql" | "gql" => Some(&GRAPHQL),

        // markdown
        "md" | "markdown" => Some(&MARKDOWN),

        // html
        "html" | "htm" | "xhtml" => Some(&HTML),

        // css
        "css" => Some(&CSS),

        // scss/sass
        "scss" | "sass" => Some(&SCSS),

        // latex
        "tex" | "latex" => Some(&LATEX),

        // restructuredtext
        "rst" => Some(&RST),

        // asciidoc
        "adoc" | "asciidoc" => Some(&ASCIIDOC),

        // org mode
        "org" => Some(&ORG),

        // vue
        "vue" => Some(&VUE),

        // svelte
        "svelte" => Some(&SVELTE),

        // xml
        "xml" | "xsl" | "xslt" | "xsd" | "svg" => Some(&XML),

        // json
        "json" | "jsonc" | "json5" => Some(&JSON),

        // yaml
        "yaml" | "yml" => Some(&YAML),

        // toml
        "toml" => Some(&TOML),

        // protobuf
        "proto" => Some(&PROTOBUF),

        // thrift
        "thrift" => Some(&THRIFT),

        // dockerfile
        "dockerfile" => Some(&DOCKERFILE),

        // terraform/hcl
        "tf" | "tfvars" | "hcl" => Some(&TERRAFORM),

        // nix
        "nix" => Some(&NIX),

        // cmake
        "cmake" => Some(&CMAKE),

        // make
        "mk" | "makefile" => Some(&MAKE),

        // gradle
        "gradle" => Some(&GRADLE),

        // bazel
        "bzl" | "bazel" => Some(&BAZEL),

        // mermaid
        "mmd" | "mermaid" => Some(&MERMAID),

        // graphviz
        "dot" | "gv" => Some(&GRAPHVIZ),

        // plantuml
        "puml" | "plantuml" | "pu" => Some(&PLANTUML),

        // d2
        "d2" => Some(&D2),

        _ => None,
    }
}

/// Check if a file extension represents source code (not markup/data/config)
pub fn is_code_extension(ext: &str) -> bool {
    from_extension(ext).map(|l| l.is_code()).unwrap_or(false)
}

/// Check if a file path represents source code
pub fn is_code_path<P: AsRef<Path>>(path: P) -> bool {
    from_path(path).map(|l| l.is_code()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_extension_programming() {
        assert_eq!(from_extension("rs").map(|l| l.id), Some("rust"));
        assert_eq!(from_extension("py").map(|l| l.id), Some("python"));
        assert_eq!(from_extension("js").map(|l| l.id), Some("javascript"));
        assert_eq!(from_extension("ts").map(|l| l.id), Some("typescript"));
        assert_eq!(from_extension("go").map(|l| l.id), Some("go"));
        assert_eq!(from_extension("java").map(|l| l.id), Some("java"));
        assert_eq!(from_extension("kt").map(|l| l.id), Some("kotlin"));
        assert_eq!(from_extension("swift").map(|l| l.id), Some("swift"));
        assert_eq!(from_extension("cpp").map(|l| l.id), Some("cpp"));
        assert_eq!(from_extension("cs").map(|l| l.id), Some("csharp"));
    }

    #[test]
    fn test_from_extension_case_insensitive() {
        assert_eq!(from_extension("RS").map(|l| l.id), Some("rust"));
        assert_eq!(from_extension("Py").map(|l| l.id), Some("python"));
        assert_eq!(from_extension("JS").map(|l| l.id), Some("javascript"));
    }

    #[test]
    fn test_from_extension_markup() {
        assert_eq!(from_extension("md").map(|l| l.id), Some("markdown"));
        assert_eq!(from_extension("html").map(|l| l.id), Some("html"));
        assert_eq!(from_extension("css").map(|l| l.id), Some("css"));
    }

    #[test]
    fn test_from_extension_data() {
        assert_eq!(from_extension("json").map(|l| l.id), Some("json"));
        assert_eq!(from_extension("yaml").map(|l| l.id), Some("yaml"));
        assert_eq!(from_extension("toml").map(|l| l.id), Some("toml"));
    }

    #[test]
    fn test_from_path() {
        assert_eq!(from_path("src/main.rs").map(|l| l.id), Some("rust"));
        assert_eq!(from_path("app/index.tsx").map(|l| l.id), Some("typescript"));
        assert_eq!(from_path("docs/README.md").map(|l| l.id), Some("markdown"));
    }

    #[test]
    fn test_from_path_unknown() {
        assert!(from_path("no_extension").is_none());
        assert!(from_path("file.xyz").is_none());
    }

    #[test]
    fn test_is_code() {
        assert!(from_extension("rs").unwrap().is_code());
        assert!(from_extension("py").unwrap().is_code());
        assert!(from_extension("sql").unwrap().is_code());
        assert!(from_extension("sh").unwrap().is_code());

        assert!(!from_extension("md").unwrap().is_code());
        assert!(!from_extension("json").unwrap().is_code());
        assert!(!from_extension("yaml").unwrap().is_code());
    }

    #[test]
    fn test_is_code_extension() {
        assert!(is_code_extension("rs"));
        assert!(is_code_extension("py"));
        assert!(!is_code_extension("md"));
        assert!(!is_code_extension("xyz"));
    }

    #[test]
    fn test_language_display_name() {
        assert_eq!(from_extension("rs").unwrap().name, "Rust");
        assert_eq!(from_extension("cpp").unwrap().name, "C++");
        assert_eq!(from_extension("cs").unwrap().name, "C#");
        assert_eq!(from_extension("ts").unwrap().name, "TypeScript");
    }

    #[test]
    fn test_language_category() {
        assert_eq!(
            from_extension("rs").unwrap().category,
            LanguageCategory::Programming
        );
        assert_eq!(
            from_extension("sh").unwrap().category,
            LanguageCategory::Shell
        );
        assert_eq!(
            from_extension("sql").unwrap().category,
            LanguageCategory::Query
        );
        assert_eq!(
            from_extension("md").unwrap().category,
            LanguageCategory::Markup
        );
        assert_eq!(
            from_extension("json").unwrap().category,
            LanguageCategory::Data
        );
    }
}
