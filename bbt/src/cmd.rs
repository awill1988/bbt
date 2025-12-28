use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "bbt")]
#[command(about = "big brain time - document ingestion and RAG", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate database schema from JSON payload
    Gen {
        #[command(subcommand)]
        subcommand: GenCommands,
    },
    /// Sync documents from a directory
    Sync(SyncCommand),
    /// Query indexed documents
    Query(QueryCommand),
}

struct QueryResultOutput {
    id: String,
    score: f32,
    text: String,
    citation: String,
    source: String,
    metadata: std::collections::HashMap<String, String>,
}

#[derive(Args)]
pub struct SyncCommand {
    /// Paths to directories to sync
    #[clap(value_parser, required = true)]
    paths: Vec<PathBuf>,

    /// File extensions to include (use 'code' for source code preset)
    #[clap(long, value_delimiter = ',', default_value = "code")]
    ext: Vec<String>,

    /// Only scan git repositories
    #[clap(long, default_value = "true")]
    git_only: bool,

    /// Honor .gitignore files
    #[clap(long, default_value = "true")]
    honor_gitignore: bool,

    /// Force reprocessing even if unchanged
    #[clap(long, env = "BBT_FORCE_SYNC", default_value = "false")]
    force: bool,

    /// Reset the processing state before syncing
    #[clap(long, env = "BBT_RESET_STATE", default_value = "false")]
    reset_state: bool,

    /// Index git commits in addition to documents
    #[clap(long, default_value = "false")]
    commits: bool,

    /// Maximum number of commits to index per repository
    #[clap(long, default_value = "1000")]
    max_commits: usize,

    /// Only index commits since this date (ISO 8601 format: YYYY-MM-DD)
    #[clap(long)]
    commits_since: Option<String>,
}

#[derive(Args)]
pub struct QueryCommand {
    /// Query text
    #[clap(long)]
    query: String,

    /// Override top k results
    #[clap(long)]
    top_k: Option<usize>,

    /// Retrieval mode override
    #[clap(long, value_enum)]
    mode: Option<QueryMode>,

    /// Filter by source path prefix
    #[clap(long)]
    source_prefix: Option<String>,

    /// Citation redaction mode
    #[clap(long, value_enum, default_value = "full")]
    citation_mode: CitationMode,

    /// Output format
    #[clap(long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Wrap width for text output
    #[clap(long, default_value_t = 100)]
    wrap_width: usize,

    /// Show scores in text output
    #[clap(long, default_value = "false")]
    show_scores: bool,

    /// Query commits instead of documents
    #[clap(long, default_value = "false")]
    commits: bool,

    /// Filter by commit classification (feat, fix, refactor, docs, test, chore, style, perf, ci, build, revert)
    #[clap(long)]
    commit_type: Option<String>,

    /// Filter by author email
    #[clap(long)]
    author: Option<String>,

    /// Filter commits since this date (ISO 8601: YYYY-MM-DD)
    #[clap(long)]
    since: Option<String>,

    /// Filter commits until this date (ISO 8601: YYYY-MM-DD)
    #[clap(long)]
    until: Option<String>,

    /// Recency weight for hybrid scoring (0.0 = pure semantic, 1.0 = pure recency)
    #[clap(long, default_value = "0.0")]
    recency_weight: f32,

    /// Time decay function for recency scoring
    #[clap(long, value_enum, default_value = "exponential")]
    time_decay: TimeDecay,

    /// Sort order for results
    #[clap(long, value_enum, default_value = "score")]
    sort_by: SortBy,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SortBy {
    /// Sort by relevance score (default)
    Score,
    /// Sort by date, oldest first
    DateAsc,
    /// Sort by date, newest first
    DateDesc,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum TimeDecay {
    /// Sharp dropoff: e^(-0.1 * days)
    Exponential,
    /// Gradual: 1 - (days / 365)
    Linear,
    /// Smooth: 1 / (1 + log(days + 1))
    Logarithmic,
}

impl TimeDecay {
    fn calculate_score(self, days_old: f64) -> f32 {
        match self {
            // exponential: e^(-λ * days) where λ = 0.01 (half-life ~70 days)
            Self::Exponential => (-0.01 * days_old).exp() as f32,
            // linear: 1 - (days / 365), clamped to [0, 1]
            Self::Linear => (1.0 - (days_old / 365.0)).max(0.0) as f32,
            // logarithmic: 1 / (1 + log(days + 1))
            Self::Logarithmic => (1.0 / (1.0 + (days_old + 1.0).ln())) as f32,
        }
    }
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum QueryMode {
    Vector,
    Bm25,
    Hybrid,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum CitationMode {
    Full,
    Basename,
    Hash,
    Redact,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum OutputFormat {
    Text,
    Json,
}

impl From<QueryMode> for backbone::config::RetrievalMode {
    fn from(value: QueryMode) -> Self {
        match value {
            QueryMode::Vector => backbone::config::RetrievalMode::Vector,
            QueryMode::Bm25 => backbone::config::RetrievalMode::Bm25,
            QueryMode::Hybrid => backbone::config::RetrievalMode::Hybrid,
        }
    }
}

#[derive(Subcommand)]
enum GenCommands {
    /// Generate SQL schema from JSON payload
    Schema {
        /// Input JSON file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output SQL file path
        #[arg(short, long)]
        output: PathBuf,

        /// JSON Schema output path (default: <input>.schema.json)
        #[arg(long)]
        json_schema_output: Option<PathBuf>,

        /// Model repository ID
        #[arg(long, env = "BBT_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// Model filename
        #[arg(long, env = "BBT_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "BBT_MODEL_CACHE_DIR")]
        cache_dir: Option<PathBuf>,
    },
    /// Generate JSON Schema from JSON payload
    JsonSchema {
        /// Input JSON file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output JSON Schema file path
        #[arg(short, long)]
        output: PathBuf,

        /// Model repository ID
        #[arg(long, env = "BBT_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// Model filename
        #[arg(long, env = "BBT_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "BBT_MODEL_CACHE_DIR")]
        cache_dir: Option<PathBuf>,
    },
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Gen { subcommand } => match subcommand {
            GenCommands::Schema {
                input,
                output,
                json_schema_output,
                repo_id,
                filename,
                cache_dir,
            } => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(generate_schema(
                    input,
                    output,
                    json_schema_output,
                    repo_id,
                    filename,
                    cache_dir,
                ))
            }
            GenCommands::JsonSchema {
                input,
                output,
                repo_id,
                filename,
                cache_dir,
            } => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(generate_json_schema(
                    input, output, repo_id, filename, cache_dir,
                ))
            }
        },
        Commands::Sync(args) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(sync_documents(args))
        }
        Commands::Query(args) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(query_documents(args))
        }
    }
}

/// expand extension presets into actual file extensions
fn expand_extensions(ext_list: &[String]) -> Vec<String> {
    let mut extensions: Vec<String> = Vec::new();

    for ext in ext_list {
        match ext.to_lowercase().as_str() {
            "code" => {
                // source code preset: comprehensive language support
                let code_exts = [
                    // javascript/typescript
                    "js", "jsx", "ts", "tsx", "mjs", "cjs",
                    // rust
                    "rs",
                    // golang
                    "go", "mod", "sum",
                    // python
                    "py", "pyi", "pyx",
                    // c/c++/clang
                    "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "inl",
                    // c# / f# / vb
                    "cs", "fs", "fsx", "fsi", "vb",
                    // assembly
                    "asm", "s", "S",
                    // bash/shell
                    "sh", "bash", "zsh", "fish",
                    // nix
                    "nix",
                    // terraform/hcl
                    "tf", "tfvars", "hcl",
                    // ios development
                    "swift", "m", "mm", "xib", "storyboard", "plist", "xcconfig",
                    // android development
                    "kt", "kts", "java", "xml", "gradle", "pro",
                    // jvm languages
                    "scala", "sc", "clj", "cljs", "cljc", "edn",
                    // functional languages
                    "hs", "lhs", "ml", "mli", "re", "rei", "ex", "exs", "erl", "hrl",
                    // web languages
                    "php", "rb", "rake", "gemspec",
                    // scripting
                    "lua", "pl", "pm", "perl",
                    // scientific/data
                    "jl", "r", "R", "Rmd",
                    // systems languages
                    "zig", "v", "dart",
                    // blockchain
                    "sol", "move",
                    // build systems
                    "cmake", "mk", "ninja", "bzl", "bazel",
                    // visual studio / msbuild
                    "vcxproj", "sln", "props", "targets", "csproj", "vbproj", "fsproj",
                    // linker and compiler
                    "ld", "lds", "pc",
                    // protocol/interface definition
                    "proto", "thrift", "graphql", "gql",
                    // config languages
                    "dhall",
                    // database
                    "sql",
                    // template files
                    "j2", "jinja", "jinja2", "hbs", "handlebars", "mustache",
                    "tmpl", "template", "erb", "ejs", "tpl",
                    // diagram/graph formats
                    "mmd", "mermaid", "dot", "gv", "puml", "plantuml", "pu", "d2",
                    // documentation formats
                    "org", "tex", "latex", "adoc", "asciidoc", "rst",
                    // common config/doc files
                    "toml", "yaml", "yml", "json", "jsonc", "md", "txt", "ini", "conf", "config",
                    // container/virtualization
                    "dockerfile", "containerfile", "vagrantfile",
                    // build scripts
                    "makefile", "gnumakefile", "justfile",
                ];
                extensions.extend(code_exts.iter().map(|s| s.to_string()));
            }
            "docs" => {
                let doc_exts = ["md", "txt", "rst", "adoc"];
                extensions.extend(doc_exts.iter().map(|s| s.to_string()));
            }
            _ => {
                // treat as literal extension
                extensions.push(ext.to_lowercase());
            }
        }
    }

    extensions
}

async fn sync_documents(args: SyncCommand) -> Result<()> {
    use backbone::config::BbtConfig;
    use backbone::document::loader::DocumentLoader;
    use backbone::document::text::chunking::chunk_text;
    use backbone::embedding::{ensure_onnx_model, EmbeddingModelInfo, ExecutionProvider, OnnxEmbedder};
    use backbone::retrieval::bm25::{load_index, save_index};
    use backbone::storage::StateStore;
    use backbone::tracing::init_tracing;
    use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
    use console::Term;
    use qdrant_client::Qdrant;
    use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Value};
    use std::fs;
    use std::collections::HashMap;
    use uuid::Uuid;

    eprintln!("debug: loading config");
    let config = BbtConfig::from_env()?;
    eprintln!("debug: config loaded");

    eprintln!("debug: creating document loader");
    let loader = DocumentLoader::new()?;
    eprintln!("debug: loader created");
    let state_path = config.state_store_path.clone();
    eprintln!("debug: checking reset_state flag");
    if args.reset_state {
        if state_path.exists() {
            tracing::info!(
                "resetting state store at {}",
                state_path.to_string_lossy()
            );
            fs::remove_file(&state_path)?;
        }
        let wal_path = PathBuf::from(format!("{}-wal", state_path.to_string_lossy()));
        let shm_path = PathBuf::from(format!("{}-shm", state_path.to_string_lossy()));
        if wal_path.exists() {
            let _ = fs::remove_file(wal_path);
        }
        if shm_path.exists() {
            let _ = fs::remove_file(shm_path);
        }
    }
    eprintln!("debug: creating state store at {:?}", state_path);
    let mut state_store = StateStore::new(&state_path)?;
    eprintln!("debug: state store created");
    eprintln!("debug: finalizing incomplete");
    let incomplete = state_store.finalize_incomplete("interrupted")?;
    if incomplete > 0 {
        tracing::info!("marked {} in-progress files as failed", incomplete);
    }
    eprintln!("debug: expanding extensions");
    let extensions = expand_extensions(&args.ext);
    eprintln!("debug: extensions expanded: {} total", extensions.len());

    eprintln!("debug: creating term");
    let term = Term::stderr();
    let term_width = usize::from(term.size().1);
    eprintln!("debug: creating multi progress");
    let progress = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
    progress.set_move_cursor(true);
    eprintln!("debug: creating scan_bar");
    let scan_bar = progress.add(ProgressBar::new_spinner());
    scan_bar.set_style(
        ProgressStyle::with_template("{prefix} {spinner} {msg}")?
            .tick_chars("|/-\\"),
    );
    scan_bar.set_prefix("scan");
    scan_bar.enable_steady_tick(std::time::Duration::from_millis(120));

    eprintln!("debug: creating log_bar");
    let log_bar = progress.add(ProgressBar::new(0));
    log_bar.set_style(ProgressStyle::with_template("{prefix} {msg}")?);
    log_bar.set_prefix("log");
    log_bar.set_message("ready");

    eprintln!("debug: creating log channel and thread");
    let log_message_width = log_message_width(term_width, log_bar.prefix().len());
    let (log_tx, log_rx) = std::sync::mpsc::channel::<String>();
    let log_bar_handle = log_bar.clone();
    let log_thread = std::thread::spawn(move || {
        while let Ok(line) = log_rx.recv() {
            if line == "__bbt_log_close__" {
                break;
            }
            let message = truncate_log_line(&line, log_message_width);
            log_bar_handle.set_message(message);
        }
    });

    eprintln!("debug: initializing tracing");
    let _guard = init_tracing("bbt", Some(log_tx.clone()))?;
    eprintln!("debug: tracing initialized");

    eprintln!("debug: building scan scope key");
    let scan_scope_key = build_scan_scope_key(&args, &extensions);
    eprintln!("debug: scan scope key: {}", scan_scope_key);
    let scan_message_width = scan_message_width(term_width, scan_bar.prefix().len());

    eprintln!("debug: getting scan run from state store");
    let scan_run = state_store.get_scan_run(&scan_scope_key)?;
    eprintln!("debug: got scan run: {:?}", scan_run.is_some());
    let (candidates, scan_totals, scan_run_id) = if let Some(run) = scan_run {
        eprintln!("debug: scan run exists, checking status");
        if run.status == backbone::storage::ScanStatus::Complete {
            eprintln!("debug: scan status is complete, getting summary");
            let summary = state_store.scan_entries_summary(&scan_scope_key, &run.run_id)?;
            eprintln!("debug: summary: files_seen={}, run.files_seen={}", summary.files_seen, run.files_seen);
            if summary.files_seen == 0 && run.files_seen > 0 {
                eprintln!("debug: cache empty, re-scanning");
                tracing::warn!("scan cache empty; re-scanning");
                let scan_progress = state_store.start_scan(&scan_scope_key)?;
                let (candidates, scan_totals) = collect_candidates(
                    &args,
                    &extensions,
                    &scan_bar,
                    &mut state_store,
                    &scan_scope_key,
                    &scan_progress,
                    scan_message_width,
                )?;
                state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
                scan_bar.finish_and_clear();
                progress.remove(&scan_bar);
                tracing::info!(
                    "scan complete: {} files, {} bytes",
                    scan_totals.files,
                    scan_totals.bytes
                );
                (candidates, scan_totals, scan_progress.run_id)
            } else {
                eprintln!("debug: using cached scan");
                if summary.files_seen != run.files_seen
                    || summary.bytes_seen != run.bytes_seen
                {
                    state_store.update_scan_run_counts(
                        &scan_scope_key,
                        &run.run_id,
                        summary.files_seen,
                        summary.bytes_seen,
                    )?;
                    tracing::info!(
                        "scan totals reconciled from cached entries ({} files, {} bytes)",
                        summary.files_seen,
                        summary.bytes_seen
                    );
                }
                eprintln!("debug: setting scan_bar message");
                scan_bar.set_message(format!(
                    "{} files, {} bytes",
                    summary.files_seen, summary.bytes_seen
                ));
                tracing::info!(
                    "using cached scan with {} files and {} bytes",
                    summary.files_seen,
                    summary.bytes_seen
                );
                eprintln!("debug: listing scan entries");
                let entries = state_store.list_scan_entries(&scan_scope_key, &run.run_id)?;
                eprintln!("debug: got {} entries", entries.len());
                if summary.files_seen > 0 && entries.is_empty() {
                    tracing::warn!("scan cache empty; re-scanning");
                    let scan_progress = state_store.start_scan(&scan_scope_key)?;
                    let (candidates, scan_totals) = collect_candidates(
                        &args,
                        &extensions,
                        &scan_bar,
                        &mut state_store,
                        &scan_scope_key,
                        &scan_progress,
                        scan_message_width,
                    )?;
                    state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
                    scan_bar.finish_and_clear();
                    progress.remove(&scan_bar);
                    tracing::info!(
                        "scan complete: {} files, {} bytes",
                        scan_totals.files,
                        scan_totals.bytes
                    );
                    (candidates, scan_totals, scan_progress.run_id)
                } else {
                    eprintln!("debug: mapping entries to candidates");
                    let candidates = entries
                        .into_iter()
                        .map(|entry| FileCandidate {
                            path: PathBuf::from(entry.source_path),
                            size_bytes: entry.file_size_bytes,
                            modified_at: entry.modified_at,
                        })
                        .collect::<Vec<_>>();
                    eprintln!("debug: mapped {} candidates", candidates.len());
                    eprintln!("debug: finishing scan_bar");
                    scan_bar.finish_and_clear();
                    eprintln!("debug: removing scan_bar from progress");
                    progress.remove(&scan_bar);
                    eprintln!("debug: logging scan cached");
                    tracing::info!(
                        "scan cached: {} files, {} bytes",
                        summary.files_seen,
                        summary.bytes_seen
                    );
                    eprintln!("debug: returning candidates tuple");
                    (
                        candidates,
                        ScanTotals {
                            files: summary.files_seen,
                            bytes: summary.bytes_seen,
                        },
                        run.run_id.clone(),
                    )
                }
            }
        } else {
            let scan_progress = state_store.start_scan(&scan_scope_key)?;
            scan_bar.set_message(format!(
                "{} files, {} bytes",
                scan_progress.files_seen, scan_progress.bytes_seen
            ));
            if scan_progress.files_seen > 0 {
                tracing::info!(
                    "resuming scan with {} files and {} bytes already counted",
                    scan_progress.files_seen,
                    scan_progress.bytes_seen
                );
            }
            let (candidates, scan_totals) = collect_candidates(
                &args,
                &extensions,
                &scan_bar,
                &mut state_store,
                &scan_scope_key,
                &scan_progress,
                scan_message_width,
            )?;
            state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
            scan_bar.finish_and_clear();
            progress.remove(&scan_bar);
            tracing::info!(
                "scan complete: {} files, {} bytes",
                scan_totals.files,
                scan_totals.bytes
            );
            (candidates, scan_totals, scan_progress.run_id)
        }
    } else {
        let scan_progress = state_store.start_scan(&scan_scope_key)?;
        scan_bar.set_message(format!(
            "{} files, {} bytes",
            scan_progress.files_seen, scan_progress.bytes_seen
        ));
        let (candidates, scan_totals) = collect_candidates(
            &args,
            &extensions,
            &scan_bar,
            &mut state_store,
            &scan_scope_key,
            &scan_progress,
            scan_message_width,
        )?;
        state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
        scan_bar.finish_and_clear();
        progress.remove(&scan_bar);
        tracing::info!(
            "scan complete: {} files, {} bytes",
            scan_totals.files,
            scan_totals.bytes
        );
        (candidates, scan_totals, scan_progress.run_id)
    };

    eprintln!("debug: scan complete, got {} candidates", candidates.len());
    let total_files = scan_totals.files;
    let total_bytes = scan_totals.bytes;
    eprintln!("debug: creating files_bar with {} files", total_files);
    let files_bar = progress.insert_before(&log_bar, ProgressBar::new(total_files));
    eprintln!("debug: creating bytes_bar with {} bytes", total_bytes);
    let bytes_bar = progress.insert_before(&log_bar, ProgressBar::new(total_bytes));

    let files_style = ProgressStyle::with_template(
        "{prefix} [{bar:40.cyan/blue}] {pos}/{len} {msg}",
    )?;
    let bytes_style = ProgressStyle::with_template(
        "{prefix} [{bar:40.green/green}] {bytes}/{total_bytes} {msg}",
    )?;

    files_bar.set_style(files_style);
    bytes_bar.set_style(bytes_style);
    files_bar.set_prefix("files");
    bytes_bar.set_prefix("bytes");
    let files_count_width = total_files.to_string().len();
    let bytes_count_width = format!("{}", HumanBytes(total_bytes)).len();
    let files_message_width = progress_message_width(
        term_width,
        files_bar.prefix().len(),
        40,
        files_count_width,
        files_count_width,
    );
    let bytes_message_width = progress_message_width(
        term_width,
        bytes_bar.prefix().len(),
        40,
        bytes_count_width,
        bytes_count_width,
    );

    let mut resumed_files = 0_u64;
    let mut resumed_bytes = 0_u64;
    if !args.force {
        let summary =
            state_store.processed_summary_for_scan(&scan_scope_key, &scan_run_id)?;
        resumed_files = summary.files_seen.min(total_files);
        resumed_bytes = summary.bytes_seen.min(total_bytes);
    }

    if resumed_files > 0 {
        tracing::info!(
            "resuming with {} files and {} bytes already processed",
            resumed_files,
            resumed_bytes
        );
    }

    eprintln!("debug: setting progress bar positions");
    files_bar.set_position(resumed_files);
    bytes_bar.set_position(resumed_bytes);

    eprintln!("debug: creating model_info");
    let model_info = EmbeddingModelInfo::default_model();
    let vector_dimensions = model_info.dimensions;

    // ensure model is downloaded
    eprintln!("debug: ensuring onnx model is downloaded");
    let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
    eprintln!("debug: model path: {:?}", model_path);
    eprintln!("debug: creating embedder");
    let mut embedder = OnnxEmbedder::new(&model_path, model_info, ExecutionProvider::detect())?;
    eprintln!("debug: embedder created");

    // load or create bm25 index
    eprintln!("debug: loading bm25 index");
    tracing::info!("loading bm25 index from {:?}", config.bm25_index_path);
    let mut bm25_index = load_index(&config.bm25_index_path)?;
    tracing::info!("bm25 index loaded with {} documents", bm25_index.num_docs());
    eprintln!("debug: bm25 index loaded");

    // use rest client instead of grpc
    eprintln!("debug: creating qdrant client, url={}", config.qdrant_url);
    let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
        eprintln!("debug: using api key auth");
        Qdrant::from_url(&config.qdrant_url)
            .api_key(api_key.clone())
            .timeout(std::time::Duration::from_secs(60))
            .build()?
    } else {
        eprintln!("debug: no api key, using simple auth");
        Qdrant::from_url(&config.qdrant_url)
            .timeout(std::time::Duration::from_secs(60))
            .build()?
    };
    eprintln!("debug: qdrant client created");

    let collection_name = "bbt".to_string();

    // note: health_check() hangs with grpc client, skipping explicit health check
    // connection will be verified when we actually use the client
    eprintln!("debug: skipping health check (grpc client health_check hangs)");
    tracing::info!("connecting to qdrant at {}", config.qdrant_url);

    // check if collection exists, create if needed
    eprintln!("debug: checking if collection exists");
    use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
    let collection_exists = qdrant_client
        .collection_exists(&collection_name)
        .await
        .unwrap_or(false);
    eprintln!("debug: collection_exists={}", collection_exists);

    if !collection_exists {
        eprintln!("debug: creating collection");
        tracing::info!("creating collection '{}'", collection_name);
        let vector_params = VectorParamsBuilder::new(vector_dimensions as u64, Distance::Cosine).build();
        qdrant_client
            .create_collection(CreateCollectionBuilder::new(&collection_name).vectors_config(vector_params))
            .await?;
        tracing::info!("collection '{}' created", collection_name);
        eprintln!("debug: collection created");
    } else {
        eprintln!("debug: using existing collection");
        tracing::info!("using existing collection '{}'", collection_name);
    }

    eprintln!("debug: logging sync start");
    tracing::info!(
        "starting sync for paths: {:?}, with extensions: {:?}, git_only={}, honor_gitignore={}",
        args.paths,
        extensions,
        args.git_only,
        args.honor_gitignore
    );
    eprintln!("debug: starting main sync loop");

    for candidate in candidates {
        let filename = candidate
            .path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = candidate.path.as_path();
        let file_size_bytes = candidate.size_bytes;
        let modified_at = candidate.modified_at;
        let source_path = file_path.to_string_lossy().to_string();

        // detect git repository for display
        let repo_name = find_git_repo(file_path)
            .map(|root| get_repo_name(&root))
            .unwrap_or_else(|| "no-repo".to_string());
        let display_name = format!("{}/{}", repo_name, filename);

        // skip empty files silently (e.g., __init__.py)
        if file_size_bytes == 0 {
            tracing::debug!("skipping empty file {:?}", file_path);
            files_bar.inc(1);
            // note: not incrementing bytes_bar since size is 0
            continue;
        }

        let skip_existing = !args.force
            && (state_store.should_skip(
                &source_path,
                file_size_bytes,
                modified_at,
            ) || state_store.should_skip_failure(
                &source_path,
                file_size_bytes,
                modified_at,
            ));
        if skip_existing {
            tracing::debug!("skipping unchanged file {:?}", file_path);
            continue;
        }

        tracing::debug!("processing file: {:?}", file_path);

        let doc = match loader.load(file_path) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("failed to load file {:?}: {}", file_path, e);
                let _ = state_store.record_failure(
                    &source_path,
                    None,
                    file_size_bytes,
                    modified_at,
                    format!("load failed: {e}"),
                );
                let _ = state_store.mark_failed(&source_path, e.to_string());
                let files_message = right_align_truncate(&display_name, files_message_width);
                let bytes_message = right_align_truncate(&display_name, bytes_message_width);
                files_bar.set_message(files_message);
                bytes_bar.set_message(bytes_message);
                files_bar.inc(1);
                bytes_bar.inc(candidate.size_bytes);
                continue;
            }
        };

        if let Err(e) = state_store.mark_processing(
            doc.file_hash.clone(),
            source_path.clone(),
            file_size_bytes,
            modified_at,
        ) {
            tracing::warn!("failed to update state for {:?}: {}", file_path, e);
        }

        let chunks = match chunk_text(&doc.content, config.chunk_size, config.chunk_overlap) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("failed to chunk file {:?}: {}", file_path, e);
                let _ = state_store.record_failure(
                    &source_path,
                    Some(doc.file_hash.clone()),
                    file_size_bytes,
                    modified_at,
                    format!("chunk failed: {e}"),
                );
                let _ = state_store.mark_failed(&source_path, e.to_string());
                let files_message = right_align_truncate(&display_name, files_message_width);
                let bytes_message = right_align_truncate(&display_name, bytes_message_width);
                files_bar.set_message(files_message);
                bytes_bar.set_message(bytes_message);
                files_bar.inc(1);
                bytes_bar.inc(candidate.size_bytes);
                continue;
            }
        };

        if chunks.is_empty() {
            // files with only whitespace or very small content produce no chunks
            // this is not an error, just skip them silently
            tracing::debug!(
                "skipping file {:?} (no chunks produced - likely only whitespace)",
                file_path
            );
            files_bar.inc(1);
            bytes_bar.inc(candidate.size_bytes);
            continue;
        }

        let embeddings = match embedder.embed(&chunks) {
            Ok(em) => em,
            Err(e) => {
                tracing::error!(
                    "failed to embed chunks for {:?}: {}",
                    file_path,
                    e
                );
                let _ = state_store.record_failure(
                    &source_path,
                    Some(doc.file_hash.clone()),
                    file_size_bytes,
                    modified_at,
                    format!("embedding failed: {e}"),
                );
                let _ = state_store.mark_failed(&source_path, e.to_string());
                let files_message = right_align_truncate(&display_name, files_message_width);
                let bytes_message = right_align_truncate(&display_name, bytes_message_width);
                files_bar.set_message(files_message);
                bytes_bar.set_message(bytes_message);
                files_bar.inc(1);
                bytes_bar.inc(candidate.size_bytes);
                continue;
            }
        };

        // debug: trace embedding values
        if !embeddings.is_empty() {
            let first_emb = &embeddings[0];
            let sum: f32 = first_emb.iter().sum();
            let first_5: Vec<f32> = first_emb.iter().take(5).copied().collect();
            tracing::debug!(
                "sync_documents: embeddings count={}, first vector: len={}, sum={}, first_5={:?}",
                embeddings.len(),
                first_emb.len(),
                sum,
                first_5
            );
        }

        if embeddings.is_empty() {
            tracing::warn!(
                "skipping file {:?} because embeddings are empty",
                file_path
            );
            let _ = state_store.record_failure(
                &source_path,
                Some(doc.file_hash.clone()),
                file_size_bytes,
                modified_at,
                "embeddings empty".to_string(),
            );
            let _ = state_store.mark_failed(&source_path, "embeddings empty".to_string());
            let files_message = right_align_truncate(&display_name, files_message_width);
            let bytes_message = right_align_truncate(&display_name, bytes_message_width);
            files_bar.set_message(files_message);
            bytes_bar.set_message(bytes_message);
            files_bar.inc(1);
            bytes_bar.inc(candidate.size_bytes);
            continue;
        }

        if embeddings.len() != chunks.len() {
            tracing::error!(
                "embedding count mismatch for {:?}: chunks={}, embeddings={}",
                file_path,
                chunks.len(),
                embeddings.len()
            );
            let _ = state_store.record_failure(
                &source_path,
                Some(doc.file_hash.clone()),
                file_size_bytes,
                modified_at,
                "embedding count mismatch".to_string(),
            );
            let _ =
                state_store.mark_failed(&source_path, "embedding count mismatch".to_string());
            let files_message = right_align_truncate(&display_name, files_message_width);
            let bytes_message = right_align_truncate(&display_name, bytes_message_width);
            files_bar.set_message(files_message);
            bytes_bar.set_message(bytes_message);
            files_bar.inc(1);
            bytes_bar.inc(candidate.size_bytes);
            continue;
        }

        // create points and add to bm25 index
        let mut points = Vec::new();
        for (chunk_text, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            // generate unique id for chunk
            let chunk_id = Uuid::new_v4().to_string();

            // create qdrant point
            let mut payload: HashMap<String, Value> = HashMap::new();
            payload.insert("text".to_string(), chunk_text.clone().into());
            payload.insert(
                "source".to_string(),
                file_path.to_string_lossy().to_string().into(),
            );
            points.push(PointStruct::new(chunk_id.clone(), embedding, payload));

            // add to bm25 index
            if let Err(e) = bm25_index.add_document(&chunk_id, &chunk_text) {
                tracing::warn!("failed to add chunk {} to bm25 index: {}", chunk_id, e);
            }
        }

        // retry upsert with exponential backoff
        let mut retry_count = 0;
        let max_retries = 3;
        let mut success = false;

        while retry_count <= max_retries && !success {
            let upsert_request =
                UpsertPointsBuilder::new(collection_name.clone(), points.clone()).build();
            match qdrant_client.upsert_points(upsert_request).await {
                Ok(_) => {
                    tracing::debug!("successfully ingested file {:?}", file_path);
                    let _ = state_store.mark_complete(&source_path, points.len());
                    let _ = state_store.clear_failure(&source_path);
                    let files_message = right_align_truncate(&display_name, files_message_width);
                    let bytes_message = right_align_truncate(&display_name, bytes_message_width);
                    files_bar.set_message(files_message);
                    bytes_bar.set_message(bytes_message);
                    files_bar.inc(1);
                    bytes_bar.inc(candidate.size_bytes);
                    success = true;
                }
                Err(e) => {
                    if retry_count < max_retries {
                        let delay_ms = 100 * (2_u64.pow(retry_count));
                        tracing::warn!(
                            "upsert failed for {:?} (attempt {}/{}), retrying in {}ms: {}",
                            file_path,
                            retry_count + 1,
                            max_retries + 1,
                            delay_ms,
                            e
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                        retry_count += 1;
                    } else {
                        tracing::error!(
                            "failed to upsert points for {:?} after {} attempts: {}",
                            file_path,
                            max_retries + 1,
                            e
                        );
                        let _ = state_store.record_failure(
                            &source_path,
                            Some(doc.file_hash.clone()),
                            file_size_bytes,
                            modified_at,
                            format!("upsert failed: {e}"),
                        );
                        let _ = state_store.mark_failed(&source_path, e.to_string());
                        let files_message = right_align_truncate(&display_name, files_message_width);
                        let bytes_message = right_align_truncate(&display_name, bytes_message_width);
                        files_bar.set_message(files_message);
                        bytes_bar.set_message(bytes_message);
                        files_bar.inc(1);
                        bytes_bar.inc(candidate.size_bytes);
                        retry_count += 1;
                    }
                }
            }
        }
    }

    files_bar.finish_with_message("done");
    bytes_bar.finish_with_message("done");

    // save bm25 index
    tracing::info!("saving bm25 index with {} documents", bm25_index.num_docs());
    if let Err(e) = save_index(&bm25_index, &config.bm25_index_path) {
        tracing::error!("failed to save bm25 index: {}", e);
    }

    eprintln!("debug: document sync complete, checking if commits flag is set");
    eprintln!("debug: args.commits = {}", args.commits);

    // index git commits if requested
    if args.commits {
        eprintln!("debug: starting commit indexing");
        tracing::info!("indexing git commits");
        if let Err(e) = sync_commits(&args, &config, &qdrant_client, &mut embedder).await {
            tracing::error!("failed to index commits: {}", e);
        }
        eprintln!("debug: commit indexing complete");
    }

    tracing::info!("sync completed");
    let _ = log_tx.send("__bbt_log_close__".to_string());
    drop(log_tx);
    let _ = log_thread.join();
    Ok(())
}

async fn sync_commits(
    args: &SyncCommand,
    config: &backbone::config::BbtConfig,
    qdrant_client: &qdrant_client::Qdrant,
    embedder: &mut backbone::embedding::OnnxEmbedder,
) -> Result<()> {
    use backbone::git::{CommitExtractor, chunk_commit, CommitChunkingStrategy};
    use chrono::NaiveDate;
    use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder};
    use std::collections::HashMap;

    let collection_name = "bbt_commits";
    let vector_dimensions = config.embedding_dims;

    // create commits collection if it doesn't exist
    let collection_exists = qdrant_client
        .collection_exists(collection_name)
        .await
        .unwrap_or(false);

    if !collection_exists {
        tracing::info!("creating commits collection '{}'", collection_name);
        let vector_params = VectorParamsBuilder::new(vector_dimensions as u64, Distance::Cosine).build();
        qdrant_client
            .create_collection(CreateCollectionBuilder::new(collection_name).vectors_config(vector_params))
            .await?;
        tracing::info!("commits collection '{}' created", collection_name);
    }

    // parse commits_since date if provided
    let since = if let Some(date_str) = &args.commits_since {
        match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            Ok(date) => Some(date.and_hms_opt(0, 0, 0).unwrap().and_utc()),
            Err(e) => {
                tracing::warn!("failed to parse commits_since date '{}': {}", date_str, e);
                None
            }
        }
    } else {
        None
    };

    // discover git repositories in sync paths
    let mut repo_paths = Vec::new();
    for path in &args.paths {
        if path.is_dir() {
            // recursively find all repos in directory (max depth 10)
            eprintln!("debug: recursively scanning {} for git repositories", path.display());
            let repos = find_git_repos_recursive(path, 10);
            eprintln!("debug: found {} repositories in {}", repos.len(), path.display());
            for repo_path in repos {
                if !repo_paths.contains(&repo_path) {
                    repo_paths.push(repo_path.clone());
                    eprintln!("debug:   - {}", repo_path.display());
                }
            }
        } else {
            // for files, find the containing repo
            if let Some(repo_path) = find_git_repo(path) {
                if !repo_paths.contains(&repo_path) {
                    repo_paths.push(repo_path);
                }
            }
        }
    }

    eprintln!("debug: total: found {} git repositories for commit indexing", repo_paths.len());
    tracing::info!("found {} git repositories for commit indexing", repo_paths.len());

    let mut total_commits = 0;
    let mut total_chunks = 0;
    let repo_count = repo_paths.len();

    // process each repository
    for repo_path in repo_paths {
        let repo_name = get_repo_name(&repo_path);
        tracing::info!("processing repository: {}", repo_name);

        // extract commits
        let extractor = match CommitExtractor::new(&repo_path) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!("failed to open repository {:?}: {}", repo_path, err);
                continue;
            }
        };

        let commits = match extractor.extract_commits(since, Some(args.max_commits)) {
            Ok(c) => c,
            Err(err) => {
                tracing::error!("failed to extract commits from {:?}: {}", repo_path, err);
                continue;
            }
        };

        tracing::info!("extracted {} commits from {}", commits.len(), repo_name);
        total_commits += commits.len();

        // process each commit
        for commit in commits {
            // chunk commit (message only for now)
            let chunks = match chunk_commit(&commit, CommitChunkingStrategy::MessageOnly, config.chunk_size, config.chunk_overlap) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("failed to chunk commit {}: {}", commit.commit_hash, e);
                    continue;
                }
            };

            total_chunks += chunks.len();

            // extract chunk texts
            let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

            // generate embeddings
            let embeddings = match embedder.embed(&chunk_texts) {
                Ok(e) => e,
                Err(err) => {
                    tracing::error!("failed to embed chunks for commit {}: {}", commit.commit_hash, err);
                    continue;
                }
            };

            // debug: trace embedding values
            if !embeddings.is_empty() {
                let first_emb = &embeddings[0];
                let sum: f32 = first_emb.iter().sum();
                let first_5: Vec<f32> = first_emb.iter().take(5).copied().collect();
                tracing::debug!(
                    "sync_commits: embeddings count={}, first vector: len={}, sum={}, first_5={:?}",
                    embeddings.len(),
                    first_emb.len(),
                    sum,
                    first_5
                );
            }

            // create qdrant points
            let mut points = Vec::new();
            for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
                let mut payload = HashMap::new();
                payload.insert("text".to_string(), chunk.content.clone().into());
                payload.insert("commit_hash".to_string(), commit.commit_hash.clone().into());
                payload.insert("repository_name".to_string(), commit.metadata.repository_name.clone().into());
                payload.insert("repository_path".to_string(), commit.repository_path.to_string_lossy().to_string().into());
                payload.insert("chunk_type".to_string(), chunk.metadata.chunk_type.as_str().to_string().into());
                payload.insert("chunk_index".to_string(), (chunk.metadata.chunk_index as i64).into());
                payload.insert("total_chunks".to_string(), (chunk.metadata.total_chunks as i64).into());
                payload.insert("author_name".to_string(), commit.metadata.author.name.clone().into());
                payload.insert("author_email".to_string(), commit.metadata.author.email.clone().into());
                payload.insert("committer_name".to_string(), commit.metadata.committer.name.clone().into());
                payload.insert("committer_email".to_string(), commit.metadata.committer.email.clone().into());
                payload.insert("commit_time".to_string(), commit.metadata.commit_time.to_rfc3339().into());
                payload.insert("classification".to_string(), commit.metadata.classification.as_str().to_string().into());
                payload.insert("files_changed_count".to_string(), (commit.metadata.diff_stats.files_changed as i64).into());
                payload.insert("insertions".to_string(), (commit.metadata.diff_stats.insertions as i64).into());
                payload.insert("deletions".to_string(), (commit.metadata.diff_stats.deletions as i64).into());

                // store file paths as array
                let file_paths_values: Vec<qdrant_client::qdrant::Value> = commit.metadata.diff_stats.file_paths
                    .iter()
                    .map(|path| path.clone().into())
                    .collect();
                payload.insert("file_paths".to_string(), qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                        qdrant_client::qdrant::ListValue { values: file_paths_values }
                    ))
                }.into());

                let point = PointStruct::new(
                    chunk.id.clone(),
                    embedding.clone(),
                    payload,
                );
                points.push(point);
            }

            // upsert to qdrant
            let upsert_request = UpsertPointsBuilder::new(collection_name.to_string(), points).build();
            if let Err(e) = qdrant_client.upsert_points(upsert_request).await {
                tracing::error!("failed to upsert commit {}: {}", commit.commit_hash, e);
            }
        }
    }

    tracing::info!(
        "indexed {} commits ({} chunks) from {} repositories",
        total_commits,
        total_chunks,
        repo_count
    );

    Ok(())
}

struct FileCandidate {
    path: PathBuf,
    size_bytes: u64,
    modified_at: Option<chrono::DateTime<chrono::Utc>>,
}

enum ScanEvent {
    File(FileCandidate),
    Error(String),
}

struct ScanTotals {
    files: u64,
    bytes: u64,
}

fn collect_candidates(
    args: &SyncCommand,
    extensions: &[String],
    scan_bar: &indicatif::ProgressBar,
    state_store: &mut backbone::storage::StateStore,
    scan_scope_key: &str,
    scan_progress: &backbone::storage::ScanProgress,
    scan_message_width: usize,
) -> Result<(Vec<FileCandidate>, ScanTotals)> {
    use ignore::{WalkBuilder, WalkState};
    use std::collections::HashSet;
    use std::sync::{mpsc, Arc};
    use std::thread;

    let mut candidates = Vec::new();
    let mut totals = ScanTotals {
        files: scan_progress.files_seen,
        bytes: scan_progress.bytes_seen,
    };

    let mut roots = Vec::new();
    for path in &args.paths {
        if !path.is_dir() {
            tracing::warn!("path {:?} is not a directory, skipping", path);
            continue;
        }
        roots.push(path.clone());
    }

    if roots.is_empty() {
        return Ok((candidates, totals));
    }

    let extensions_set: HashSet<String> = extensions.iter().cloned().collect();
    let extensions_set = Arc::new(extensions_set);

    let (tx, rx) = mpsc::channel::<ScanEvent>();
    let worker_tx = tx.clone();
    let worker_roots = roots.clone();
    let worker_extensions = Arc::clone(&extensions_set);
    let git_only = args.git_only;
    let honor_gitignore = args.honor_gitignore;

    let worker = thread::spawn(move || {
        let mut builder = WalkBuilder::new(&worker_roots[0]);
        for path in worker_roots.iter().skip(1) {
            builder.add(path);
        }

        builder.hidden(false).ignore(false);
        if honor_gitignore {
            builder.git_ignore(true).git_global(true).git_exclude(true);
        } else {
            builder.git_ignore(false).git_global(false).git_exclude(false);
        }
        if git_only {
            builder.require_git(true);
        }

        // add .dockerignore support
        builder.add_custom_ignore_filename(".dockerignore");

        // filter out common directories that should never be scanned
        builder.filter_entry(|entry| {
            let file_name = entry.file_name().to_string_lossy();
            let path_str = entry.path().to_string_lossy();

            // always exclude these directories
            let excluded_dirs = [
                // python
                ".venv", "venv", "__pycache__", ".tox", ".pytest_cache",
                "site-packages", ".eggs", "*.egg-info",
                // node.js
                "node_modules", ".npm", ".yarn", ".pnp", ".pnp.js",
                // rust
                "target",
                // general build/cache
                ".cache", "build", "dist", ".next", ".nuxt", ".svelte-kit",
                // version control
                ".git", ".svn", ".hg",
                // ide/editor
                ".vscode", ".idea", ".vs", ".DS_Store",
                // docker/containers
                ".docker",
                // data/logs
                "data", "logs", "tmp", "temp",
            ];

            for excluded in &excluded_dirs {
                if file_name == *excluded || file_name.ends_with(excluded) {
                    return false;
                }
            }

            // skip if path contains any excluded directory
            for excluded in &excluded_dirs {
                if path_str.contains(&format!("/{}/", excluded)) ||
                   path_str.contains(&format!("\\{}/", excluded)) {
                    return false;
                }
            }

            true
        });

        let threads = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(4);
        builder.threads(threads);

        let walker = builder.build_parallel();
        walker.run(|| {
            let sender = worker_tx.clone();
            let extensions = Arc::clone(&worker_extensions);
            Box::new(move |result| {
                let entry = match result {
                    Ok(entry) => entry,
                    Err(err) => {
                        let _ = sender.send(ScanEvent::Error(format!(
                            "scan entry error: {}",
                            err
                        )));
                        return WalkState::Continue;
                    }
                };

                if !entry
                    .file_type()
                    .map(|file_type| file_type.is_file())
                    .unwrap_or(false)
                {
                    return WalkState::Continue;
                }

                let file_path = entry.path();
                let ext = match file_path.extension().and_then(|value| value.to_str()) {
                    Some(ext) => ext.to_lowercase(),
                    None => return WalkState::Continue,
                };
                if !extensions.contains(&ext) {
                    return WalkState::Continue;
                }

                let metadata = match entry.metadata() {
                    Ok(metadata) => metadata,
                    Err(err) => {
                        let _ = sender.send(ScanEvent::Error(format!(
                            "failed to read metadata for {:?}: {}",
                            file_path, err
                        )));
                        return WalkState::Continue;
                    }
                };

                let size_bytes = metadata.len();
                let modified_at = metadata
                    .modified()
                    .ok()
                    .map(|time| chrono::DateTime::<chrono::Utc>::from(time));

                let candidate = FileCandidate {
                    path: file_path.to_path_buf(),
                    size_bytes,
                    modified_at,
                };

                if sender.send(ScanEvent::File(candidate)).is_err() {
                    return WalkState::Quit;
                }

                WalkState::Continue
            })
        });
    });

    drop(tx);

    for event in rx {
        match event {
            ScanEvent::File(candidate) => {
                let source_path = candidate.path.to_string_lossy().to_string();
                let progress = match state_store.record_scan_entry(
                    scan_scope_key,
                    &scan_progress.run_id,
                    &source_path,
                    candidate.size_bytes,
                    candidate.modified_at,
                ) {
                    Ok(progress) => progress,
                    Err(e) => {
                        tracing::warn!(
                            "failed to update scan state for {:?}: {}",
                            candidate.path,
                            e
                        );
                        backbone::storage::ScanProgress {
                            run_id: scan_progress.run_id.clone(),
                            files_seen: totals.files,
                            bytes_seen: totals.bytes,
                        }
                    }
                };

                totals.files = progress.files_seen;
                totals.bytes = progress.bytes_seen;
                scan_bar.set_message(format_scan_message(
                    totals.files,
                    totals.bytes,
                    &source_path,
                    scan_message_width,
                ));

                candidates.push(candidate);
            }
            ScanEvent::Error(message) => {
                tracing::warn!("{message}");
            }
        }
    }

    if worker.join().is_err() {
        return Err(anyhow::anyhow!("scan worker panicked"));
    }

    scan_bar.set_message(format!(
        "{} files, {} bytes",
        totals.files, totals.bytes
    ));

    Ok((candidates, totals))
}

fn build_scan_scope_key(args: &SyncCommand, extensions: &[String]) -> String {
    let mut paths: Vec<String> = args
        .paths
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    paths.sort();

    let mut ext_list = extensions.to_vec();
    ext_list.sort();

    format!(
        "paths={};ext={};git_only={};honor_gitignore={}",
        paths.join("|"),
        ext_list.join("|"),
        args.git_only,
        args.honor_gitignore
    )
}

fn log_message_width(term_width: usize, prefix_len: usize) -> usize {
    let left_width = prefix_len.saturating_add(1);
    term_width.saturating_sub(left_width).saturating_sub(1)
}

fn truncate_log_line(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let value_chars: Vec<char> = value.chars().collect();
    if value_chars.len() <= width {
        return pad_right(value, width);
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let available = width - 3;
    let prefix_len = (available + 1) / 2;
    let suffix_len = available - prefix_len;
    let prefix = value_chars[..prefix_len].iter().collect::<String>();
    let suffix = value_chars[value_chars.len() - suffix_len..]
        .iter()
        .collect::<String>();
    pad_right(&format!("{prefix}...{suffix}"), width)
}

fn pad_right(value: &str, width: usize) -> String {
    let length = value.chars().count();
    if length >= width {
        value.to_string()
    } else {
        format!("{}{}", value, " ".repeat(width - length))
    }
}

fn progress_message_width(
    term_width: usize,
    prefix_len: usize,
    bar_width: usize,
    left_value_width: usize,
    right_value_width: usize,
) -> usize {
    let fixed = prefix_len
        .saturating_add(1) // space after prefix
        .saturating_add(1) // [
        .saturating_add(bar_width)
        .saturating_add(1) // ]
        .saturating_add(1) // space
        .saturating_add(left_value_width)
        .saturating_add(1) // /
        .saturating_add(right_value_width)
        .saturating_add(1); // space before msg
    term_width.saturating_sub(fixed)
}

fn scan_message_width(term_width: usize, prefix_len: usize) -> usize {
    let spinner_len = 1_usize;
    let left_width = prefix_len.saturating_add(spinner_len).saturating_add(2);
    term_width.saturating_sub(left_width).saturating_sub(1)
}

fn format_scan_message(
    files: u64,
    bytes: u64,
    source_path: &str,
    width: usize,
) -> String {
    let counts = format!("{} files {} bytes", files, bytes);
    if width == 0 {
        return counts;
    }

    let spacer = " ";
    let available = width.saturating_sub(counts.len()).saturating_sub(spacer.len());
    let file_display = right_align_truncate(source_path, available);
    format!("{counts}{spacer}{file_display}")
}

/// find git repository root for a given path
fn find_git_repo(path: &Path) -> Option<PathBuf> {
    // start from the path itself if it's a directory, or its parent if it's a file
    let start = if path.is_dir() {
        Some(path)
    } else {
        path.parent()
    };

    let mut current = start;
    while let Some(dir) = current {
        let git_dir = dir.join(".git");
        if git_dir.exists() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

/// recursively find all git repositories under a directory
fn find_git_repos_recursive(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    use std::collections::HashSet;
    use walkdir::WalkDir;

    let mut repos = Vec::new();
    let mut seen = HashSet::new();

    for entry in WalkDir::new(root)
        .max_depth(max_depth)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let file_name = e.file_name().to_string_lossy();

            // always allow .git directories
            if file_name == ".git" {
                return true;
            }

            // skip common directories that should never be scanned
            let excluded_dirs = [
                // python
                ".venv", "venv", "__pycache__", ".tox", ".pytest_cache",
                "site-packages", ".eggs",
                // node.js
                "node_modules", ".npm", ".yarn", ".pnp",
                // rust
                "target",
                // general build/cache
                ".cache", "build", "dist", ".next", ".nuxt", ".svelte-kit",
                // version control (except .git)
                ".svn", ".hg",
                // ide/editor
                ".vscode", ".idea", ".vs", ".DS_Store",
                // docker/containers
                ".docker",
                // data/logs
                "data", "logs", "tmp", "temp",
            ];

            for excluded in &excluded_dirs {
                if file_name == *excluded {
                    return false;
                }
            }

            true
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("skipping entry: {}", e);
                continue;
            }
        };

        // check if this is a .git directory
        if entry.file_name() == ".git" && entry.file_type().is_dir() {
            if let Some(repo_root) = entry.path().parent() {
                let canonical = match repo_root.canonicalize() {
                    Ok(p) => p,
                    Err(_) => repo_root.to_path_buf(),
                };

                // deduplicate using canonical paths
                if seen.insert(canonical.clone()) {
                    repos.push(canonical);
                    tracing::debug!("found git repo: {}", repo_root.display());
                }
            }
        }
    }

    repos
}

/// extract repository name from git repo root
fn get_repo_name(repo_root: &Path) -> String {
    repo_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn right_align_truncate(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let value_chars: Vec<char> = value.chars().collect();
    let value_len = value_chars.len();

    let truncated = if value_len > width {
        if width <= 3 {
            ".".repeat(width)
        } else {
            let suffix_len = width - 3;
            let suffix = value_chars[value_len - suffix_len..].iter().collect::<String>();
            format!("...{suffix}")
        }
    } else {
        value.to_string()
    };

    let truncated_len = truncated.chars().count();
    if truncated_len >= width {
        truncated
    } else {
        format!("{}{}", " ".repeat(width - truncated_len), truncated)
    }
}

async fn query_documents(args: QueryCommand) -> Result<()> {
    use backbone::config::{BbtConfig, RetrievalMode};
    use backbone::embedding::{ensure_onnx_model, EmbeddingModelInfo, OnnxEmbedder};
    use backbone::embedding::ExecutionProvider as EmbeddingExecutionProvider;
    use backbone::reranking::{ensure_rerank_model, ExecutionProvider as RerankExecutionProvider, OnnxReranker, RerankModelInfo};
    use backbone::retrieval::{Bm25Scorer, FusionStrategy, fuse_results};
    use backbone::retrieval::bm25::load_index;
    use backbone::tracing::init_tracing;
    use qdrant_client::qdrant::SearchPointsBuilder;
    use qdrant_client::Qdrant;

    let _guard = init_tracing("bbt", None)?;
    let mut config = BbtConfig::from_env()?;

    if let Some(mode) = args.mode {
        config.retrieval_mode = mode.into();
    }
    if let Some(top_k) = args.top_k {
        config.top_k = top_k;
    }

    // force vector mode for commit queries (no bm25 index for commits)
    if args.commits && config.retrieval_mode != RetrievalMode::Vector {
        tracing::info!("forcing vector mode for commit query (bm25/hybrid not supported for commits)");
        config.retrieval_mode = RetrievalMode::Vector;
    }

    let query = args.query.trim();
    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    if config.top_k == 0 {
        anyhow::bail!("top_k must be greater than 0");
    }

    // load bm25 index if needed
    let bm25_index = if config.retrieval_mode == RetrievalMode::Bm25
        || config.retrieval_mode == RetrievalMode::Hybrid
    {
        tracing::info!("loading bm25 index from {:?}", config.bm25_index_path);
        Some(load_index(&config.bm25_index_path)?)
    } else {
        None
    };

    // calculate initial retrieval limit
    // if reranking is enabled, retrieve more candidates for better recall
    let rerank_multiplier = if config.enable_rerank { 3 } else { 1 };
    let initial_limit = config.top_k * rerank_multiplier;

    // perform retrieval based on mode
    let candidates: Vec<QueryResultOutput> = match config.retrieval_mode {
        RetrievalMode::Vector => {
            tracing::info!("vector search with top_k={}", initial_limit);

            let model_info = EmbeddingModelInfo::default_model();
            let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
            let mut embedder = OnnxEmbedder::new(&model_path, model_info, EmbeddingExecutionProvider::detect())?;

            let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
                Qdrant::from_url(&config.qdrant_url)
                    .api_key(api_key.clone())
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            } else {
                Qdrant::from_url(&config.qdrant_url)
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            };

            let embeddings = embedder.embed(&[query.to_string()])?;
            let query_vector = embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("failed to embed query"))?;

            // select collection based on query type
            let collection_name = if args.commits {
                "bbt_commits".to_string()
            } else {
                "bbt".to_string()
            };

            let collection_exists = qdrant_client
                .collection_exists(&collection_name)
                .await
                .unwrap_or(false);
            if !collection_exists {
                let hint = if args.commits {
                    "run sync --commits first"
                } else {
                    "run sync first"
                };
                anyhow::bail!("collection '{}' not found, {}", collection_name, hint);
            }

            let mut search_builder =
                SearchPointsBuilder::new(collection_name.clone(), query_vector, initial_limit as u64)
                    .with_payload(true);

            // apply filters
            if let Some(filter) = if args.commits {
                build_commit_filter(&args)?
            } else {
                build_query_filter(&args)
            } {
                search_builder = search_builder.filter(filter);
            }

            let response = qdrant_client.search_points(search_builder).await?;

            // convert qdrant results to intermediate format
            let mut results = Vec::new();
            for point in response.result {
                let payload = point.payload;
                let text = payload
                    .get("text")
                    .and_then(|value| value_as_string(value))
                    .unwrap_or_default();
                let source_raw = payload
                    .get("source")
                    .and_then(|value| value_as_string(value));
                let citation_raw = format_citation(source_raw.as_deref());
                let citation = display_citation(&citation_raw, source_raw.as_deref(), args.citation_mode);
                let source = display_source(source_raw.as_deref(), args.citation_mode);
                let id = point_id_to_string(point.id);

                // extract metadata (commit-specific or general)
                let mut metadata = std::collections::HashMap::new();
                if args.commits {
                    // extract commit metadata
                    if let Some(commit_hash) = payload.get("commit_hash").and_then(|v| value_as_string(v)) {
                        metadata.insert("commit_hash".to_string(), commit_hash);
                    }
                    if let Some(author_name) = payload.get("author_name").and_then(|v| value_as_string(v)) {
                        metadata.insert("author_name".to_string(), author_name);
                    }
                    if let Some(author_email) = payload.get("author_email").and_then(|v| value_as_string(v)) {
                        metadata.insert("author_email".to_string(), author_email);
                    }
                    if let Some(commit_time) = payload.get("commit_time").and_then(|v| value_as_string(v)) {
                        metadata.insert("commit_time".to_string(), commit_time);
                    }
                    if let Some(classification) = payload.get("classification").and_then(|v| value_as_string(v)) {
                        metadata.insert("classification".to_string(), classification);
                    }
                    if let Some(insertions) = payload.get("insertions").and_then(|v| value_as_string(v)) {
                        metadata.insert("insertions".to_string(), insertions);
                    }
                    if let Some(deletions) = payload.get("deletions").and_then(|v| value_as_string(v)) {
                        metadata.insert("deletions".to_string(), deletions);
                    }
                    if let Some(repository_name) = payload.get("repository_name").and_then(|v| value_as_string(v)) {
                        metadata.insert("repository_name".to_string(), repository_name);
                    }
                    // file_paths is an array, join to string
                    if let Some(files) = payload.get("file_paths") {
                        if let Some(qdrant_client::qdrant::Value { kind: Some(qdrant_client::qdrant::value::Kind::ListValue(list)) }) = Some(files) {
                            let file_list: Vec<String> = list
                                .values
                                .iter()
                                .filter_map(|v| value_as_string(v))
                                .collect();
                            if !file_list.is_empty() {
                                metadata.insert("file_paths".to_string(), file_list.join(", "));
                            }
                        }
                    }
                }

                results.push(QueryResultOutput {
                    id,
                    score: point.score,
                    text,
                    citation,
                    source,
                    metadata,
                });
            }

            // apply hybrid scoring for commits
            if args.commits && args.recency_weight > 0.0 {
                use chrono::Utc;
                let now = Utc::now();

                for result in &mut results {
                    if let Some(commit_time_str) = result.metadata.get("commit_time") {
                        // parse commit time
                        if let Ok(commit_time) = chrono::DateTime::parse_from_rfc3339(commit_time_str) {
                            let commit_time_utc = commit_time.with_timezone(&Utc);
                            let age = now.signed_duration_since(commit_time_utc);
                            let days_old = age.num_days() as f64;

                            // calculate time decay score
                            let time_score = args.time_decay.calculate_score(days_old);

                            // hybrid score: (1-w) * semantic + w * recency
                            let semantic_score = result.score;
                            result.score = (1.0 - args.recency_weight) * semantic_score
                                + args.recency_weight * time_score;
                        }
                    }
                }
            }

            // sort results based on sort_by parameter
            if args.commits {
                use chrono::Utc;
                match args.sort_by {
                    SortBy::Score => {
                        // default: sort by score descending (highest first)
                        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    SortBy::DateAsc => {
                        // oldest first
                        results.sort_by(|a, b| {
                            let a_time = a.metadata.get("commit_time")
                                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                                .map(|dt| dt.with_timezone(&Utc));
                            let b_time = b.metadata.get("commit_time")
                                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                                .map(|dt| dt.with_timezone(&Utc));
                            a_time.cmp(&b_time)
                        });
                    }
                    SortBy::DateDesc => {
                        // newest first
                        results.sort_by(|a, b| {
                            let a_time = a.metadata.get("commit_time")
                                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                                .map(|dt| dt.with_timezone(&Utc));
                            let b_time = b.metadata.get("commit_time")
                                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                                .map(|dt| dt.with_timezone(&Utc));
                            b_time.cmp(&a_time)
                        });
                    }
                }
            } else {
                // for documents, always sort by score
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            }

            results
        }
        RetrievalMode::Bm25 => {
            tracing::info!("bm25 search with top_k={}", initial_limit);

            let index = bm25_index.as_ref().unwrap();
            let scorer = Bm25Scorer::with_params(config.bm25_k1, config.bm25_b);
            let bm25_results = scorer.search(index, query, initial_limit);

            tracing::info!("bm25 returned {} results", bm25_results.len());

            // convert bm25 results to QueryResultOutput format
            bm25_results
                .into_iter()
                .map(|result| {
                    let source_raw = index
                        .get_document(&result.doc_id)
                        .and_then(|_doc| {
                            // extract source from chunk_text metadata if available
                            // for now, use doc_id as fallback
                            Some(result.doc_id.clone())
                        });
                    let citation_raw = format_citation(source_raw.as_deref());
                    let citation = display_citation(&citation_raw, source_raw.as_deref(), args.citation_mode);
                    let source = display_source(source_raw.as_deref(), args.citation_mode);

                    QueryResultOutput {
                        id: result.doc_id,
                        score: result.score,
                        text: result.text,
                        citation,
                        source,
                        metadata: std::collections::HashMap::new(),
                    }
                })
                .collect()
        }
        RetrievalMode::Hybrid => {
            tracing::info!("hybrid search (vector + bm25) with top_k={}", initial_limit);

            // perform vector search
            let model_info = EmbeddingModelInfo::default_model();
            let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
            let mut embedder = OnnxEmbedder::new(&model_path, model_info, EmbeddingExecutionProvider::detect())?;

            let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
                Qdrant::from_url(&config.qdrant_url)
                    .api_key(api_key.clone())
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            } else {
                Qdrant::from_url(&config.qdrant_url)
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            };

            let embeddings = embedder.embed(&[query.to_string()])?;
            let query_vector = embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("failed to embed query"))?;

            let collection_name = "bbt".to_string();
            let collection_exists = qdrant_client
                .collection_exists(&collection_name)
                .await
                .unwrap_or(false);
            if !collection_exists {
                anyhow::bail!("collection '{}' not found, run sync first", collection_name);
            }

            let mut search_builder =
                SearchPointsBuilder::new(collection_name.clone(), query_vector, initial_limit as u64 * 2)
                    .with_payload(true);

            if let Some(filter) = build_query_filter(&args) {
                search_builder = search_builder.filter(filter);
            }

            let response = qdrant_client.search_points(search_builder).await?;

            // convert vector results
            let vector_results: Vec<(String, f32, String)> = response
                .result
                .into_iter()
                .map(|point| {
                    let id = point_id_to_string(point.id);
                    let text = point
                        .payload
                        .get("text")
                        .and_then(|value| value_as_string(value))
                        .unwrap_or_default();
                    (id, point.score, text)
                })
                .collect();

            // perform bm25 search
            let index = bm25_index.as_ref().unwrap();
            let scorer = Bm25Scorer::with_params(config.bm25_k1, config.bm25_b);
            let bm25_results = scorer.search(index, query, initial_limit * 2);

            tracing::info!(
                "hybrid: vector returned {}, bm25 returned {}",
                vector_results.len(),
                bm25_results.len()
            );

            // fuse results using weighted sum
            let strategy = FusionStrategy::WeightedSum {
                vector_weight: config.vector_weight,
                bm25_weight: config.bm25_weight,
            };
            let fused = fuse_results(&vector_results, &bm25_results, strategy, initial_limit)?;

            tracing::info!("fusion returned {} results", fused.len());

            // convert fused results to QueryResultOutput
            fused
                .into_iter()
                .map(|result| {
                    let source_raw = Some(result.doc_id.clone());
                    let citation_raw = format_citation(source_raw.as_deref());
                    let citation = display_citation(&citation_raw, source_raw.as_deref(), args.citation_mode);
                    let source = display_source(source_raw.as_deref(), args.citation_mode);

                    QueryResultOutput {
                        id: result.doc_id,
                        score: result.score,
                        text: result.text,
                        citation,
                        source,
                        metadata: std::collections::HashMap::new(),
                    }
                })
                .collect()
        }
    };

    // apply reranking if enabled
    let results = if config.enable_rerank && !candidates.is_empty() {
        tracing::info!("reranking {} candidates", candidates.len());

        // load reranker model
        let rerank_model_info = RerankModelInfo::custom(
            config.rerank_model_repo.clone(),
            config.rerank_model_file.clone(),
            512,
            true,
        );
        let rerank_model_path = ensure_rerank_model(&rerank_model_info, &config.model_cache_dir)?;
        let reranker = OnnxReranker::new(
            &rerank_model_path,
            rerank_model_info,
            RerankExecutionProvider::detect(),
        )?;

        // create (query, document) pairs
        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|candidate| (query.to_string(), candidate.text.clone()))
            .collect();

        // score with reranker
        let rerank_scores = reranker.score_batch(&pairs)?;

        // combine candidates with rerank scores
        let mut ranked: Vec<(QueryResultOutput, f32)> = candidates
            .into_iter()
            .zip(rerank_scores.into_iter())
            .collect();

        // sort by rerank score (descending)
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // take top rerank_top_k results
        let final_limit = config.rerank_top_k.min(ranked.len());
        let final_results: Vec<QueryResultOutput> = ranked
            .into_iter()
            .take(final_limit)
            .map(|(mut result, rerank_score)| {
                result.score = rerank_score;
                result
            })
            .collect();

        tracing::info!("reranking complete, returning {} results", final_results.len());
        final_results
    } else {
        // no reranking, use original vector search results
        candidates.into_iter().take(config.top_k).collect()
    };

    match args.format {
        OutputFormat::Json => {
            let results_json: Vec<serde_json::Value> = results
                .iter()
                .map(|result| {
                    serde_json::json!({
                        "id": &result.id,
                        "score": result.score,
                        "text": &result.text,
                        "citation": &result.citation,
                        "source": &result.source,
                    })
                })
                .collect();
            let output =
                serde_json::to_string_pretty(&serde_json::json!({ "results": results_json }))?;
            println!("{output}");
        }
        OutputFormat::Text => {
            render_text_results(&results, args.wrap_width, args.show_scores, args.commits);
        }
    }

    Ok(())
}

fn build_query_filter(args: &QueryCommand) -> Option<qdrant_client::qdrant::Filter> {
    use qdrant_client::qdrant::{Condition, Filter};

    let mut conditions = Vec::new();
    if let Some(source_prefix) = &args.source_prefix {
        let source_prefix = source_prefix.trim();
        if !source_prefix.is_empty() {
            // use text matching for path filtering
            // works best with path segments (e.g., 'backbone' or 'bookmarks/backbone')
            // rather than full absolute paths
            conditions.push(Condition::matches_text("source", source_prefix.to_string()));
        }
    }

    if conditions.is_empty() {
        None
    } else {
        Some(Filter::must(conditions))
    }
}

fn build_commit_filter(args: &QueryCommand) -> Result<Option<qdrant_client::qdrant::Filter>> {
    use backbone::error::BbtError;
    use qdrant_client::qdrant::{Condition, Filter, Range};
    use chrono::NaiveDate;

    let mut conditions = Vec::new();

    // filter by commit classification
    if let Some(commit_type) = &args.commit_type {
        conditions.push(Condition::matches("classification", commit_type.to_lowercase()));
    }

    // filter by author email
    if let Some(author) = &args.author {
        conditions.push(Condition::matches("author_email", author.to_string()));
    }

    // filter by date range (since)
    if let Some(since_str) = &args.since {
        let date = NaiveDate::parse_from_str(since_str, "%Y-%m-%d")
            .map_err(|e| BbtError::Schema(format!("invalid date format for --since: {} (expected YYYY-MM-DD)", e)))?;
        let since_time = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| BbtError::Schema("invalid time".to_string()))?
            .and_utc();

        conditions.push(Condition::range(
            "commit_time",
            Range {
                gte: Some(since_time.timestamp() as f64),
                ..Default::default()
            },
        ));
    }

    // filter by date range (until)
    if let Some(until_str) = &args.until {
        let date = NaiveDate::parse_from_str(until_str, "%Y-%m-%d")
            .map_err(|e| BbtError::Schema(format!("invalid date format for --until: {} (expected YYYY-MM-DD)", e)))?;
        let until_time = date
            .and_hms_opt(23, 59, 59)
            .ok_or_else(|| BbtError::Schema("invalid time".to_string()))?
            .and_utc();

        conditions.push(Condition::range(
            "commit_time",
            Range {
                lte: Some(until_time.timestamp() as f64),
                ..Default::default()
            },
        ));
    }

    Ok(if conditions.is_empty() {
        None
    } else {
        Some(Filter::must(conditions))
    })
}

fn format_citation(source: Option<&str>) -> String {
    if let Some(source_path) = source {
        let path = Path::new(source_path);
        if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
            return stem.to_string();
        }
    }
    "unknown".to_string()
}

fn display_citation(
    citation: &str,
    source: Option<&str>,
    mode: CitationMode,
) -> String {
    match mode {
        CitationMode::Full | CitationMode::Basename => citation.to_string(),
        CitationMode::Hash => hash_label(source.unwrap_or(citation)),
        CitationMode::Redact => "redacted".to_string(),
    }
}

fn display_source(source: Option<&str>, mode: CitationMode) -> String {
    match mode {
        CitationMode::Full => source.unwrap_or("unknown").to_string(),
        CitationMode::Basename => source
            .and_then(|value| Path::new(value).file_name().and_then(|name| name.to_str()))
            .unwrap_or("unknown")
            .to_string(),
        CitationMode::Hash => hash_label(source.unwrap_or("unknown")),
        CitationMode::Redact => "redacted".to_string(),
    }
}

fn hash_label(value: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    if let Ok(secret) = std::env::var("BBT_CITATION_SECRET") {
        if !secret.is_empty() {
            hasher.update(secret.as_bytes());
        }
    }
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let digest_hex = format!("{digest:x}");
    let prefix_len = 12.min(digest_hex.len());
    format!("src_{}", &digest_hex[..prefix_len])
}

fn value_as_string(value: &qdrant_client::qdrant::Value) -> Option<String> {
    use qdrant_client::qdrant::value::Kind as ValueKind;

    match &value.kind {
        Some(ValueKind::StringValue(value)) => Some(value.clone()),
        Some(ValueKind::IntegerValue(value)) => Some(value.to_string()),
        Some(ValueKind::DoubleValue(value)) => Some(value.to_string()),
        Some(ValueKind::BoolValue(value)) => Some(value.to_string()),
        _ => None,
    }
}

fn point_id_to_string(id: Option<qdrant_client::qdrant::PointId>) -> String {
    use qdrant_client::qdrant::point_id::PointIdOptions;

    match id.and_then(|value| value.point_id_options) {
        Some(PointIdOptions::Num(value)) => value.to_string(),
        Some(PointIdOptions::Uuid(value)) => value,
        None => "unknown".to_string(),
    }
}

fn render_text_results(results: &[QueryResultOutput], wrap_width: usize, show_scores: bool, is_commits: bool) {
    if results.is_empty() {
        println!("no results");
        return;
    }

    let width = if wrap_width == 0 { 100 } else { wrap_width };
    println!("results: {}", results.len());

    for (index, result) in results.iter().enumerate() {
        println!();
        println!("result {}/{}", index + 1, results.len());
        if show_scores {
            println!("score: {:.4}", result.score);
        }

        if is_commits {
            // commit-specific formatting
            let commit_hash = result.metadata.get("commit_hash")
                .map(|s| &s[..7.min(s.len())])  // short hash (7 chars)
                .unwrap_or("unknown");
            let classification = result.metadata.get("classification")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let author_name = result.metadata.get("author_name")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let author_email = result.metadata.get("author_email")
                .map(|s| s.as_str())
                .unwrap_or("");
            let commit_time = result.metadata.get("commit_time")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let insertions = result.metadata.get("insertions")
                .map(|s| s.as_str())
                .unwrap_or("0");
            let deletions = result.metadata.get("deletions")
                .map(|s| s.as_str())
                .unwrap_or("0");
            let file_paths = result.metadata.get("file_paths")
                .map(|s| s.as_str())
                .unwrap_or("");
            let repo_name = result.metadata.get("repository_name")
                .map(|s| s.as_str())
                .unwrap_or("unknown");

            // format: [hash] type: message first line
            let message_first_line = result.text.lines().next().unwrap_or("");
            println!("[{}] {}: {}", commit_hash, classification, message_first_line);
            println!("repository: {}", repo_name);
            println!("author: {} <{}>", author_name, author_email);
            println!("date: {}", commit_time);
            if file_paths.is_empty() {
                println!("changes: +{} -{}", insertions, deletions);
            } else {
                println!("files: {}", file_paths);
                println!("changes: +{} -{}", insertions, deletions);
            }
            println!();
            println!("message:");
            let wrapped_lines = wrap_text(&result.text, width);
            if wrapped_lines.is_empty() {
                println!("  [empty]");
            } else {
                for line in wrapped_lines {
                    println!("  {}", line);
                }
            }
        } else {
            // document-specific formatting (original)
            let text_chars = result.text.chars().count();
            let wrapped_lines = wrap_text(&result.text, width);

            println!("citation: {}", result.citation);
            println!("source: {}", result.source);
            println!("text_chars: {}", text_chars);
            println!("text:");
            if wrapped_lines.is_empty() {
                println!("  [empty]");
            } else {
                for line in wrapped_lines {
                    println!("  {}", line);
                }
            }
        }

        println!("{}", "-".repeat(40));
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.is_empty() {
            line.push_str(word);
            continue;
        }

        if line.len() + word.len() + 1 > width {
            lines.push(line);
            line = word.to_string();
        } else {
            line.push(' ');
            line.push_str(word);
        }
    }

    if !line.is_empty() {
        lines.push(line);
    }

    lines
}


async fn generate_schema(
    input: PathBuf,
    output: PathBuf,
    json_schema_output: Option<PathBuf>,
    repo_id: Option<String>,
    filename: Option<String>,
    cache_dir: Option<PathBuf>,
) -> Result<()> {
    use backbone::agent::parser::extract_create_table;
    use backbone::agent::prompt::{build_schema_task_prompt, SCHEMA_SYSTEM_PROMPT};
    use backbone::llm::downloader::ensure_gguf_model;
    use backbone::llm::model::{LlamaModel, Message, MessageRole, ModelConfig};
    use backbone::schema::inference::infer_json_schema;
    use backbone::schema::hints::derive_field_hints;
    use std::fs;

    let model_path = ensure_gguf_model(repo_id, filename, cache_dir)?;

    let json_str = fs::read_to_string(&input)?;
    let json_value: serde_json::Value = serde_json::from_str(&json_str)?;

    let schema = infer_json_schema(&json_value);
    let hints = derive_field_hints(&schema);
    let user_prompt = build_schema_task_prompt(&schema, &hints);

    let model_config = ModelConfig::default();
    let model = LlamaModel::new(model_path, model_config)?;

    let messages = vec![
        Message {
            role: MessageRole::System,
            content: SCHEMA_SYSTEM_PROMPT.to_string(),
        },
        Message {
            role: MessageRole::User,
            content: user_prompt,
        },
    ];

    let response = model.generate(messages)?;
    let sql = extract_create_table(&response)?;

    fs::write(&output, sql)?;
    println!("sql schema written to: {}", output.display());

    if let Some(schema_output) = json_schema_output {
        let formatted_json = serde_json::to_string_pretty(&json_value)?;
        fs::write(&schema_output, &formatted_json)?;
        println!("json schema written to: {}", schema_output.display());
    }

    Ok(())
}

async fn generate_json_schema(
    input: PathBuf,
    output: PathBuf,
    _repo_id: Option<String>,
    _filename: Option<String>,
    _cache_dir: Option<PathBuf>,
) -> Result<()> {
    use std::fs;

    let json_str = fs::read_to_string(&input)?;
    let json_value: serde_json::Value = serde_json::from_str(&json_str)?;
    let formatted_json = serde_json::to_string_pretty(&json_value)?;

    fs::write(&output, formatted_json)?;
    println!("JSON schema written to: {}", output.display());

    Ok(())
}
