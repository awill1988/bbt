use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bbt")]
#[command(about = "big brain time - document ingestion and RAG", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate database schema from bookmark JSON
    Gen {
        #[command(subcommand)]
        subcommand: GenCommands,
    },
}

#[derive(Subcommand)]
enum GenCommands {
    /// Generate SQL schema from bookmark JSON
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

        /// Disable JSON Schema generation
        #[arg(long, default_value = "false")]
        no_json_schema: bool,

        /// Enhance JSON Schema with LLM-generated descriptions
        #[arg(long, default_value = "false")]
        enhance_json_schema: bool,

        /// HuggingFace model repository ID
        #[arg(long, env = "BOOKMARKS_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// GGUF model filename
        #[arg(long, env = "BOOKMARKS_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "BOOKMARKS_MODEL_CACHE_DIR")]
        cache_dir: Option<PathBuf>,

        /// Maximum retry attempts for schema generation
        #[arg(long, default_value = "2")]
        max_attempts: usize,
    },
}

impl Cli {
    pub async fn execute(self) -> Result<()> {
        match self.command {
            Commands::Gen { subcommand } => match subcommand {
                GenCommands::Schema {
                    input,
                    output,
                    json_schema_output,
                    no_json_schema,
                    enhance_json_schema,
                    repo_id,
                    filename,
                    cache_dir,
                    max_attempts,
                } => {
                    generate_schema(
                        input,
                        output,
                        json_schema_output,
                        no_json_schema,
                        enhance_json_schema,
                        repo_id,
                        filename,
                        cache_dir,
                        max_attempts,
                    )
                    .await
                }
            },
        }
    }
}

async fn generate_schema(
    input: PathBuf,
    output: PathBuf,
    json_schema_output: Option<PathBuf>,
    no_json_schema: bool,
    enhance_json_schema: bool,
    repo_id: Option<String>,
    filename: Option<String>,
    cache_dir: Option<PathBuf>,
    max_attempts: usize,
) -> Result<()> {
    use backbone::dag::{Dag, DagExecutor, Task, TaskContext, TaskOutput};
    use backbone::dag::task::{JsonSchemaGenerationTask, SqlGenerationTask};
    use backbone::llm::{ensure_gguf_model, LlamaModel, ModelConfig};
    use backbone::schema::{derive_field_hints, infer_json_schema};
    use backbone::tracing::init_tracing;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tracing::Instrument;

    // initialize tracing
    let _guard = init_tracing("bbt")?;

    tracing::info!("loading bookmarks from {}", input.display());

    // load and infer schema (sequential phase)
    let schema_span = tracing::info_span!("schema_inference");
    let (schema, hints) = async {
        let json_text = std::fs::read_to_string(&input)?;
        let payload: serde_json::Value = serde_json::from_str(&json_text)?;

        let item_count = if payload.is_array() {
            payload.as_array().map(|a| a.len()).unwrap_or(0)
        } else {
            1
        };
        tracing::info!("loaded {} top-level items", item_count);

        tracing::info!("inferring json schema from payload structure");
        let schema = infer_json_schema(&payload);
        let hints = derive_field_hints(&schema);

        // log detected hints
        let mut hint_summary = Vec::new();
        if !hints.timestamps.is_empty() {
            hint_summary.push(format!("{} timestamp fields", hints.timestamps.len()));
        }
        if !hints.folders.is_empty() {
            hint_summary.push(format!("{} folder/tag fields", hints.folders.len()));
        }
        if !hint_summary.is_empty() {
            tracing::info!("detected: {}", hint_summary.join(", "));
        }

        Ok::<_, anyhow::Error>((schema, hints))
    }
    .instrument(schema_span)
    .await?;

    // derive json schema output path if not specified
    let json_schema_path = if no_json_schema {
        None
    } else {
        Some(json_schema_output.unwrap_or_else(|| {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("bookmarks");
            input.with_file_name(format!("{}.schema.json", stem))
        }))
    };

    // download/load model
    tracing::info!("ensuring model is available");
    let model_path = ensure_gguf_model(repo_id, filename, cache_dir)?;

    tracing::info!("loading language model from {}", model_path.display());
    let model = Arc::new(LlamaModel::new(model_path, ModelConfig::default())?);
    tracing::info!("model loaded successfully");

    // build dag
    let mut dag = Dag::new();
    dag.add_task("sql_generation".to_string(), vec![]);
    if json_schema_path.is_some() {
        dag.add_task("json_schema_generation".to_string(), vec![]);
    }

    // create task implementations
    let mut tasks: HashMap<String, Arc<dyn Task>> = HashMap::new();
    tasks.insert(
        "sql_generation".to_string(),
        Arc::new(SqlGenerationTask),
    );
    if json_schema_path.is_some() {
        tasks.insert(
            "json_schema_generation".to_string(),
            Arc::new(JsonSchemaGenerationTask {
                enhance_with_llm: enhance_json_schema,
            }),
        );
    }

    // prepare task context
    let ctx = TaskContext {
        schema,
        hints,
        model,
        max_attempts,
    };

    // execute dag
    tracing::info!("invoking llm to generate schemas (this may take a while)");
    let executor = DagExecutor::new(dag)?;
    let results = executor.execute(tasks, ctx).await?;

    // write outputs
    if let Some(sql_result) = results.get("sql_generation") {
        match &sql_result.output {
            TaskOutput::Sql(sql) => {
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&output, sql)?;
                tracing::info!(
                    output = %output.display(),
                    duration_ms = sql_result.duration_ms,
                    "wrote sql schema"
                );
            }
            _ => {
                return Err(anyhow::anyhow!("unexpected output type from sql generation"));
            }
        }
    }

    if let Some(json_path) = json_schema_path {
        if let Some(json_result) = results.get("json_schema_generation") {
            match &json_result.output {
                TaskOutput::JsonSchema(json) => {
                    if let Some(parent) = json_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&json_path, json)?;
                    tracing::info!(
                        output = %json_path.display(),
                        duration_ms = json_result.duration_ms,
                        "wrote json schema"
                    );
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "unexpected output type from json schema generation"
                    ));
                }
            }
        }
    }

    Ok(())
}
