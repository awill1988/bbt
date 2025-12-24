use crate::agent::executor::generate_schema_sql_with_retry;
use crate::error::Result;
use crate::llm::model::LlamaModel;
use crate::schema::hints::FieldHints;
use crate::schema::inference::Schema;
use crate::schema::json_schema::to_json_schema;
use async_trait::async_trait;
use std::sync::Arc;

/// context passed to each task during execution
#[derive(Clone)]
pub struct TaskContext {
    pub schema: Schema,
    pub hints: FieldHints,
    pub model: Arc<LlamaModel>,
    pub max_attempts: usize,
}

/// output from a task execution
#[derive(Debug, Clone)]
pub enum TaskOutput {
    Sql(String),
    JsonSchema(String),
}

/// result of a task execution including metadata
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub output: TaskOutput,
    pub duration_ms: u64,
}

/// trait for tasks that can be executed in the dag
#[async_trait]
pub trait Task: Send + Sync {
    /// unique identifier for this task
    fn id(&self) -> &str;

    /// execute the task with given context
    async fn execute(&self, ctx: TaskContext) -> Result<TaskOutput>;
}

/// task that generates sql schema using llm
pub struct SqlGenerationTask;

#[async_trait]
impl Task for SqlGenerationTask {
    fn id(&self) -> &str {
        "sql_generation"
    }

    #[tracing::instrument(skip(self, ctx), fields(task.id = %self.id(), task.type = "sql_generation"))]
    async fn execute(&self, ctx: TaskContext) -> Result<TaskOutput> {
        tracing::info!("starting sql schema generation");

        let sql = generate_schema_sql_with_retry(
            &ctx.model,
            &ctx.schema,
            &ctx.hints,
            ctx.max_attempts,
        )
        .await?;

        tracing::info!(sql_length = sql.len(), "sql schema generated");
        Ok(TaskOutput::Sql(sql))
    }
}

/// task that generates json schema (rule-based with optional llm enhancement)
pub struct JsonSchemaGenerationTask {
    pub enhance_with_llm: bool,
}

#[async_trait]
impl Task for JsonSchemaGenerationTask {
    fn id(&self) -> &str {
        "json_schema_generation"
    }

    #[tracing::instrument(skip(self, ctx), fields(task.id = %self.id(), task.type = "json_schema_generation", enhance_with_llm = self.enhance_with_llm))]
    async fn execute(&self, ctx: TaskContext) -> Result<TaskOutput> {
        tracing::info!("starting json schema generation");

        // rule-based conversion
        let json_schema_value = to_json_schema(&ctx.schema, Some("bookmarks"))?;

        // optional llm enhancement
        let final_schema = if self.enhance_with_llm {
            tracing::info!("enhancing json schema with llm");
            crate::agent::json_schema_enhancer::enhance_json_schema(
                &ctx.model,
                &json_schema_value,
                &ctx.hints,
            )
            .await?
        } else {
            json_schema_value
        };

        let json_string = serde_json::to_string_pretty(&final_schema)?;
        tracing::info!(
            json_length = json_string.len(),
            enhanced = self.enhance_with_llm,
            "json schema generated"
        );

        Ok(TaskOutput::JsonSchema(json_string))
    }
}
