use crate::dag::graph::Dag;
use crate::dag::task::{Task, TaskContext, TaskResult};
use crate::error::{BbtError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::Instrument;

/// executes a dag of tasks with parallel execution at each level
pub struct DagExecutor {
    dag: Dag,
}

impl DagExecutor {
    /// create a new executor for the given dag
    pub fn new(dag: Dag) -> Result<Self> {
        dag.validate()?;
        Ok(Self { dag })
    }

    /// execute all tasks in the dag, parallelizing tasks at each level
    pub async fn execute(
        &self,
        tasks: HashMap<String, Arc<dyn Task>>,
        ctx: TaskContext,
    ) -> Result<HashMap<String, TaskResult>> {
        let task_count = tasks.len();
        let levels = self.dag.topological_levels()?;
        let mut results = HashMap::new();

        let span = tracing::info_span!("dag_executor::execute", dag.task_count = task_count);
        let _enter = span.enter();

        tracing::info!(
            total_tasks = task_count,
            levels = levels.len(),
            "executing dag"
        );

        for (level_idx, task_ids) in levels.iter().enumerate() {
            let _level_span = tracing::info_span!(
                "dag_level",
                dag.level = level_idx,
                dag.level_task_count = task_ids.len()
            )
            .entered();

            tracing::info!("starting level {} with {} tasks", level_idx, task_ids.len());

            let mut join_set = JoinSet::new();

            for task_id in task_ids {
                let task = Arc::clone(tasks
                    .get(task_id)
                    .ok_or_else(|| {
                        BbtError::Schema(format!("task not found: {}", task_id))
                    })?);

                let task_ctx = ctx.clone();
                let task_id_owned = task_id.clone();

                // each task gets its own span for observability
                let task_span = tracing::info_span!(
                    "task",
                    task.id = %task_id_owned,
                    otel.kind = "internal"
                );

                // spawn task execution
                let task_future = async move {
                    let start = std::time::Instant::now();

                    tracing::info!("executing task");
                    let output = task.execute(task_ctx).await?;

                    let duration_ms = start.elapsed().as_millis() as u64;
                    tracing::info!(
                        task.duration_ms = duration_ms,
                        "task completed successfully"
                    );

                    Ok::<_, BbtError>(TaskResult {
                        task_id: task_id_owned,
                        output,
                        duration_ms,
                    })
                }
                .instrument(task_span);

                join_set.spawn(task_future);
            }

            // collect all results from this level
            while let Some(result) = join_set.join_next().await {
                let task_result = result.map_err(|e| {
                    BbtError::Schema(format!("task join error: {}", e))
                })??;

                results.insert(task_result.task_id.clone(), task_result);
            }

            tracing::info!("level {} completed", level_idx);
        }

        tracing::info!(completed_tasks = results.len(), "dag execution complete");
        Ok(results)
    }
}
