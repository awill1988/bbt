use crate::error::{BbtError, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// directed acyclic graph for task orchestration
#[derive(Debug, Clone)]
pub struct Dag {
    /// task_id -> list of task_ids it depends on
    dependencies: HashMap<String, Vec<String>>,
}

impl Dag {
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    /// add a task with its dependencies
    pub fn add_task(&mut self, id: String, dependencies: Vec<String>) {
        self.dependencies.insert(id, dependencies);
    }

    /// validate the dag (check for cycles and missing dependencies)
    pub fn validate(&self) -> Result<()> {
        // check for missing dependencies
        for (task_id, deps) in &self.dependencies {
            for dep in deps {
                if !self.dependencies.contains_key(dep) {
                    return Err(BbtError::Schema(format!(
                        "task '{}' depends on non-existent task '{}'",
                        task_id, dep
                    )));
                }
            }
        }

        // check for cycles using depth-first search
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for task_id in self.dependencies.keys() {
            if !visited.contains(task_id) {
                if self.has_cycle(task_id, &mut visited, &mut rec_stack)? {
                    return Err(BbtError::Schema(
                        "dag contains a cycle".to_string()
                    ));
                }
            }
        }

        Ok(())
    }

    /// detect cycles using dfs
    fn has_cycle(
        &self,
        task_id: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> Result<bool> {
        visited.insert(task_id.to_string());
        rec_stack.insert(task_id.to_string());

        if let Some(deps) = self.dependencies.get(task_id) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle(dep, visited, rec_stack)? {
                        return Ok(true);
                    }
                } else if rec_stack.contains(dep) {
                    return Ok(true);
                }
            }
        }

        rec_stack.remove(task_id);
        Ok(false)
    }

    /// returns tasks grouped by execution level using kahn's algorithm
    /// level 0: tasks with no dependencies
    /// level 1: tasks that depend only on level 0 tasks
    /// etc.
    pub fn topological_levels(&self) -> Result<Vec<Vec<String>>> {
        // calculate in-degree for each task
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for task_id in self.dependencies.keys() {
            in_degree.entry(task_id.clone()).or_insert(0);
        }

        for deps in self.dependencies.values() {
            for dep in deps {
                *in_degree.entry(dep.clone()).or_insert(0) += 1;
            }
        }

        // start with tasks that have no dependencies (in-degree = 0)
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut levels = Vec::new();
        let mut processed = HashSet::new();

        while !queue.is_empty() {
            // process all tasks at current level in parallel
            let level_size = queue.len();
            let mut current_level = Vec::new();

            for _ in 0..level_size {
                if let Some(task_id) = queue.pop_front() {
                    current_level.push(task_id.clone());
                    processed.insert(task_id.clone());

                    // reduce in-degree of tasks that depend on this one
                    for (id, deps) in &self.dependencies {
                        if deps.contains(&task_id) {
                            let degree = in_degree.get_mut(id).unwrap();
                            *degree -= 1;
                            if *degree == 0 && !processed.contains(id) {
                                queue.push_back(id.clone());
                            }
                        }
                    }
                }
            }

            if !current_level.is_empty() {
                levels.push(current_level);
            }
        }

        // verify all tasks were processed
        if processed.len() != self.dependencies.len() {
            return Err(BbtError::Schema(
                "dag contains unreachable tasks or cycles".to_string(),
            ));
        }

        Ok(levels)
    }
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dag() {
        let mut dag = Dag::new();
        dag.add_task("a".to_string(), vec![]);
        dag.add_task("b".to_string(), vec![]);

        assert!(dag.validate().is_ok());

        let levels = dag.topological_levels().unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 2);
    }

    #[test]
    fn test_dag_with_dependencies() {
        let mut dag = Dag::new();
        dag.add_task("a".to_string(), vec![]);
        dag.add_task("b".to_string(), vec!["a".to_string()]);
        dag.add_task("c".to_string(), vec!["a".to_string()]);

        assert!(dag.validate().is_ok());

        let levels = dag.topological_levels().unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0], vec!["a"]);
        assert_eq!(levels[1].len(), 2);
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = Dag::new();
        dag.add_task("a".to_string(), vec!["b".to_string()]);
        dag.add_task("b".to_string(), vec!["a".to_string()]);

        assert!(dag.validate().is_err());
    }

    #[test]
    fn test_missing_dependency() {
        let mut dag = Dag::new();
        dag.add_task("a".to_string(), vec!["nonexistent".to_string()]);

        assert!(dag.validate().is_err());
    }
}
