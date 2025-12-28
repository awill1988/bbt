use crate::git::models::CommitClassification;
use regex::Regex;
use once_cell::sync::Lazy;

/// regex for conventional commit format: type(scope)?: subject
static CONVENTIONAL_COMMIT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(feat|fix|refactor|docs|test|chore|style|perf|ci|build|revert)(\(.+\))?!?:")
        .expect("invalid conventional commit regex")
});

/// classify a commit based on its message
pub fn classify_commit(message: &str) -> CommitClassification {
    let first_line = message.lines().next().unwrap_or("");

    // try conventional commit pattern first
    if let Some(captures) = CONVENTIONAL_COMMIT_PATTERN.captures(first_line) {
        let commit_type = captures.get(1).map(|m| m.as_str()).unwrap_or("");
        return CommitClassification::from_str(commit_type);
    }

    // fallback to keyword matching
    classify_by_keywords(message)
}

/// classify commit by keywords in the message
fn classify_by_keywords(message: &str) -> CommitClassification {
    let lower = message.to_lowercase();

    // feature keywords
    if lower.contains("add") || lower.contains("implement") || lower.contains("feature") {
        return CommitClassification::Feat;
    }

    // fix keywords
    if lower.contains("fix") || lower.contains("bug") || lower.contains("patch") || lower.contains("hotfix") {
        return CommitClassification::Fix;
    }

    // refactor keywords
    if lower.contains("refactor") || lower.contains("restructure") || lower.contains("clean up") {
        return CommitClassification::Refactor;
    }

    // docs keywords
    if lower.contains("docs") || lower.contains("documentation") || lower.contains("readme") {
        return CommitClassification::Docs;
    }

    // test keywords
    if lower.contains("test") {
        return CommitClassification::Test;
    }

    // chore keywords
    if lower.contains("chore") || lower.contains("maintenance") || lower.contains("deps") || lower.contains("dependency") {
        return CommitClassification::Chore;
    }

    // style keywords
    if lower.contains("style") || lower.contains("format") || lower.contains("lint") {
        return CommitClassification::Style;
    }

    // performance keywords
    if lower.contains("perf") || lower.contains("performance") || lower.contains("optimize") {
        return CommitClassification::Perf;
    }

    // ci keywords
    if lower.contains("ci") || lower.contains("github actions") || lower.contains("workflow") {
        return CommitClassification::Ci;
    }

    // build keywords
    if lower.contains("build") || lower.contains("compile") || lower.contains("cargo") {
        return CommitClassification::Build;
    }

    // revert keywords
    if lower.contains("revert") {
        return CommitClassification::Revert;
    }

    // default to unknown
    CommitClassification::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conventional_commit_classification() {
        assert_eq!(classify_commit("feat: add new feature"), CommitClassification::Feat);
        assert_eq!(classify_commit("fix: resolve bug"), CommitClassification::Fix);
        assert_eq!(classify_commit("refactor: clean up code"), CommitClassification::Refactor);
        assert_eq!(classify_commit("docs: update readme"), CommitClassification::Docs);
        assert_eq!(classify_commit("test: add unit tests"), CommitClassification::Test);
        assert_eq!(classify_commit("chore: update dependencies"), CommitClassification::Chore);
        assert_eq!(classify_commit("style: format code"), CommitClassification::Style);
        assert_eq!(classify_commit("perf: optimize query"), CommitClassification::Perf);
        assert_eq!(classify_commit("ci: update workflow"), CommitClassification::Ci);
        assert_eq!(classify_commit("build: update cargo"), CommitClassification::Build);
        assert_eq!(classify_commit("revert: undo changes"), CommitClassification::Revert);
    }

    #[test]
    fn test_conventional_commit_with_scope() {
        assert_eq!(classify_commit("feat(api): add endpoint"), CommitClassification::Feat);
        assert_eq!(classify_commit("fix(auth): resolve login issue"), CommitClassification::Fix);
    }

    #[test]
    fn test_conventional_commit_with_breaking() {
        assert_eq!(classify_commit("feat!: breaking change"), CommitClassification::Feat);
        assert_eq!(classify_commit("fix(api)!: breaking fix"), CommitClassification::Fix);
    }

    #[test]
    fn test_keyword_classification() {
        assert_eq!(classify_commit("Add new API endpoint"), CommitClassification::Feat);
        assert_eq!(classify_commit("Fix memory leak"), CommitClassification::Fix);
        assert_eq!(classify_commit("Refactor authentication module"), CommitClassification::Refactor);
        assert_eq!(classify_commit("Update documentation for API"), CommitClassification::Docs);
        assert_eq!(classify_commit("Add test for user service"), CommitClassification::Test);
        assert_eq!(classify_commit("Update dependencies"), CommitClassification::Chore);
        assert_eq!(classify_commit("Format code with rustfmt"), CommitClassification::Style);
        assert_eq!(classify_commit("Optimize database queries"), CommitClassification::Perf);
        assert_eq!(classify_commit("Update GitHub Actions workflow"), CommitClassification::Ci);
        assert_eq!(classify_commit("Update build script"), CommitClassification::Build);
    }

    #[test]
    fn test_unknown_classification() {
        assert_eq!(classify_commit("Random commit message"), CommitClassification::Unknown);
        assert_eq!(classify_commit("Merge branch 'main'"), CommitClassification::Unknown);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(classify_commit("FIX: Resolve Bug"), CommitClassification::Fix);
        assert_eq!(classify_commit("FEAT: Add Feature"), CommitClassification::Feat);
    }
}
