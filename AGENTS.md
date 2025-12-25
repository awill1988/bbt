# Agent operating instructions (performance-critical)

- **Honor existing versions first**: Scan `Cargo.toml` and `Cargo.lock` before suggesting or pinning dependencies. Never invent a new pin if a version already exists in the lockfile. Use `cargo update` with specific package versions when upgrading.
- **Upgrade discipline**:
  - Read the full changelog/release notes for the target crate version. Treat it as the source of truth for breaking changes.
  - Map API changes: Use `cargo check` and the compiler's guidance to identify and fix functions, traits, and structs affected by the upgrade.
  - Optimize for catching regressions; if any ambiguity or possible breaking change remains, stop and ask for user feedback before proceeding.

## Idiomatic Rust Development

- **Error Handling**: Embrace `Result<T, E>` for recoverable errors and `panic!` for unrecoverable errors. Use `?` operator to propagate errors. Use `anyhow` for application-level error handling and `thiserror` for library-level error types.
- **Data Structures**: Use `enum`s to model states and different kinds of data. `Option<T>` should be used when a value can be absent.
- **Resource Management**: Rely on Rust's ownership and RAII (Resource Acquisition Is Initialization) for managing resources like files and network connections. When a variable goes out of scope, its destructor is called and the resource is freed.
- **Concurrency**: Use channels (`std::sync::mpsc` or `tokio::sync::mpsc`) for message passing between threads. Use `Arc<Mutex<T>>` for shared state, but prefer message passing when possible.
- **Performance**: Use iterators and their combinators for efficient data processing. Avoid unnecessary allocations. Profile hot paths when performance is critical.
- **Tooling**: Run `cargo clippy` regularly to get suggestions on improving code to be more correct and idiomatic. Use `cargo fmt` to maintain a consistent code style.

## Embeddings and temporal features

- **Goal**: Accurate document organization and retrieval via clustering.
- **Approach**: Concatenate temporal features to text embeddings for automatic temporal + semantic grouping.
- **Missing dates**: Use median timestamp or a neutral value (0.5 normalized) to avoid null-date clustering.
- **Implementation**: For performance-critical tasks like embedding generation, leverage the existing ONNX runtime via the `ort` crate.
- **Rationale**: Placement accuracy over interpretability; clustering naturally balances content similarity with temporal proximity.