# AGENTS.md

This file gives agents practical guidance for working in this repository.

Scope: The scope of this AGENTS.md is the entire repository unless a more deeply nested AGENTS.md overrides any rule.

## Purpose
- Maintain consistent patterns for PyO3 wrappers around `torch.fx` types.
- Keep public APIs aligned with PyTorch `torch.fx` semantics while remaining idiomatic Rust.
- Ensure formatting, linting, and tests pass locally and in CI.

## Repo Overview
- Crate: `torch-fx-rs` — Rust APIs to interoperate with PyTorch `torch.fx` Graphs and GraphModules via PyO3.
- Re-exports from `src/lib.rs`:
  - `Graph` (wrapper for `torch.fx.Graph`): `src/fx/graph.rs`
  - `GraphModule` (wrapper for `torch.fx.GraphModule`): `src/fx/graph_module.rs`
  - `Node` (wrapper for `torch.fx.Node`): `src/fx/node.rs`
  - `CustomFn` and `FunctionWrapper` (Python-callable wrapper for Rust functions): `src/fx/custom_fn.rs`
  - `types` (FX-related enums and conversions: `Op`, `Target`, `Argument`, `TensorMeta`, `Dtype`, `Layout`, `Device`, `MemoryFormat`): `src/fx/mod.rs`
- Integration tests: `tests/fx.rs` (runs against Python + PyTorch).

## Tooling & Versions
- Rust edition: 2021
- Dependencies: `pyo3 = 0.19.1`, `indexmap = 1.9.2`, `tracing = 0.1.37`
- Python: CI uses Python 3.10 with `torch` from pip.
- PyO3: You can pin the interpreter via `PYO3_PYTHON=/path/to/python3.10`.

## Build, Lint, Test
- Format + Lint (CI-enforced):
  - `cargo fmt -- --check`
  - `cargo clippy --all-targets -- --deny warnings`
- Tests (requires Python + PyTorch installed):
  - Ensure `torch` is installed in the active Python: `pip install torch`
  - If multiple Pythons exist, use `PYO3_PYTHON=$(which python3.10)`
  - Run: `cargo test`
- Docs:
  - `cargo doc --open`

## Coding Conventions
- Style and hygiene:
  - Keep code formatted with `rustfmt`; CI will fail otherwise.
  - Keep clippy clean with `--deny warnings`.
  - Prefer `?` for error propagation; return `PyResult<T>` for Python-facing fallible APIs.
- Public API surface:
  - Mirror `torch.fx` semantics where feasible; document any divergence in rustdoc.
  - Maintain Rust naming conventions (snake_case) while staying close to Python behavior.
  - Update `README.md` API sections when adding or changing public items.
- Wrapper pattern (Graph/GraphModule/Node):
  - `#[repr(transparent)]` wrapper around `PyAny`.
  - Implement: `PyNativeType`, `ToPyObject`, `AsRef<PyAny>`, `AsPyPointer`, `Deref<Target = PyAny>`.
  - Provide `IntoPy<Py<...>> for &...`, `From<&...> for Py<...>`, `From<&...> for &PyAny`, `PyTypeInfo` with `MODULE = Some("torch.fx")`.
  - Implement `FromPyObject<'_> for &'_ ...` by downcasting.
- Conversions and types (`src/fx/mod.rs`):
  - For `Op`, `Target`, `Argument`, `Dtype`, `Layout`, `Device`, `MemoryFormat`, `TensorMeta`, provide `FromPyObject` and `IntoPy` as used.
  - Validate Python types using `__class__`, `__module__`, and `repr` patterns as already established.
- Error handling:
  - Use `PyResult` and return informative `PyErr`s for interop failures; avoid panics.
  - In internal invariants mirroring Python semantics, following existing code, occasional `.unwrap()` is acceptable but prefer explicit error paths.
- Logging:
  - Use `tracing` for debug logs. Avoid new logging dependencies.
- Dependencies:
  - Keep dependencies minimal; introduce new crates only with clear justification.

## Python Interop (PyO3)
- GIL:
  - Hold the GIL for all Python API calls (`Python::with_gil`, etc.).
- Object creation:
  - Use `__new__` then `__init__` patterns for Python classes, as in current wrappers.
- Lifetimes:
  - Constructors return GIL-bound references (e.g., `&Graph`, `&GraphModule`, `&Node`), not owned values.
- Custom callables:
  - `CustomFn` is a `#[pyclass]` that forwards to `FunctionWrapper = Arc<dyn Fn(&PyTuple, Option<&PyDict>) -> PyResult<PyObject> + Send + Sync>`.

## Safety Notes
- Zero-copy tensor views:
  - `GraphModule` exposes `extract_parameters_view`, `extract_buffers_view`, `get_parameter_view`, `get_buffer_view` which return `BufferView<'py>`.
  - `BufferView` ties the borrow to the GIL token and implements `Deref<Target = [u8]>`; do not store beyond the GIL scope and avoid mutating tensors while viewing. If persistence is needed, copy the data via `&*view`.

## Adding New Functionality
- New wrappers:
  - Follow the exact pattern used by `Graph`, `GraphModule`, `Node`.
- New `Graph`/`Node` methods:
  - Mirror torch.fx APIs; convert inputs/outputs via existing `Argument`/`Target` helpers.
  - Return GIL-bound references where appropriate.
- Extending `types`:
  - When adding new enum variants or FX concepts, add both `FromPyObject` and `IntoPy`. Update helper functions consistently.
- Tests:
  - Add targeted tests in `tests/fx.rs` that exercise Python interop and behavior. Use `pyo3::prepare_freethreaded_python()` and `Python::with_gil` per existing tests.

## CI Expectations
- Workflow at `.github/workflows/rust.yml` runs:
  - `cargo fmt -- --check`
  - `cargo clippy --all-targets -- --deny warnings`
  - Sets up Python 3.10 and installs `torch`
  - `cargo test`
- Ensure all of the above pass locally before opening a PR.

## Commit & PR Rules
- Follow cbea.ms/git-commit style for messages:
  - Separate subject from body with a blank line
  - Subject in imperative mood, ≤ 50 chars, no trailing period
  - Wrap body at ~72 chars; explain what and why, not how
  - Prefer present tense and active voice; avoid filler
- Commit granularity:
  - One logical change per commit; keep commits small and focused
  - Ensure each commit builds and tests pass; avoid WIP commits
  - Do not mix formatting-only changes with functional changes
  - Keep unrelated refactors out of the same commit
  - Order related commits logically (e.g., CI/config → docs → API → tests)
- Message patterns (examples, adapt as needed):
  - "Use Python 3.10 in CI"
  - "Introduce BufferView for safe zero-copy access"
  - "Replace slice-returning APIs with *_view variants"
  - "Add Node::meta_pydict for zero-copy meta access"
- PR guidelines:
  - Title mirrors commit subject style; be concise and imperative
  - Body summarizes what changed and why, lists breaking changes and migration
  - Mention test coverage and docs updates; link related issues
  - Keep PRs reasonably scoped; prefer a small series of focused commits

## Project Layout
- `src/lib.rs`: Crate entry (re-exports).
- `src/fx/*.rs`: Core implementation modules.
- `tests/fx.rs`: Integration tests against Python/torch.
- `README.md`: High-level API overview and links.

## Do / Don’t
- Do:
  - Keep changes minimal, targeted, and consistent with existing patterns.
  - Add rustdoc with links to the corresponding `torch.fx` docs for new APIs.
  - Update `README.md` when public APIs change.
- Don’t:
  - Introduce unrelated refactors or dependencies.
  - Change wrapper patterns without strong rationale.
  - Commit formatting or clippy violations.
  - Add license headers unless specifically requested.

## Common Commands
- Full local check: `cargo fmt && cargo clippy --all-targets -- --deny warnings && cargo test`
- Pin Python for PyO3: `PYO3_PYTHON=$(which python3.10) cargo test`
- Open docs: `cargo doc --open`
