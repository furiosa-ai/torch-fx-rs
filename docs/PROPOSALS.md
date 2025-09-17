# Proposals and Follow-ups

This document tracks additional improvement proposals discussed during recent changes.

## 1) Node-based APIs to avoid name copies

Provide reference-returning variants that avoid copying `String` names:

- `Graph::flatten_node_args_nodes(&self, node_name: impl AsRef<str>) -> PyResult<Option<Vec<&Node>>>>`
- `Graph::users_nodes(&self, node_name: impl AsRef<str>) -> PyResult<Option<Vec<&Node>>>>`

Rationale: current APIs return `Vec<String>` and copy names. The proposed variants keep Python references and reduce allocation, matching the zero-copy design goal.

## 2) Validate `Op` mapping for call_method/call_module

In `src/fx/mod.rs`, confirm the mapping between Python `op` strings and the `Op` enum. Today, `"call_method"` maps to `Op::CallModule` and `"call_module"` maps to `Op::CallMethod`. If this is intentional (compat shimming), document it; otherwise, correct and add tests.

## 3) Document safety and usage of TensorView

Add README/API docs with examples for:

- Borrowing buffer/parameter bytes: `get_*_view` and `extract_*_view`
- Lifetime constraints: the view is tied to the GIL token and should not be stored beyond its scope
- Read-only semantics: do not mutate the underlying tensor while viewing

## 4) Optional typed views

Consider augmenting `TensorView` to carry element size/dtype and logical extents. For strided views, expose both storage range and logical shape/stride (possibly via `TensorMeta`) to let consumers reason about layout without copying.

## 5) Iterator-based collection APIs

Add iterator-returning variants to avoid building `HashMap`s when not needed:

- `GraphModule::iter_parameters_view<'py>(&self, py: Python<'py>) -> PyResult<impl Iterator<Item=(String, TensorView<'py>)>>`
- `GraphModule::iter_buffers_view<'py>(&self, py: Python<'py>) -> PyResult<impl Iterator<Item=(String, TensorView<'py>)>>`

Rationale: reduce intermediate allocations for large models.

## 6) Tests for view APIs

Add dedicated tests for `get_*_view`/`extract_*_view`, verifying length and content, and that views cannot outlive the GIL scope.

