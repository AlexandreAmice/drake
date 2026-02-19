# ISSUE #22226 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/22226
Title: Provide ability to call SolveInParallel with program generators.
Owner branch: issue-22226-feature
Last updated: 2026-02-19 02:32:37 UTC
Status: Implemented and validated

## Problem Statement
`SolveInParallel` currently requires materializing vectors of program pointers and optional metadata up front. This is memory-heavy for large workloads and pushes callers to duplicate parallel-solve logic that should stay centralized.

## Scope Decision
- In scope:
- Add a public generator-based C++ overload for `SolveInParallel` with index range + optional metadata generators.
- Preserve key semantics from existing implementation: thread-local solver cache, `kMaxThreads` rewrite, serial fallback for non-thread-safe programs.
- Add tests for basic correctness, null-program skip semantics, optional generator plumbing, and range validation.
- Out of scope:
- Python bindings.
- Refactoring all existing callsites.

## Implemented API
Added overload in `solvers/solve.h`:
- `SolveInParallel(prog_generator, range_start, range_end, initial_guesses_generator, solver_options_generator, solver_ids_generator, parallelism, dynamic_schedule)`
- `prog_generator(thread_num, i)` returns `const MathematicalProgram*`.
- Returning `nullptr` skips that index and leaves result as `kSolutionResultNotSet`.

## Success Criteria
- New overload compiles and passes `//solvers:solve_in_parallel_test`.
- Existing solve behavior remains intact (`//solvers:solve_test` passes).
- Range validation and optional generator usage are tested.

## Implementation Plan
1. Add new overload declaration + docs in `solvers/solve.h`.
2. Implement overload in `solvers/solve.cc` using the same solving semantics as existing code paths.
3. Add tests in `solvers/test/solve_in_parallel_test.cc` for:
- basic generator solve,
- null-program skip,
- optional generator invocation,
- invalid range throw.
4. Run `solve_in_parallel_test` and `solve_test`.
5. Commit with explicit Codex feature implementation note.

## Test Plan
- `bazel test //solvers:solve_in_parallel_test --test_output=errors`
- `bazel test //solvers:solve_test --test_output=errors`

## Risks / Open Questions
- Returned program pointer lifetime remains caller-owned; this is documented but still an API contract footgun if violated.

## Execution Log
- 2026-02-19 00:00:00 UTC: Created initial plan before code edits.
- 2026-02-19 02:20:00 UTC: Implemented generator overload in `solvers/solve.h` and `solvers/solve.cc` with `nullptr => skip` semantics.
- 2026-02-19 02:23:00 UTC: Added coverage in `solvers/test/solve_in_parallel_test.cc` for generator basic path, null skip, optional generators, and range validation.
- 2026-02-19 02:24:00 UTC: Ran `clang-format` on touched C++ files.
- 2026-02-19 02:28:00 UTC: Ran `bazel test //solvers:solve_in_parallel_test --test_output=errors`; passed.
- 2026-02-19 02:29:00 UTC: Ran `bazel test //solvers:solve_test --test_output=errors`; passed.
- 2026-02-19 02:36:00 UTC: Detected accidental duplicate generator API declarations during cleanup pass; consolidated to a single overload signature and removed duplicate implementation path.
- 2026-02-19 02:38:00 UTC: Updated generator tests to match final signature and confirmed `nullptr => skip` semantics (`kSolutionResultNotSet`) rather than throw.
- 2026-02-19 02:39:00 UTC: Re-ran `bazel test //solvers:solve_in_parallel_test //solvers:solve_test --test_output=errors`; both passed.
