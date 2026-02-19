# ISSUE #22226 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/22226
Title: Provide ability to call SolveInParallel with program generators.
Owner branch: issue-22226-feature
Last updated: 2026-02-19 02:42:47 UTC
Status: Ready for PR

## Problem Statement
`SolveInParallel` currently requires materializing vectors of program pointers and optional metadata up front. This is memory-heavy for large workloads and encourages reimplementation of solver-parallelism details in downstream code.

## Scope Decision
- In scope:
- Add public generator-based `SolveInParallel` C++ overload with index range and optional metadata generators.
- Preserve existing semantics from vector-based API: per-thread solver cache, `CommonSolverOption::kMaxThreads` rewrite, serial fallback for non-thread-safe programs.
- Route existing vector overload through the generator implementation to avoid behavior drift.
- Add unit tests for generator correctness and invalid input handling.
- Out of scope:
- Python bindings for generator API.
- Refactoring existing callsites.

## Implemented API
Added callback aliases in `solvers/solve.h`:
- `SolveInParallelProgramGenerator`
- `SolveInParallelInitialGuessGenerator`
- `SolveInParallelSolverOptionsGenerator`
- `SolveInParallelSolverIdGenerator`

Added overload:
- `SolveInParallel(prog_generator, initial_guess_generator, solver_options_generator, solver_id_generator, range_start, range_end, parallelism, dynamic_schedule)`

Semantics:
- `range_start <= range_end` is required.
- Program generator must return non-null (null now throws, same as vector API behavior).
- Optional metadata generators may return null / nullopt.

## Success Criteria
- New overload compiles and passes `//solvers:solve_in_parallel_test`.
- Existing vector overload remains behavior-compatible via delegation.
- Invalid range and null-program behavior are tested.

## Implementation Plan
1. Add generator aliases and overload declaration in `solvers/solve.h`.
2. Implement shared generator-based execution in `solvers/solve.cc`.
3. Delegate vector overload to generator overload via index adapter lambdas.
4. Add/update tests in `solvers/test/solve_in_parallel_test.cc`.
5. Run targeted tests and iterate.

## Test Plan
- `bazel test //solvers:solve_in_parallel_test --test_output=errors`

## Risks / Open Questions
- Generator callback lifetime contract remains caller responsibility for returned program pointers.

## Execution Log
- 2026-02-19 00:00:00 UTC: Created initial plan before code edits.
- 2026-02-19 02:30:00 UTC: Added generator callback aliases and overload declaration in `solvers/solve.h`.
- 2026-02-19 02:33:00 UTC: Implemented generator-based solve path in `solvers/solve.cc`; delegated vector overload through generator adapters.
- 2026-02-19 02:36:00 UTC: Updated `solvers/test/solve_in_parallel_test.cc` for generator-overload equivalence and invalid-input throws.
- 2026-02-19 02:38:00 UTC: Ran `bazel test //solvers:solve_in_parallel_test --test_output=errors` (pass).
