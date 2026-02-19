# ISSUE #22226 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/22226
Title: Provide ability to call SolveInParallel with program generators.
Owner branch: issue-22226-feature
Fork PR: https://github.com/AlexandreAmice/drake/pull/93
Prior upstream PR + feedback: https://github.com/RobotLocomotion/drake/pull/22225
Last updated: 2026-02-19 03:08:00 UTC
Status: Ready for review

## Problem Statement
`SolveInParallel` currently requires materializing vectors of program pointers and optional metadata up front. This is memory-heavy for large workloads and encourages reimplementation of solver-parallelism details in downstream code.

## Scope Decision
- In scope:
- Add public generator-based `SolveInParallel` C++ overload with index range.
- Consolidate per-index inputs into one callback payload to match upstream review direction.
- Preserve existing semantics from vector-based API: per-thread solver cache, `CommonSolverOption::kMaxThreads` rewrite, serial fallback for non-thread-safe programs.
- Route existing vector overload through the generator implementation to avoid behavior drift.
- Add unit tests for generator correctness and invalid input handling.
- Out of scope:
- Python bindings for generator API.
- Refactoring existing callsites.

## Upstream Feedback That Must Be Incorporated
Source: review comments on `RobotLocomotion/drake#22225`.
- Prefer a single callback with a data payload over multiple optional callbacks.
- Allow callback to return `bool` to indicate "skip this index" without requiring null sentinels.
- Keep convenience wrappers if desired, but make the payload callback the canonical API.

## Current API Design (post-feedback)
In `solvers/solve.h`:
- Canonical payload type:
- `struct SolveInParallelIthProgramData { prog, initial_guess, solver_options, solver_id }`
- Canonical callback:
- `using SolveInParallelIthProgramGenerator = std::function<bool(int, int64_t, SolveInParallelIthProgramData*)>;`
- Canonical overload:
- `SolveInParallel(make_ith_program, range_start, range_end, parallelism, dynamic_schedule)`
- Compatibility overload retained:
- Separate generator callbacks (program / initial guess / options / solver id) adapt into canonical callback.

Semantics now:
- `range_start <= range_end` is required.
- Canonical callback returns `false` to skip solve for index i (`kSolutionResultNotSet`).
- Canonical callback may also set `prog == nullptr` and return `true` to skip.
- Solver id `nullopt` still means "choose best solver".
- Compatibility overload preserves previous null-return conventions for optional metadata generators.

## Success Criteria
- Canonical payload overload compiles and passes `//solvers:solve_in_parallel_test`.
- Existing vector overload remains behavior-compatible via delegation.
- Invalid range and skip behavior (`false` return and null program) are tested.
- Fork PR #93 stays reviewable against `main` with a focused diff.

## Implementation Plan
1. Rebase/retarget fork PR to `main` so the delta is auditable.
2. Introduce canonical payload callback type and overload in `solvers/solve.h`.
3. Rework `solvers/solve.cc` internals to cache payload data per index.
4. Adapt compatibility overload (separate callbacks) into canonical callback and delegate.
5. Extend tests in `solvers/test/solve_in_parallel_test.cc` for:
6. Invalid range.
7. Callback returning `false` skips index.
8. Null `prog` in payload also skips index.
9. Run targeted tests before pushing.
10. Push branch and update PR notes with explicit upstream feedback incorporation.

## Test Plan
- `bazel test //solvers:solve_in_parallel_test --test_output=errors`

## Risks / Open Questions
- Program pointer lifetime remains caller responsibility until `SolveInParallel` returns.
- API growth risk: keeping both canonical + compatibility overloads increases header surface area, but eases migration and preserves ergonomics.

## Execution Log
- 2026-02-19 00:00:00 UTC: Created initial plan before code edits.
- 2026-02-19 02:30:00 UTC: Added generator callback aliases and overload declaration in `solvers/solve.h`.
- 2026-02-19 02:33:00 UTC: Implemented generator-based solve path in `solvers/solve.cc`; delegated vector overload through generator adapters.
- 2026-02-19 02:36:00 UTC: Updated `solvers/test/solve_in_parallel_test.cc` for generator-overload equivalence and invalid-input throws.
- 2026-02-19 02:38:00 UTC: Ran `bazel test //solvers:solve_in_parallel_test --test_output=errors` (pass).
- 2026-02-19 02:46:00 UTC: Investigated large PR #93 diff; root cause was wrong base branch (`master`).
- 2026-02-19 02:48:00 UTC: Retargeted fork PR #93 base to `main`; diff reduced to expected feature-only scope.
- 2026-02-19 02:55:00 UTC: Reviewed prior upstream discussion in `RobotLocomotion/drake#22225` and extracted callback-structure feedback.
- 2026-02-19 03:00:00 UTC: Refactored API and implementation to canonical payload callback with bool skip semantics.
- 2026-02-19 03:03:00 UTC: Re-ran `bazel test //solvers:solve_in_parallel_test --test_output=errors` (pass) on refactor.
- 2026-02-19 03:07:00 UTC: Extended test to cover both skip modes (`return false` and `prog == nullptr`), then reran targeted test (pass).
