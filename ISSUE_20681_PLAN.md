# ISSUE 20681 Plan - Add optional solver argument to `solvers::Solve()`

Issue: https://github.com/RobotLocomotion/drake/issues/20681

## Objective

Add a first-class API path for callers to provide an explicit solver instance when invoking top-level `solvers::Solve`, eliminating repeated `if (solver != nullptr) { ... } else { ... }` logic in downstream wrappers.

## Scope

- In scope:
  - Add a new overload (or signature extension) for `solvers::Solve` that accepts `const SolverInterface*`.
  - Preserve existing behavior when no solver is provided (`ChooseBestSolver` path).
  - Add unit tests proving explicit solver usage and no-regression behavior.
  - Keep API and docs coherent in `solve.h`.
- Out of scope:
  - Broad API propagation across all wrappers (IK, IRIS, Toppra, etc.) in this issue branch.
  - Changes to `SolveInParallel` solver-id APIs.

## Constraints and design decisions

- Must support out-of-tree solvers, so API accepts `const SolverInterface*` (not `SolverId`) per issue discussion.
- Must remain source-compatible for existing 1/2/3-arg `Solve` call sites.
- Must keep semantics of option precedence and initial guess unchanged.
- Must avoid any solver ownership transfer; caller owns the pointer lifetime.

## Implementation plan

1. API update in `solvers/solve.h`.
   - Extend `Solve(const MathematicalProgram&, optional initial_guess, optional solver_options, const SolverInterface* solver = nullptr)`.
   - Keep `Solve(prog)` and `Solve(prog, initial_guess)` convenience overloads.
   - Add doc text for explicit solver semantics.
2. Behavior update in `solvers/solve.cc`.
   - If `solver != nullptr`, call that solver directly.
   - Else keep existing `ChooseBestSolver` + `MakeSolver` flow.
   - Preserve debug log message with the actual solver id used.
3. Tests in `solvers/test/solve_test.cc`.
   - Add a test that passes a known available solver (`LinearSystemSolver`) explicitly and validates solution + solver id.
   - Keep no-regression checks for existing call patterns.
4. Build/test gate.
   - Run `bazel test //solvers/test:solve_test`.
   - If needed, run targeted related tests for compilation confidence.

## Verification checklist

- [x] New API compiles without ambiguity at current call sites.
- [x] Explicit solver pointer path is exercised by unit tests.
- [x] Existing tests continue to pass (`//solvers:solve_test`).
- [x] No in-tree C++ compile breakage observed for this target.

## Risks and mitigations

- Risk: overload ambiguity with existing signatures.
  - Mitigation: use a single canonical signature for optional arguments with explicit convenience overloads.
- Risk: explicit incompatible solver pointer usage fails at runtime.
  - Mitigation: rely on existing solver-side checks; optionally add throw expectation test if stable.

## Progress log

- 2026-02-19: Created implementation plan with scope, design constraints, and test gates prior to code edits.
- 2026-02-19: Added local Bazel worktree prerequisites (`gen/environment.bazelrc` symlink and `gen/python_version.txt`) so tests can execute from this worktree.
- 2026-02-19: Updated `solvers::Solve` API to accept `const SolverInterface* solver` and to use `ChooseBestSolver` only when `solver == nullptr`.
- 2026-02-19: Added `SolveTest.ExplicitSolverArgument` coverage in `solvers/test/solve_test.cc`.
- 2026-02-19: Ran `bazel test //solvers:solve_test` and confirmed pass.
