# ISSUE #20892 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/20892
Title: Opaque Error Message For Quadratic Cost
Owner branch: issue-20892-fix
Last updated: 2026-02-19 01:35:51 UTC
Status: Completed

## Problem Statement
When conic solver paths rewrite convex quadratic costs and PSD decomposition fails (e.g., numerically indefinite / rank-deficient matrices), users can receive opaque errors such as `Y is not positive semidefinite.` The message lacks context about which solve pathway and cost transformation failed.

## Success Criteria
- Failure messages surfaced through solver code include actionable context (e.g., quadratic cost to conic conversion/decomposition failed).
- Regression test proves message quality improvement on a reproducible failing case.
- No change to successful solve behavior.

## Technical Hypothesis
Call sites that invoke `math::DecomposePSDmatrixIntoXtransposeTimesX` can catch decomposition exceptions and rethrow with richer context. Per maintainer suggestion in issue comments, introducing non-throwing decomposition helpers may be ideal long-term, but an immediate scoped improvement can be achieved at solver call sites.

## Implementation Plan
1. Identify all solver call sites that decompose PSD matrices during quadratic-cost epigraph rewriting.
2. Choose one consistent helper or wrapper for contextual exception translation to avoid duplicated ad-hoc messages.
3. Update affected code paths (likely Mosek, Clarabel/other conic transformations as applicable).
4. Add regression test:
- Construct convex-tagged quadratic cost with near-indefinite numerical matrix that triggers decomposition failure during solve.
- Assert thrown message contains user-facing context and original failure hint.
5. Run solver-specific tests and impacted suite.

## Test Plan
- Primary targeted tests in `solvers/test/...` files touched.
- Focused run for affected solver(s), e.g.:
  - `bazel test //solvers/test:mosek_solver_test` (or nearest existing target)
  - Additional small target covering common conversion utilities if touched.

## Risks / Open Questions
- Some solver paths may already have partial context; we need consistent wording.
- External solver availability can constrain which tests run locally; prioritize tests that do not require proprietary solver licenses.

## Execution Log
- 2026-02-19 01:35:51 UTC: Created worktree and detailed implementation plan before code changes.
- 2026-02-19 01:45:12 UTC: Identified solver-side call sites for PSD decomposition during quadratic-cost conic conversion in `solvers/scs_solver.cc` and `solvers/mosek_solver_internal.cc`.
- 2026-02-19 01:45:12 UTC: Added contextual exception wrapping in both SCS and Mosek conversion paths to include solver context and guidance about potentially incorrect `is_convex=true` usage.
- 2026-02-19 01:45:12 UTC: Added regression test `TestScs.QuadraticCostConversionErrorMessage` to assert user-facing message quality when decomposition fails.
- 2026-02-19 01:48:40 UTC: Initial `//solvers:scs_solver_test` run failed to compile because the new test used `DRAKE_EXPECT_THROWS_MESSAGE` without adding `//common/test_utilities:expect_throws_message` to `scs_solver_test` deps.
- 2026-02-19 01:49:25 UTC: Updated `solvers/BUILD.bazel` to add the missing test dependency.
- 2026-02-19 01:52:49 UTC: Ran `bazel test //solvers:scs_solver_test --test_output=errors`; test passed.
- 2026-02-19 01:52:58 UTC: SCS validation complete; Mosek codepath updated but not runtime-validated locally due external solver availability constraints.
