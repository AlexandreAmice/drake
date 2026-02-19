# ISSUE #7946 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/7946
Title: Polynomial library throws exception when integrating a constant polynomial
Owner branch: issue-7946-fix
Last updated: 2026-02-19 01:35:51 UTC
Status: Completed

## Problem Statement
`Polynomial::Integral(double integration_constant)` can throw `don't know the variable name` when the polynomial is constant. The API should support integrating constant polynomials for single-variable polynomial workflows without forcing users to manually provide an indeterminate name.

## Success Criteria
- Integrating a constant polynomial constructed through common Drake pathways does not throw.
- Resulting polynomial is mathematically correct (slope equals constant term, with integration constant applied).
- Existing behavior for explicitly named / multi-variable contexts remains unchanged.
- Unit tests cover the previously failing scenario and at least one nearby edge case.

## Technical Hypothesis
The current implementation relies on inferring a unique indeterminate; for constants there may be zero indeterminates and code paths that require a variable name throw. We should define a deterministic fallback for the single-variable `Polynomial` API path (likely using `Polynomial::kDefaultIndeterminateName`, currently `"t"`) when there are no indeterminates.

## Implementation Plan
1. Reproduce failure in current test suite context.
2. Locate `Polynomial::Integral` implementation and identify where variable resolution fails on zero indeterminates.
3. Implement minimal behavior change:
- When integrating without explicit variable and indeterminate set is empty, use default variable `t`.
- Keep throwing for ambiguous multi-variable cases when no variable is provided.
4. Add/extend tests in `common/symbolic/test/polynomial_test.cc`:
- Regression test from issue (constant derivative integrated with nonzero constant).
- Check derivative of integrated result returns original constant polynomial.
5. Run targeted tests and ensure no regressions.

## Test Plan
- `bazel test //common/symbolic/test:polynomial_test`
- If needed for confidence: `bazel test //common/symbolic/...`

## Risks / Open Questions
- If maintainers prefer throwing for zero-indeterminate constants, fallback-to-`t` may be considered semantic change.
- Need to ensure fallback applies only where mathematically unambiguous.

## Execution Log
- 2026-02-19 01:35:51 UTC: Created worktree and detailed implementation plan before code changes.
- 2026-02-19 01:39:38 UTC: Located the failing path in `common/polynomial.cc` (legacy polynomial, not symbolic polynomial). Implemented fallback to default variable `t` when integrating a constant polynomial with no existing variable terms.
- 2026-02-19 01:39:38 UTC: Added regression test `PolynomialTest.IntegralOfConstantDerivative` in `common/test/polynomial_test.cc` to reproduce the issue and assert round-trip correctness.
- 2026-02-19 01:41:06 UTC: Created local Bazel prerequisites in the worktree (`gen/environment.bazelrc`, `gen/python_version.txt`) so tests can run from this isolated checkout.
- 2026-02-19 01:41:56 UTC: Ran `bazel test //common:polynomial_test --test_output=errors`; test passed.
- 2026-02-19 01:42:02 UTC: Issue fix validated; ready for PR preparation.
