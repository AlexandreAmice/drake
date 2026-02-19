# ISSUE 20451 Plan - Return sample probability for schema distributions

Issue: https://github.com/RobotLocomotion/drake/issues/20451

## Objective

Add a reusable probability-evaluation API for `schema::Distribution` and `DistributionVariant` so callers can evaluate probability density / mass at a sampled value without per-type branching.

## Scope

- In scope:
  - Extend scalar distribution classes with probability evaluation method(s).
  - Add variant-level helper function(s) (similar style to existing `Sample`, `Mean`, `ToSymbolic` helpers).
  - Add focused tests for all scalar distribution kinds.
- Out of scope:
  - Autodiff wrt distribution parameters (current schema types store `double` parameters).
  - Vector distribution probability APIs (can follow-up).
  - Python bindings unless required by build breakage.

## Design notes

- Issue discussion suggests "CalcProbabilityDensity available in the base class" as a desired direction.
- For continuous distributions (`Gaussian`, `Uniform`), evaluate PDF.
- For discrete distributions (`UniformDiscrete`, deterministic), evaluate PMF-like probability at exact support values.
- API naming will make this mixed semantics explicit via docs.

## Implementation plan

1. API surface in `common/schema/stochastic.h`.
   - Add virtual method on `Distribution` for probability-at-value evaluation.
   - Add free-function helper on `DistributionVariant`.
2. Implement in `common/schema/stochastic.cc`.
   - Deterministic: 1 at exact value, otherwise 0.
   - Gaussian: normal density at `x`.
   - Uniform: reciprocal interval length within support, else 0.
   - UniformDiscrete: probability mass across occurrences.
   - Variant helper delegates through `ToDistribution`.
3. Tests in `common/schema/test/stochastic_test.cc`.
   - Add explicit checks for each distribution class.
   - Validate edge behavior outside support.
4. Build/test gate.
   - Run `bazel test //common/schema:test/stochastic_test`.

## Verification checklist

- [x] New API is documented and consistent with existing helper style.
- [x] Probability values are numerically correct in tests.
- [x] Existing stochastic tests remain green (`//common/schema:stochastic_test`).

## Risks and mitigations

- Risk: confusion between density vs mass semantics.
  - Mitigation: explicit docs and class-by-class test assertions.
- Risk: degenerate parameter values (e.g., zero-width uniform).
  - Mitigation: define deterministic fallback behavior and test it.

## Progress log

- 2026-02-19: Created implementation plan with scope limits and numerical behavior targets before editing code.
- 2026-02-19: Added local Bazel worktree prerequisites (`gen/environment.bazelrc` symlink and `gen/python_version.txt`) so tests can execute from this worktree.
- 2026-02-19: Added `Distribution::CalcProbabilityDensity(double)` and per-type implementations for `Deterministic`, `Gaussian`, `Uniform`, and `UniformDiscrete`.
- 2026-02-19: Added variant helper `schema::CalcProbabilityDensity(const DistributionVariant&, double)`.
- 2026-02-19: Extended `common/schema/test/stochastic_test.cc` with scalar and variant-level probability-density assertions.
- 2026-02-19: Ran `bazel test //common/schema:stochastic_test` and confirmed pass.
