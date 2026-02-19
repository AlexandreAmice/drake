# ISSUE 20812 Plan - Support mixture distributions in schema::Distribution

Issue: https://github.com/RobotLocomotion/drake/issues/20812

## Objective

Add a new scalar mixture/composite distribution type under `common/schema/stochastic.*` so users can define weighted combinations of existing scalar distributions and sample from them.

## Scope

- In scope:
  - Add one new scalar distribution class (mixture/composite semantics).
  - Extend `DistributionVariant` to include the new type.
  - Implement `Sample`, `Mean`, and `ToSymbolic` behavior.
  - Add YAML round-trip and behavior tests.
- Out of scope:
  - Recursive mixture-of-mixture support unless naturally straightforward.
  - Vector-valued mixture distributions in this branch.
  - Python bindings unless required by build breakage.

## Design decisions

- Use relative weights (normalized internally) to reduce user friction.
- Keep options heterogeneous across existing scalar distribution kinds.
- Reject invalid configurations (empty options, negative weights, all-zero weights).
- Keep symbolic behavior consistent with existing stochastic symbolic forms.

## Implementation plan

1. Type additions in `common/schema/stochastic.h`.
   - Introduce new class (name: `Mixture` or `Composite`).
   - Add nested `Option` struct with `relative_probability` and distribution payload.
   - Update `DistributionVariant` alias and docs.
2. Behavior in `common/schema/stochastic.cc`.
   - Sampling via `std::discrete_distribution`.
   - Mean as weighted average of option means.
   - Symbolic expression via random-uniform branch over cumulative weights.
3. Tests in `common/schema/test/stochastic_test.cc`.
   - YAML parse + save.
   - Mean correctness.
   - Sample support checks using deterministic sub-distributions for stable assertions.
4. Build/test gate.
   - Run `bazel test //common/schema:test/stochastic_test`.

## Verification checklist

- [x] New distribution type serializes and deserializes via YAML.
- [x] Deterministic sample-selection tests pass.
- [x] Existing stochastic distribution tests remain green (`//common/schema:stochastic_test`).

## Risks and mitigations

- Risk: recursive variant design complexity.
  - Mitigation: use a non-recursive payload variant for option internals if needed.
- Risk: floating-point edge handling for weights.
  - Mitigation: normalize once and test invalid configurations.

## Progress log

- 2026-02-19: Created implementation plan with constraints, data model, and test strategy before coding.
- 2026-02-19: Added local Bazel worktree prerequisites (`gen/environment.bazelrc` symlink and `gen/python_version.txt`) so tests can execute from this worktree.
- 2026-02-19: Added `schema::Mixture` with weighted options and integrated it into `DistributionVariant`.
- 2026-02-19: Implemented `Mixture::Sample`, `Mixture::Mean`, and `Mixture::ToSymbolic` with normalized relative probabilities and input validation.
- 2026-02-19: Added `StochasticTest.MixtureTest` for YAML round-trip, sampling support checks, mean checks, and invalid-configuration throws.
- 2026-02-19: Ran `bazel test //common/schema:stochastic_test` and confirmed pass.
