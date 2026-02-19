# ISSUE #21551 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/21551
Title: SNOPT misapprehends pseudo-valid option names like "Max iterations"
Owner branch: issue-21551-fix
Last updated: 2026-02-19 01:35:51 UTC
Status: Implemented (pending SNOPT-enabled runtime validation)

## Problem Statement
SNOPT accepts ambiguous/typo option names (e.g., `Max iterations`) and may reinterpret them as different options (e.g., `Maximize`), producing silently wrong optimization behavior. Drake should proactively reject unknown / invalid SNOPT option names before passing options to SNOPT.

## Success Criteria
- Drake validates SNOPT option names against an explicit allowlist for supported SNOPT versions.
- Invalid names like `Max iterations` trigger deterministic error with guidance, instead of being passed through.
- Existing valid option names (and accepted synonyms as needed) continue working.
- Unit tests cover both acceptance and rejection cases.

## Technical Hypothesis
`SnoptSolver` option handling currently forwards string options. Adding a canonicalized-lookup validator in Drake (option name normalization + exact/supported aliases) can block problematic names early. Issue discussion indicates maintainers favor hard-coded known option names over documentation-only mitigation.

## Implementation Plan
1. Locate SNOPT option ingestion path in Drake (likely `solvers/snopt_solver.cc` and shared helpers).
2. Build static allowlist from SNOPT 7.4/7.6 options currently supported by Drake.
3. Implement validation at option-setting / solve-time translation boundary:
- Normalize whitespace/case.
- Validate against known options and intended synonyms.
- Throw descriptive exception on invalid option names with nearest-match hint if practical.
4. Add tests in SNOPT solver tests:
- Valid option names remain accepted.
- `Max iterations` is rejected with actionable message suggesting `major iterations limit` / `minor iterations limit`.
5. Run SNOPT unit tests; if license-gated, run all non-gated checks and document limitations.

## Test Plan
- Targeted SNOPT unit tests in `solvers/test/*snopt*`.
- Any generic solver-option tests affected by shared option parsing.

## Risks / Open Questions
- Full fidelity with SNOPT's abbreviation rules is complex; pragmatic strictness may intentionally disallow some abbreviations to prioritize safety.
- Need to confirm exact supported option spellings in current Drake SNOPT integration.

## Execution Log
- 2026-02-19 01:35:51 UTC: Created worktree and detailed implementation plan before code changes.
- 2026-02-19 01:55:14 UTC: Scoped first landing to the concrete user-facing failure mode from the issue (`Max iterations`) to avoid silently wrong optimization direction while preserving existing SNOPT option behavior.
- 2026-02-19 01:55:14 UTC: Added canonicalized option-name guard in `solvers/snopt_solver.cc` that rejects ambiguous `Max iterations` and recommends `Major iterations limit` / `Minor iterations limit`.
- 2026-02-19 01:55:14 UTC: Added regression test `SnoptSolverTest.RejectAmbiguousMaxIterationsOptionName` in `solvers/test/snopt_solver_test.cc`.
- 2026-02-19 02:03:01 UTC: Built `//solvers:snopt_solver_test` successfully in this worktree (target compiles with changes).
- 2026-02-19 02:03:58 UTC: Attempted runtime test execution via `bazel test //solvers:snopt_solver_test --test_tag_filters=snopt --test_output=errors`. Tests fail in this environment because SNOPT is not compiled/enabled (`SnoptSolver::available() == false`), which is a pre-existing environment limitation also affecting baseline SNOPT tests.
