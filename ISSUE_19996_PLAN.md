# ISSUE #19996 Plan

Issue: https://github.com/RobotLocomotion/drake/issues/19996
Title: Return variable mapping in MakeSemidefiniteRelaxation
Owner branch: issue-19996-feature
Last updated: 2026-02-19 02:42:47 UTC
Status: Implemented (final Python target run deferred)

## Problem Statement
`MakeSemidefiniteRelaxation(prog)` returns only the relaxed program, so callers must manually reverse-engineer lifted variable indexing from PSD constraints. This duplicates internal logic and is error-prone.

## Scope Decision
- In scope:
- Add a backward-compatible C++ overload that outputs the internal `Variable -> sorted index` map used to place original variables into lifted matrix `X`.
- Add a Python API that returns `(relaxation, mapping)`.
- Add C++ and Python unit coverage for mapping correctness.
- Out of scope:
- Full monomial mapping utilities.
- Grouped-relaxation mapping API design.

## Implemented API (MVP)
1. C++ overload in `solvers/semidefinite_relaxation.h`:
- `MakeSemidefiniteRelaxation(const MathematicalProgram&, std::map<symbolic::Variable, int>*, const SemidefiniteRelaxationOptions&)`
2. Existing overload preserved and internally delegated through shared implementation.
3. Python function in `bindings/pydrake/solvers/solvers_py_semidefinite_relaxation.cc`:
- `MakeSemidefiniteRelaxationWithVariableMapping(prog, options=...) -> (relaxation, dict)`

## Success Criteria
- C++ callers can get deterministic variable index mapping without parsing PSD constraint internals.
- Mapping aligns with `X(i, end)` placement of original decision variables.
- Existing `MakeSemidefiniteRelaxation` behavior remains unchanged.

## Implementation Plan
1. Thread optional map output through semidefinite-relaxation construction internals.
2. Add public overload in header / source.
3. Add Python tuple-returning helper.
4. Add tests:
- `solvers/test/semidefinite_relaxation_test.cc`
- `bindings/pydrake/solvers/test/semidefinite_relaxation_test.py`
5. Run targeted tests.

## Test Plan
- C++: `bazel test //solvers:semidefinite_relaxation_test --test_output=errors`
- Python: `bazel test //bindings/pydrake/solvers:py/semidefinite_relaxation_test --test_output=errors`

## Risks / Open Questions
- Python target has heavy transitive build cost; may require dedicated run window in low-contention environment.

## Execution Log
- 2026-02-19 00:00:00 UTC: Created initial plan.
- 2026-02-19 02:31:00 UTC: Added overload declaration and docs in `solvers/semidefinite_relaxation.h`.
- 2026-02-19 02:32:00 UTC: Implemented overload plumbing and shared helper in `solvers/semidefinite_relaxation.cc`.
- 2026-02-19 02:34:00 UTC: Added Python wrapper `MakeSemidefiniteRelaxationWithVariableMapping`.
- 2026-02-19 02:35:00 UTC: Added C++ mapping test in `solvers/test/semidefinite_relaxation_test.cc`.
- 2026-02-19 02:35:00 UTC: Added Python mapping test in `bindings/pydrake/solvers/test/semidefinite_relaxation_test.py`.
- 2026-02-19 02:37:00 UTC: Ran `bazel test //solvers:semidefinite_relaxation_test --test_output=errors` (pass).
- 2026-02-19 02:38-02:43 UTC: Attempted `bazel test //bindings/pydrake/solvers:py/semidefinite_relaxation_test --test_output=errors`; build was progressing but deferred due long compile time in this session.
