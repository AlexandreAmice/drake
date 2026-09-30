#include "drake/solvers/solver_base.h"

#include <limits>
#include <utility>

#include <fmt/format.h>

#include "drake/common/drake_assert.h"
#include "drake/common/nice_type_name.h"
#include "drake/common/scope_exit.h"
#include "drake/solvers/solver_cache_profiler.h"

namespace drake {
namespace solvers {
using internal::SolverCachePhase;
using internal::SolverCachePhaseScope;

SolverBase::SolverBase(
    const SolverId& id, std::function<bool()> available,
    std::function<bool()> enabled,
    std::function<bool(const MathematicalProgram&)> are_satisfied,
    std::function<std::string(const MathematicalProgram&)> explain_unsatisfied)
    : solver_id_(id),
      default_available_(std::move(available)),
      default_enabled_(std::move(enabled)),
      default_are_satisfied_(std::move(are_satisfied)),
      default_explain_unsatisfied_(std::move(explain_unsatisfied)) {}

SolverBase::~SolverBase() = default;

MathematicalProgramResult SolverBase::Solve(
    const MathematicalProgram& prog,
    const std::optional<Eigen::VectorXd>& initial_guess,
    const std::optional<SolverOptions>& solver_options) const {
  MathematicalProgramResult result;
  this->Solve(prog, initial_guess, solver_options, &result);
  return result;
}

namespace {
std::string ShortName(const SolverInterface& solver) {
  return NiceTypeName::RemoveNamespaces(NiceTypeName::Get(solver));
}
}  // namespace

void SolverBase::Solve(const MathematicalProgram& prog,
                       const std::optional<Eigen::VectorXd>& initial_guess,
                       const std::optional<SolverOptions>& solver_options,
                       MathematicalProgramResult* result) const {
  SolverCachePhaseScope phase(SolverCachePhase::kResultPrepare);
  // Validate the pairing before clearing the result, so accidentally passing
  // another program leaves both the solution and its cache intact.
  bool retain_cache = result->get_solver_cache() != nullptr;
  const SolverOptions::OptionValue* retain_option = nullptr;
  for (const SolverOptions* source :
       {&prog.solver_options(), solver_options ? &*solver_options : nullptr}) {
    if (source == nullptr) continue;
    const auto solver = source->options.find(solver_id().name());
    if (solver == source->options.end()) continue;
    const auto option = solver->second.find("retain_solver_cache");
    if (option == solver->second.end()) continue;
    retain_option = &option->second;
  }
  if (retain_option != nullptr) {
    const int* value = std::get_if<int>(retain_option);
    if (value == nullptr || (*value != 0 && *value != 1)) {
      throw std::invalid_argument("retain_solver_cache must be 0 or 1");
    }
    retain_cache = *value != 0;
  }
  if (retain_cache && result->get_solver_cache()) {
    result->get_solver_cache()->CheckCompatibility(prog, solver_id());
  }
  const bool reuse_result = retain_cache && result->has_solver_cache();
  if (reuse_result)
    result->PrepareForSolve();
  else
    *result = {};
  bool prepared = false;
  ScopeExit clear_unprepared([&] {
    if (!prepared && reuse_result) {
      auto cache = result->ReleaseSolverCache();
      *result = {};
      result->SetSolverCache(std::move(cache));
    }
  });
  if (!available()) {
    const std::string name = ShortName(*this);
    throw std::invalid_argument(fmt::format(
        "{} cannot Solve because {}::available() is false, i.e.,"
        " {} has not been compiled as part of this binary."
        " Refer to the {} class overview documentation for how to compile it.",
        name, name, name, name));
  }
  if (!enabled()) {
    const std::string name = ShortName(*this);
    throw std::invalid_argument(fmt::format(
        "{} cannot Solve because {}::enabled() is false, i.e.,"
        " {} has not been properly configured for use."
        " Typically this means that an environment variable has not been set."
        " Refer to the {} class overview documentation for how to enable it.",
        name, name, name, name));
  }
  if (!AreProgramAttributesSatisfied(prog)) {
    throw std::invalid_argument(ExplainUnsatisfiedProgramAttributes(prog));
  }
  result->set_solver_id(solver_id());
  result->SetVariableIndexForSolve(prog.decision_variable_index());
  prepared = true;
  const Eigen::VectorXd& x_init =
      initial_guess ? *initial_guess : prog.initial_guess();
  if (x_init.rows() != prog.num_vars()) {
    throw std::invalid_argument(
        fmt::format("Solve expects initial guess of size {}, got {}.",
                    prog.num_vars(), x_init.rows()));
  }
  if (!solver_options) {
    DoSolve(prog, x_init, prog.solver_options(), result);
  } else {
    SolverOptions merged_options = *solver_options;
    merged_options.Merge(prog.solver_options());
    DoSolve(prog, x_init, merged_options, result);
  }
}

bool SolverBase::available() const {
  DRAKE_DEMAND(default_available_ != nullptr);
  return default_available_();
}

bool SolverBase::enabled() const {
  DRAKE_DEMAND(default_enabled_ != nullptr);
  return default_enabled_();
}

bool SolverBase::AreProgramAttributesSatisfied(
    const MathematicalProgram& prog) const {
  DRAKE_DEMAND(default_are_satisfied_ != nullptr);
  return default_are_satisfied_(prog);
}

std::string SolverBase::ExplainUnsatisfiedProgramAttributes(
    const MathematicalProgram& prog) const {
  if (default_explain_unsatisfied_ != nullptr) {
    return default_explain_unsatisfied_(prog);
  }
  if (AreProgramAttributesSatisfied(prog)) {
    return {};
  }
  return fmt::format("{} is unable to solve a MathematicalProgram with {}.",
                     ShortName(*this), to_string(prog.required_capabilities()));
}

void SolverBase::DoSolve(const MathematicalProgram& prog,
                         const Eigen::VectorXd& initial_guess,
                         const SolverOptions& merged_options,
                         MathematicalProgramResult* result) const {
  internal::SpecificOptions options{&solver_id_, &merged_options};
  if (auto cache = result->ReleaseSolverCache()) {
    // Keep the workspace alive while its virtual operation runs. Rebuilding
    // may install a replacement; native failure invalidates the old cache.
    ScopeExit restore([&] {
      if (cache && cache->valid_ && !result->has_solver_cache())
        result->SetSolverCache(std::move(cache));
    });
    if (cache->DoSolve(prog, initial_guess, &options, result)) return;
    result->SetSolverCache(std::move(cache));
  }
  DoSolve2(prog, initial_guess, &options, result);
}

void SolverBase::DoSolve2(const MathematicalProgram&, const Eigen::VectorXd&,
                          internal::SpecificOptions*,
                          MathematicalProgramResult*) const {
  throw std::logic_error(fmt::format("{} failed to override any DoSolve method",
                                     ShortName(*this)));
}

}  // namespace solvers
}  // namespace drake
