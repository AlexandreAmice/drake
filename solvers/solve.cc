#include "drake/solvers/solve.h"

#include <memory>
#include <unordered_map>
#include <utility>

#include <common_robotics_utilities/parallelism.hpp>

#include "drake/common/nice_type_name.h"
#include "drake/common/text_logging.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/solver_interface.h"

namespace drake {
namespace solvers {
using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::DynamicParallelForIndexLoop;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForIndexLoop;

MathematicalProgramResult Solve(
    const MathematicalProgram& prog,
    const std::optional<Eigen::VectorXd>& initial_guess,
    const std::optional<SolverOptions>& solver_options) {
  const SolverId solver_id = ChooseBestSolver(prog);
  drake::log()->debug("solvers::Solve will use {}", solver_id);
  std::unique_ptr<SolverInterface> solver = MakeSolver(solver_id);
  MathematicalProgramResult result{};
  solver->Solve(prog, initial_guess, solver_options, &result);
  return result;
}

MathematicalProgramResult Solve(
    const MathematicalProgram& prog,
    const Eigen::Ref<const Eigen::VectorXd>& initial_guess) {
  const Eigen::VectorXd initial_guess_xd = initial_guess;
  return Solve(prog, initial_guess_xd, {});
}

MathematicalProgramResult Solve(const MathematicalProgram& prog) {
  return Solve(prog, {}, {});
}

std::vector<MathematicalProgramResult> SolveInParallel(
    const SolveInParallelProgramGenerator& prog_generator,
    const int64_t range_start, const int64_t range_end,
    const std::optional<SolveInParallelInitialGuessGenerator>&
        initial_guess_generator,
    const std::optional<SolveInParallelSolverOptionsGenerator>&
        solver_options_generator,
    const std::optional<SolveInParallelSolverIdGenerator>& solver_id_generator,
    const Parallelism parallelism, const bool dynamic_schedule) {
  DRAKE_THROW_UNLESS(range_start <= range_end);
  const int64_t num_progs = range_end - range_start;

  // Pre-allocate the results (i.e., don't use push_back) so that we can safely
  // write to the vector on multiple threads concurrently.
  std::vector<MathematicalProgramResult> results(
      static_cast<size_t>(num_progs));

  // Track which results are set during the par-for loop. Some programs are not
  // thread safe so will be skipped inside the loop, and we'll need to circle
  // back and solve them serially later. (N.B. we cannot use vector<bool> here
  // because it's not thread safe.)
  std::vector<uint8_t> result_is_populated(static_cast<size_t>(num_progs), 0);

  // Cache generated programs so that non-thread-safe programs encountered
  // during the parallel pass can be solved serially without re-generating.
  std::vector<const MathematicalProgram*> generated_progs(
      static_cast<size_t>(num_progs), nullptr);

  // As unique types of solvers are encountered by the threads, we will cache
  // the solvers into this data structure so that we don't have to create and
  // destroy them at every call to Solve.
  std::vector<std::unordered_map<SolverId, std::unique_ptr<SolverInterface>>>
      solvers(parallelism.num_threads());

  // The worker lambda behaves slightly differently depending on whether we are
  // in the par-for loop or in the single-threaded cleanup pass later on.
  // This is the worker callback for the i'th program.
  auto solve_ith = [&](const bool in_parallel, const int thread_num,
                       const int64_t i_rel) {
    const int64_t i = i_rel + range_start;
    const size_t k = static_cast<size_t>(i_rel);
    const MathematicalProgram* prog = generated_progs[k];
    if (prog == nullptr && !result_is_populated[k]) {
      prog = prog_generator(thread_num, i);
      generated_progs[k] = prog;
    }
    if (prog == nullptr) {
      result_is_populated[k] = true;
      return;
    }

    // If this program is not thread safe, then skip it and save it for later.
    if (in_parallel && !prog->IsThreadSafe()) {
      return;
    }

    // Access (or choose) the required solver.
    const std::optional<SolverId> maybe_solver_id =
        solver_id_generator.has_value() ? (*solver_id_generator)(thread_num, i)
                                        : std::optional<SolverId>{};
    const SolverId solver_id = maybe_solver_id.has_value()
                                   ? *maybe_solver_id
                                   : ChooseBestSolver(*prog);

    // Find (or create) the specified solver.
    auto solver_iter = solvers[thread_num].find(solver_id);
    if (solver_iter == solvers[thread_num].end()) {
      // If this thread has not solved this type of program yet, save the solver
      // for use later.
      solver_iter = solvers[thread_num].emplace_hint(solver_iter, solver_id,
                                                     MakeSolver(solver_id));
    }
    const SolverInterface& solver = *(solver_iter->second);

    // Adjust the solver options to obey `parallelism`. If this solve is part of
    // the par-for, then the solver is only allowed one thread; otherwise
    // (during the serial cleanup pass) it is allowed `parallelism`. Note that
    // if the user set a solver-specific options to use more threads, that could
    // cause it to exceed its limit.
    std::optional<SolverOptions> new_options;
    const SolverOptions* options_for_i =
        solver_options_generator.has_value()
            ? (*solver_options_generator)(thread_num, i)
            : nullptr;
    if (options_for_i != nullptr) {
      // TODO(jwnimmer-tri) The SolverBase should offer an overload to take a
      // list of options, so that we can avoid this copying.
      new_options.emplace(*options_for_i);
    } else {
      new_options.emplace();
    }
    new_options->SetOption(CommonSolverOption::kMaxThreads,
                           in_parallel ? 1 : parallelism.num_threads());

    // Convert the initial guess from nullable to optional.
    // TODO(jwnimmer-tri) The SolverBase should offer a nullable overload so
    // that we can avoid this copying.
    std::optional<Eigen::VectorXd> initial_guess;
    const Eigen::VectorXd* initial_guess_for_i =
        initial_guess_generator.has_value()
            ? (*initial_guess_generator)(thread_num, i)
            : nullptr;
    if (initial_guess_for_i != nullptr) {
      initial_guess = *initial_guess_for_i;
    }

    // Solve the program.
    solver.Solve(*prog, initial_guess, new_options, &(results[k]));
    result_is_populated[k] = true;
  };
  const auto solve_ith_parallel = [&](const int thread_num, const int64_t i) {
    solve_ith(/* in parallel */ true, thread_num, i);
  };

  const auto solve_ith_serial = [&](const int64_t i) {
    solve_ith(/* in parallel */ false, /* thread_num */ 0, i);
  };

  // Call solve_ith in parallel for all of the progs if more than one thread is
  // available.
  if (parallelism.num_threads() > 1) {
    if (dynamic_schedule) {
      DynamicParallelForIndexLoop(
          DegreeOfParallelism(parallelism.num_threads()), 0, num_progs,
          solve_ith_parallel, ParallelForBackend::BEST_AVAILABLE);
    } else {
      StaticParallelForIndexLoop(DegreeOfParallelism(parallelism.num_threads()),
                                 0, num_progs, solve_ith_parallel,
                                 ParallelForBackend::BEST_AVAILABLE);
    }
  }

  // De-allocate the solvers cache except for the first worker. (We'll use the
  // first worker's cache for the serial solves, below.) The clearing is
  // important in case the solvers hold onto scarce resources (e.g., licenses).
  solvers.resize(1);

  // Finish solving the programs that couldn't be solved in parallel.
  for (int64_t i = 0; i < num_progs; ++i) {
    if (!result_is_populated[static_cast<size_t>(i)]) {
      solve_ith_serial(i);
    }
  }

  return results;
}

std::vector<MathematicalProgramResult> SolveInParallel(
    const std::vector<const MathematicalProgram*>& progs,
    const std::vector<const Eigen::VectorXd*>* initial_guesses,
    const std::vector<const SolverOptions*>* solver_options,
    const std::vector<std::optional<SolverId>>* solver_ids,
    const Parallelism parallelism, const bool dynamic_schedule) {
  DRAKE_THROW_UNLESS(std::all_of(progs.begin(), progs.end(), [](auto* prog) {
    return prog != nullptr;
  }));
  if (initial_guesses != nullptr) {
    DRAKE_THROW_UNLESS(initial_guesses->size() == progs.size());
  }
  if (solver_options != nullptr) {
    DRAKE_THROW_UNLESS(solver_options->size() == progs.size());
  }
  if (solver_ids != nullptr) {
    DRAKE_THROW_UNLESS(solver_ids->size() == progs.size());
  }
  std::optional<SolveInParallelInitialGuessGenerator> initial_guess_generator;
  std::optional<SolveInParallelSolverOptionsGenerator> solver_options_generator;
  std::optional<SolveInParallelSolverIdGenerator> solver_id_generator;
  if (initial_guesses != nullptr) {
    initial_guess_generator = [initial_guesses](int, int64_t i) {
      return (*initial_guesses)[static_cast<size_t>(i)];
    };
  }
  if (solver_options != nullptr) {
    solver_options_generator = [solver_options](int, int64_t i) {
      return (*solver_options)[static_cast<size_t>(i)];
    };
  }
  if (solver_ids != nullptr) {
    solver_id_generator = [solver_ids](int, int64_t i) {
      return (*solver_ids)[static_cast<size_t>(i)];
    };
  }
  const SolveInParallelProgramGenerator program_generator =
      [&progs](int, int64_t i) {
        return progs[static_cast<size_t>(i)];
      };

  return SolveInParallel(program_generator, /*range_start=*/0,
                         /*range_end=*/static_cast<int64_t>(progs.size()),
                         initial_guess_generator, solver_options_generator,
                         solver_id_generator, parallelism, dynamic_schedule);
}

std::vector<MathematicalProgramResult> SolveInParallel(
    const std::vector<const MathematicalProgram*>& progs,
    const std::vector<const Eigen::VectorXd*>* initial_guesses,
    const SolverOptions* solver_options,
    const std::optional<SolverId>& solver_id, const Parallelism parallelism,
    const bool dynamic_schedule) {
  // Broadcast the option and id arguments into vectors (if given).
  std::optional<std::vector<const SolverOptions*>> broadcast_options;
  std::optional<std::vector<std::optional<SolverId>>> broadcast_ids;
  if (solver_options != nullptr) {
    broadcast_options.emplace(progs.size(), solver_options);
  }
  if (solver_id.has_value()) {
    broadcast_ids.emplace(progs.size(), solver_id);
  }
  // Delegate to the primary overload.
  return SolveInParallel(
      progs, initial_guesses,
      broadcast_options.has_value() ? &(*broadcast_options) : nullptr,
      broadcast_ids.has_value() ? &(*broadcast_ids) : nullptr,  // BR
      parallelism, dynamic_schedule);
}

}  // namespace solvers
}  // namespace drake
