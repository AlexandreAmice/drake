#include "drake/geometry/optimization/dev/cspace_free_path.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "drake/common/symbolic/monomial_util.h"
#include "drake/geometry/optimization/cspace_free_internal.h"
#include "drake/geometry/optimization/cspace_free_structs.h"
#include "drake/multibody/rational/rational_forward_kinematics_internal.h"

namespace drake {
namespace geometry {
namespace optimization {
std::unordered_map<symbolic::Variable, symbolic::Polynomial>
initialize_path_map(
    CspaceFreePath* cspace_free_path, int maximum_path_degree,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& s_variables) {
  std::unordered_map<symbolic::Variable, symbolic::Polynomial> ret;
  const VectorX<symbolic::Monomial> basis = symbolic::MonomialBasis(
      symbolic::Variables{cspace_free_path->mu_}, maximum_path_degree);

  for (int i = 0; i < s_variables.size(); ++i) {
    const symbolic::Variable s{s_variables(i)};
    // construct a dense polynomial
    symbolic::Polynomial::MapType path_monomial_to_coeff;
    for (int j = 0; j <= maximum_path_degree; ++j) {
      const symbolic::Variable cur_var{fmt::format("s_{}(mu_{})_coeff", i, j)};
      path_monomial_to_coeff.emplace(basis(j), symbolic::Expression{cur_var});
    }
    ret.insert({s, symbolic::Polynomial(std::move(path_monomial_to_coeff))});
  }
  return ret;
}

PlaneSeparatesGeometriesOnPath::PlaneSeparatesGeometriesOnPath(
    const PlaneSeparatesGeometries& plane_geometries,
    const symbolic::Variable& mu,
    const std::unordered_map<symbolic::Variable, symbolic::Polynomial>&
        path_with_y_subs,
    const symbolic::Variables& indeterminates,
    symbolic::Polynomial::SubstituteAndExpandCacheData* cached_substitutions,
    const std::optional<std::map<const symbolic::RationalFunction*,
                                 std::pair<solvers::MatrixXDecisionVariable,
                                           solvers::MatrixXDecisionVariable>>>&
        Q_lam_Q_nu_pairs_pos_side,
    const std::optional<std::map<const symbolic::RationalFunction*,
                                 std::pair<solvers::MatrixXDecisionVariable,
                                           solvers::MatrixXDecisionVariable>>>&
        Q_lam_Q_nu_pairs_neg_side)
    : plane_index{plane_geometries.plane_index} {
  using std::chrono::duration;
  //  auto method_start = std::chrono::high_resolution_clock::now();
  auto substitute_and_create_condition =
      [this, &cached_substitutions, &path_with_y_subs, &indeterminates, &mu,
       &Q_lam_Q_nu_pairs_pos_side, &Q_lam_Q_nu_pairs_neg_side](
          const symbolic::RationalFunction& rational, bool positive_side) {
        symbolic::Variables parameters;
        for (const auto& var : rational.numerator().indeterminates()) {
          parameters.insert(path_with_y_subs.at(var).decision_variables());
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        //        drake::log()->debug("Degree of rational = {}",
        //                            rational.numerator().TotalDegree());
        auto t1 = std::chrono::high_resolution_clock::now();
        t0 = std::chrono::high_resolution_clock::now();
        symbolic::Polynomial path_numerator{
            rational.numerator().SubstituteAndExpand(path_with_y_subs,
                                                     cached_substitutions)};
        t1 = std::chrono::high_resolution_clock::now();
        //        drake::log()->debug("Time to expand poly of degree {} = {}",
        //                            path_numerator.TotalDegree(),
        //                            duration<double>(t1 - t0).count());

        // The current y_slacks along with mu.
        symbolic::Variables cur_indeterminates{
            intersect(indeterminates, rational.numerator().indeterminates())};
        cur_indeterminates.insert(mu);

        t0 = std::chrono::high_resolution_clock::now();
        path_numerator.SetIndeterminates(cur_indeterminates);
        t1 = std::chrono::high_resolution_clock::now();
        //        drake::log()->debug("Time to parse indets poly = {}",
        //                            duration<double>(t1 - t0).count());
        t0 = std::chrono::high_resolution_clock::now();
        if (positive_side) {
          if (Q_lam_Q_nu_pairs_pos_side.has_value()) {
            auto [Q_lam, Q_nu] =
                Q_lam_Q_nu_pairs_pos_side.value().at(&rational);
            positive_side_conditions.emplace_back(path_numerator, mu,
                                                  parameters, Q_lam, Q_nu);
          } else {
            positive_side_conditions.emplace_back(path_numerator, mu,
                                                  parameters);
          }
        } else {
          if (Q_lam_Q_nu_pairs_neg_side.has_value()) {
            auto [Q_lam, Q_nu] =
                Q_lam_Q_nu_pairs_neg_side.value().at(&rational);
            negative_side_conditions.emplace_back(path_numerator, mu,
                                                  parameters, Q_lam, Q_nu);
          }
          else {
            negative_side_conditions.emplace_back(path_numerator, mu, parameters);
          }
        }
        t1 = std::chrono::high_resolution_clock::now();
        //        drake::log()->debug("Time to build conditions = {}\n\n",
        //                            duration<double>(t1 - t0).count());
      };
  if (Q_lam_Q_nu_pairs_pos_side.has_value()) {
    DRAKE_DEMAND(Q_lam_Q_nu_pairs_pos_side.value().size() >=
                 plane_geometries.positive_side_rationals.size());
  }
  if (Q_lam_Q_nu_pairs_neg_side.has_value()) {
    DRAKE_DEMAND(Q_lam_Q_nu_pairs_neg_side.value().size() >=
                 plane_geometries.negative_side_rationals.size());
  }
  for (const auto& rational : plane_geometries.positive_side_rationals) {
    substitute_and_create_condition(rational, true);
  }
  for (const auto& rational : plane_geometries.negative_side_rationals) {
    substitute_and_create_condition(rational, false);
  }
}

CspaceFreePath::CspaceFreePath(const multibody::MultibodyPlant<double>* plant,
                               const geometry::SceneGraph<double>* scene_graph,
                               const Eigen::Ref<const Eigen::VectorXd>& q_star,
                               int maximum_path_degree, int plane_order)
    : rational_forward_kin_(plant),
      scene_graph_{*scene_graph},
      q_star_{q_star},
      link_geometries_{internal::GetCollisionGeometries(*plant, *scene_graph)},
      plane_order_{plane_order},
      mu_(symbolic::Variable("mu")),
      max_degree_(maximum_path_degree),
      path_(initialize_path_map(this, maximum_path_degree,
                                rational_forward_kin_.s())) {
  // collision_pairs maps each pair of body to the pair of collision geometries
  // on that pair of body.
  std::map<SortedPair<multibody::BodyIndex>,
           std::vector<std::pair<const CIrisCollisionGeometry*,
                                 const CIrisCollisionGeometry*>>>
      collision_pairs;
  int num_collision_pairs = internal::GenerateCollisionPairs(
      rational_forward_kin_.plant(), scene_graph_, link_geometries_,
      &collision_pairs);

  const int num_coeffs_per_poly = plane_order + 1;
  separating_planes_.reserve(num_collision_pairs);
  for (const auto& [link_pair, geometry_pairs] : collision_pairs) {
    for (const auto& geometry_pair : geometry_pairs) {
      // Generate the separating plane for this collision pair.
      Vector3<symbolic::Polynomial> a;
      symbolic::Polynomial b;
      VectorX<symbolic::Variable> plane_decision_vars{4 * num_coeffs_per_poly};
      for (int i = 0; i < plane_decision_vars.rows(); ++i) {
        plane_decision_vars(i) =
            symbolic::Variable(fmt::format("plane_var{}", i));
      }
      CalcPathPlane<symbolic::Variable, symbolic::Variable,
                    symbolic::Polynomial>(plane_decision_vars, mu_, plane_order,
                                          &a, &b);

      // Compute the expressed body for this plane
      const multibody::BodyIndex expressed_body =
          multibody::internal::FindBodyInTheMiddleOfChain(
              rational_forward_kin_.plant(), link_pair.first(),
              link_pair.second());
      separating_planes_.emplace_back(a, b, geometry_pair.first,
                                      geometry_pair.second, expressed_body,
                                      plane_order_, plane_decision_vars);

      map_geometries_to_separating_planes_.emplace(
          SortedPair<geometry::GeometryId>(geometry_pair.first->id(),
                                           geometry_pair.second->id()),
          static_cast<int>(separating_planes_.size()) - 1);
    }
  }

  for (int i = 0; i < 3; ++i) {
    y_slack_(i) = symbolic::Variable("y" + std::to_string(i));
  }

  std::vector<std::unique_ptr<CSpaceSeparatingPlane<symbolic::Variable>>>
      separating_planes_ptrs;
  separating_planes_ptrs.reserve(separating_planes_.size());
  for (const auto& plane : separating_planes_) {
    separating_planes_ptrs.push_back(
        std::make_unique<CSpacePathSeparatingPlane<symbolic::Variable>>(plane));
  }
  // Generate the rationals for the separating planes. At this point, the plane
  // components are a function of mu, but the plane_geometries will still be in
  // terms of the s variable.
  std::vector<PlaneSeparatesGeometries> plane_geometries;
  internal::GenerateRationals(separating_planes_ptrs, y_slack_, q_star_,
                              rational_forward_kin_, &plane_geometries);

  const std::map<const CIrisCollisionGeometry*,
                 std::map<const symbolic::RationalFunction*,
                          std::pair<solvers::MatrixXDecisionVariable,
                                    solvers::MatrixXDecisionVariable>>>
      psd_multiplier_map = PreAllocateMultiplierPSD(
          plane_geometries, plane_order, maximum_path_degree);
  GeneratePathRationals(plane_geometries, psd_multiplier_map);

//        GeneratePathRationals(plane_geometries);
}

namespace {
// COPIED FROM cspace_free_polytope.cc!!
template <typename T>
void SymmetricMatrixFromLowerTriangularPart(
    int rows, const Eigen::Ref<const VectorX<T>>& lower_triangle,
    MatrixX<T>* mat) {
  mat->resize(rows, rows);
  DRAKE_DEMAND(lower_triangle.rows() == rows * (rows + 1) / 2);
  int count = 0;
  for (int j = 0; j < rows; ++j) {
    (*mat)(j, j) = lower_triangle(count++);
    for (int i = j + 1; i < rows; ++i) {
      (*mat)(i, j) = lower_triangle(count);
      (*mat)(j, i) = lower_triangle(count);
      count++;
    }
  }
}
}  // namespace

std::map<const CIrisCollisionGeometry*,
         std::map<const symbolic::RationalFunction*,
                  std::pair<solvers::MatrixXDecisionVariable,
                            solvers::MatrixXDecisionVariable>>>
CspaceFreePath::PreAllocateMultiplierPSD(
    const std::vector<PlaneSeparatesGeometries>& plane_geometries,
    const int plane_order, const int maximum_path_degree) const {
  // compute the largest PSD needed
  std::map<const CIrisCollisionGeometry*,
           std::map<const symbolic::RationalFunction*,
                    std::pair<MatrixX<symbolic::Variable>,
                              MatrixX<symbolic::Variable>>>>
      ret;
  for (const PlaneSeparatesGeometries& plane_seps_geom : plane_geometries) {
    const CSpacePathSeparatingPlane<symbolic::Variable> plane{
        separating_planes_.at(plane_seps_geom.plane_index)};
    for (const auto& [geom, rationals] :
         {std::pair(&plane.positive_side_geometry,
                    &plane_seps_geom.positive_side_rationals),
          std::pair(&plane.negative_side_geometry,
                    &plane_seps_geom.negative_side_rationals)}) {
      ret.try_emplace(*geom);
      for (const auto& rational : *rationals) {
        const auto [iterator, success] = ret.at(*geom).try_emplace(&rational);
        // this means that the rational wasn't already in the map
        if (success) {
          int rat_degree_mu = 0;
          for(const auto& [monom, coeff]: rational.numerator().monomial_to_coefficient_map()) {
            int deg_s = 0;
            for(const auto& [var, power] : monom.get_powers()) {
              for(const auto& s: rational_forward_kin_.s()) {
                if(var.equal_to(s)) {
                  deg_s += power;
                }
              }
            }
            rat_degree_mu = std::max(rat_degree_mu, deg_s);
          }
          const int poly_degree_mu =
              rat_degree_mu * maximum_path_degree +
              plane_order;
          const int num_y = internal::GetNumYInRational(rational, y_slack_);

          const int d = static_cast<int>(std::floor(poly_degree_mu / 2));
          const int lam_basis_size = d + 1 + num_y;

          MatrixX<symbolic::Variable> Q_lam{lam_basis_size, lam_basis_size};
          for (int i = 0; i < lam_basis_size; ++i) {
            Q_lam(i, i) = symbolic::Variable(fmt::format("Sl({},{})", i, i));
            for (int j = i; j < lam_basis_size; ++j) {
              Q_lam(i, j) = symbolic::Variable(fmt::format("Sl({},{})", i, j));
              Q_lam(j, i) = Q_lam(i, j);
            }
          }

          const int nu_basis_size =
              poly_degree_mu % 2 == 0 ? lam_basis_size - 1 : lam_basis_size;
          MatrixX<symbolic::Variable> Q_nu(nu_basis_size, nu_basis_size);
          for (int i = 0; i < nu_basis_size; ++i) {
            Q_nu(i, i) = symbolic::Variable(fmt::format("Snu({},{})", i, i));
            for (int j = i; j < nu_basis_size; ++j) {
              Q_nu(i, j) = symbolic::Variable(fmt::format("Snu({},{})", i, j));
              Q_nu(j, i) = Q_nu(i, j);
            }
          }
          ret.at(*geom)[&rational] = std::make_pair(Q_lam, Q_nu);
        }
      }
    }
  }
  return ret;
}

void CspaceFreePath::GeneratePathRationals(
    const std::vector<PlaneSeparatesGeometries>& plane_geometries,
    const std::optional<
        const std::map<const CIrisCollisionGeometry*,
                       std::map<const symbolic::RationalFunction*,
                                std::pair<solvers::MatrixXDecisionVariable,
                                          solvers::MatrixXDecisionVariable>>>>&
        psd_multiplier_map) {
  // plane_geometries_ currently has rationals in terms of the configuration
  // space variable. We create PlaneSeparatesGeometriesOnPath objects which are
  // in terms of the path variable and can be used to construct the
  // certification program once a path is chosen.

  // Add the auxilliary variables for matrix SOS constraints to the substitution
  // map.
  std::unordered_map<symbolic::Variable, symbolic::Polynomial>
      path_with_y_subs = path_;
  path_with_y_subs.emplace(mu_, symbolic::Polynomial(mu_));
  symbolic::Variables indeterminates{mu_};
  for (int i = 0; i < y_slack_.size(); ++i) {
    path_with_y_subs.emplace(y_slack_(i), symbolic::Polynomial(y_slack_(i)));
    indeterminates.insert(y_slack_(i));
  }

//  symbolic::Polynomial::SubstituteAndExpandCacheData cached_substitutions;
//  for (const auto& plane_geometry : plane_geometries) {
//    plane_geometries_on_path_.emplace_back(plane_geometry, mu_,
//                                           path_with_y_subs, indeterminates,
//                                           &cached_substitutions);
//  }
//  unused(psd_multiplier_map);



  const int num_threads =
        std::min(static_cast<int>(std::thread::hardware_concurrency()),
                 static_cast<int>(plane_geometries.size()));
  std::vector<symbolic::Polynomial::SubstituteAndExpandCacheData>
      cached_substitutions(num_threads);
  plane_geometries_on_path_.resize(
      plane_geometries.size(),
      PlaneSeparatesGeometriesOnPath(plane_geometries.at(0), mu_,
                                     path_with_y_subs, indeterminates,
                                     &cached_substitutions.at(0)));
  std::queue<int> vec_args;
  for (int i = 0; i < static_cast<int>(plane_geometries.size()); ++i) {
    vec_args.push(i);
  }

  std::mutex vec_arg_mtx;
  auto plane_geom_constructor_worker =
      [this, &plane_geometries, &cached_substitutions, &vec_args, &vec_arg_mtx,
       &path_with_y_subs, &indeterminates,
       &psd_multiplier_map](const int cached_idx) {
      while (!vec_args.empty()) {
        vec_arg_mtx.lock();
        const int i = vec_args.front();
        vec_args.pop();
        vec_arg_mtx.unlock();
        if (psd_multiplier_map.has_value()) {
          const CSpacePathSeparatingPlane<symbolic::Variable> plane{
              separating_planes_.at(plane_geometries.at(i).plane_index)};
          plane_geometries_on_path_.at(i) = PlaneSeparatesGeometriesOnPath(
              plane_geometries.at(i), mu_, path_with_y_subs, indeterminates,
              &cached_substitutions.at(cached_idx),
              psd_multiplier_map.value().at(plane.positive_side_geometry),
              psd_multiplier_map.value().at(plane.negative_side_geometry)
          );
        } else {
          plane_geometries_on_path_.at(i) = PlaneSeparatesGeometriesOnPath(
              plane_geometries.at(i), mu_, path_with_y_subs, indeterminates,
              &cached_substitutions.at(cached_idx));
        }
      }
      return;
      };
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // Launch all the threads
    thread_pool.emplace_back(plane_geom_constructor_worker, i);
  }
  for (int i = 0; i < num_threads; ++i) {
    // Wait for all the threads to join.
    thread_pool.at(i).join();
  }
}

[[nodiscard]] std::vector<CspaceFreePath::FindSeparationCertificateStatistics>
CspaceFreePath::FindSeparationCertificateGivenPath(
    const MatrixX<Polynomiald>& piecewise_path,
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const CspaceFreePath::FindSeparationCertificateGivenPathOptions& options,
    std::vector<std::unordered_map<SortedPair<geometry::GeometryId>,
                                   std::optional<SeparationCertificateResult>>>*
        certificates) const {
  //  std::cout << "METHOD ENTERED" << std::endl;
  const int num_pieces{static_cast<int>(piecewise_path.cols())};
  //  std::cout << "cols accessed" << std::endl;
  // import chrono for timing
  typedef std::chrono::high_resolution_clock clock;
  using std::chrono::duration;

  certificates->clear();
  certificates->reserve(num_pieces);
  // preallocate the certificate vector
  for (int i = 0; i < num_pieces; ++i) {
    certificates->emplace_back(
        std::unordered_map<SortedPair<geometry::GeometryId>,
                           std::optional<SeparationCertificateResult>>());
    certificates->back().reserve(map_geometries_to_separating_planes_.size());
    for (const auto& [pair, _] : map_geometries_to_separating_planes_) {
      if (ignored_collision_pairs.count(pair) == 0) {
        certificates->at(i).insert({pair, std::nullopt});
      }
    }
  }

  // Stores the indices in path_separating_planes_ that don't appear in
  // ignored_collision_pairs.
  std::vector<int> active_plane_indices;
  active_plane_indices.reserve(separating_planes().size());

  for (int i = 0; i < static_cast<int>(separating_planes().size()); ++i) {
    const SortedPair<geometry::GeometryId> pair(
        separating_planes()[i].positive_side_geometry->id(),
        separating_planes()[i].negative_side_geometry->id());
    if (ignored_collision_pairs.count(pair) == 0) {
      active_plane_indices.push_back(i);
    }
  }

  const int num_threads =
      options.num_threads > 0
          ? options.num_threads
          : std::min(static_cast<int>(std::thread::hardware_concurrency()),
                     static_cast<int>(active_plane_indices.size()));

  // The iᵗʰ entry of this vector is true if the iᵗʰ piece is safe, nullopt if
  // not done certifying, and false otherwise.
  std::vector<std::optional<bool>> piece_is_safe(
      static_cast<unsigned int>(num_pieces), std::nullopt);
  std::vector<std::mutex> piece_is_safe_mutex{
      static_cast<unsigned int>(num_pieces)};
  std::vector<CspaceFreePath::FindSeparationCertificateStatistics>
      certification_statistics(num_pieces, active_plane_indices);
  // Find the max degree of the separation condition polynomial.
  for (auto stats : certification_statistics) {
    for (const auto [plane_index, _] : stats.certifying_poly_degree) {
      int max_deg = 0;
      PlaneSeparatesGeometriesOnPath plane_geoms =
          plane_geometries_on_path_.at(plane_index);
      for (const auto& condition : plane_geoms.positive_side_conditions) {
        max_deg = std::max(max_deg, condition.get_p().TotalDegree());
      }
      for (const auto& condition : plane_geoms.negative_side_conditions) {
        max_deg = std::max(max_deg, condition.get_p().TotalDegree());
      }
      stats.certifying_poly_degree.at(plane_index) = max_deg;
    }
  }

  // Certify that the plane pair at the plane_count index is safe for the
  // segment given by segment_idx.
  auto certify_plane_pair_over_segment = [this, &active_plane_indices,
                                          &piecewise_path, &options,
                                          &certificates, &piece_is_safe,
                                          &piece_is_safe_mutex,
                                          &certification_statistics](
                                             int plane_count, int segment_idx) {
    // Only perform the certification if the current piece might still be safe,
    // and we are not terminating early.
    const bool cur_piece_maybe_safe =
        piece_is_safe.at(segment_idx).value_or(true);
    if (cur_piece_maybe_safe ||
        !(options.terminate_segment_certification_at_failure)) {
      const auto cert_time_start = clock::now();

      const int plane_index = active_plane_indices[plane_count];
      const Eigen::VectorX<Polynomiald> path = piecewise_path.col(segment_idx);
      const SortedPair<geometry::GeometryId> pair(
          separating_planes()[plane_index].positive_side_geometry->id(),
          separating_planes()[plane_index].negative_side_geometry->id());

      auto prog_build_time_start = std::chrono::high_resolution_clock::now();
      //      std::cout << "building program" << std::endl;
      auto certificate_program = MakeIsGeometrySeparableOnPathProgram(
          pair, piecewise_path.col(segment_idx));
      //      std::cout << "prog built" << std::endl;
      auto prog_build_time_end = std::chrono::high_resolution_clock::now();
      certification_statistics.at(segment_idx)
          .time_to_build_prog.at(plane_index) =
          duration<double>(prog_build_time_end - prog_build_time_start).count();

      //      std::cout << "solving program" << std::endl;
      auto result =
          SolveSeparationCertificateProgram(certificate_program, options);
      //      std::cout << "prog solved" << std::endl;
      certification_statistics.at(segment_idx)
          .time_to_solve_prog.at(plane_index) =
          1000 * result.result.get_solver_details<solvers::MosekSolver>()
                     .optimizer_time;  // convert time to ms.
      certification_statistics.at(segment_idx).pair_is_safe.at(plane_index) =
          result.result.is_success();

      certificates->at(segment_idx).at(pair) = result;
      piece_is_safe_mutex.at(segment_idx).lock();
      piece_is_safe.at(segment_idx) =
          piece_is_safe.at(segment_idx).value_or(true) &&
          result.result.is_success();
      piece_is_safe_mutex.at(segment_idx).unlock();

      if (options.verbose) {
        drake::log()->info(
            "SOS {}/{} completed, for Segment {}/{} is_success {}", plane_count,
            active_plane_indices.size(), segment_idx, piecewise_path.cols(),
            certificates->at(segment_idx).at(pair).has_value());
      }
      const auto cert_time_end = clock::now();
      certification_statistics.at(segment_idx)
          .total_time_to_certify_pair.at(plane_index) =
          duration<double>(cert_time_end - cert_time_start).count();
    }
  };

  // The arguments needed to certify a piece for a pair of collision bodies.
  std::queue<std::pair<int, int>> certify_args_queue;
  std::mutex certify_args_mutex;
  for (int segment_idx = 0; segment_idx < num_pieces; segment_idx++) {
    for (int plane_count = 0;
         plane_count < static_cast<int>(active_plane_indices.size());
         ++plane_count) {
      certify_args_queue.push(std::make_pair(plane_count, segment_idx));
    }
  }

  // A function which pulls off arguments pairs and certifies that a pair of
  // collision bodies is safe over a segment.
  auto certify_worker = [&certify_plane_pair_over_segment, &piece_is_safe,
                         &options, &certify_args_queue, &certify_args_mutex,
                         &num_pieces, &active_plane_indices]() {
    // Terminate early if the options say to do so and we find an unsafe
    // segment.
    bool terminate = false;
    while (!terminate) {
      // acquire and remove the front element.
      certify_args_mutex.lock();
      std::pair<int, int> args{certify_args_queue.front()};
      certify_args_queue.pop();
      certify_args_mutex.unlock();
      certify_plane_pair_over_segment(args.first, args.second);
      if (options.verbose) {
        drake::log()->info("SOS {}/{} dispatched, for Segment {}/{}",
                           args.first, active_plane_indices.size(), args.second,
                           num_pieces);
      }

      bool unsafe_segment =
          std::all_of(piece_is_safe.begin(), piece_is_safe.end(),
                      [](std::optional<bool> flag) {
                        return flag.value_or(true);
                      });
      bool terminate_early_due_to_unsafe_segment =
          options.terminate_path_certification_at_failure && unsafe_segment;

      terminate =
          certify_args_queue.empty() || terminate_early_due_to_unsafe_segment;
    }
    return;
  };

  if (options.solver_id == solvers::MosekSolver::id()) {
    // Acquire the license for the duration of the solve.
    const auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  }

  // Solve all the programs
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // Launch all the threads
    thread_pool.emplace_back(certify_worker);
  }
  for (int i = 0; i < num_threads; ++i) {
    // Wait for all the threads to join.
    thread_pool.at(i).join();
  }

  return certification_statistics;
}

[[nodiscard]] CspaceFreePath::SeparationCertificateProgram
CspaceFreePath::MakeIsGeometrySeparableOnPathProgram(
    const SortedPair<geometry::GeometryId>& geometry_pair,
    const VectorX<Polynomiald>& path) const {
  // Fail fast as building the program can be expensive.
  int plane_index{GetSeparatingPlaneIndex(geometry_pair)};
  if (plane_index < 0) {
    throw std::runtime_error(fmt::format(
        "GetIsGeometrySeparableProgram(): geometry pair ({}, {}) does not need "
        "a separation certificate",
        scene_graph_.model_inspector().GetName(geometry_pair.first()),
        scene_graph_.model_inspector().GetName(geometry_pair.second())));
  }

  DRAKE_DEMAND(rational_forward_kin_.s().rows() == path.rows());
  // Now we convert the vector of common::Polynomial to a map from the
  // configuration space variable s to symbolic::Polynomial in mu.
  std::unordered_map<symbolic::Variable, symbolic::Polynomial>
      cspace_var_to_sym_path;
  //  std::cout << "starting evaluation construction" << std::endl;
  for (int i = 0; i < path.rows(); ++i) {
    DRAKE_DEMAND(path(i).is_univariate());
    DRAKE_DEMAND(path(i).GetDegree() <= static_cast<int>(max_degree_));
    symbolic::Polynomial::MapType sym_path_map;
    for (const auto& monom : path(i).GetMonomials()) {
      sym_path_map.insert(
          {symbolic::Monomial(mu_, monom.GetDegree()), monom.coefficient});
    }
    cspace_var_to_sym_path.emplace(rational_forward_kin_.s()(i),
                                   symbolic::Polynomial{sym_path_map});
  }
  //  std::cout << "ending evaluation construction" << std::endl;
  //  std::cout << "constructing plane search prog" << std::endl;
  //  std::cout << fmt::format(
  //                  "plane index is {}, while plane_geometries path is size
  //                  {}",
  //                   plane_index, plane_geometries_on_path_.size())
  //            << std::endl;
  return ConstructPlaneSearchProgramOnPath(
      plane_geometries_on_path_.at(plane_index), cspace_var_to_sym_path);
}

[[nodiscard]] CspaceFreePath::SeparationCertificateProgram
CspaceFreePath::ConstructPlaneSearchProgramOnPath(
    const PlaneSeparatesGeometriesOnPath& plane_geometries_on_path,
    const std::unordered_map<symbolic::Variable, symbolic::Polynomial>& path)
    const {
  SeparationCertificateProgram ret{path, plane_geometries_on_path.plane_index};
  ret.prog->AddIndeterminate(mu_);
  ret.prog->AddIndeterminates(this->y_slack());

  // construct the parameter to value map
  //  std::cout << "evaluating parameters" << std::endl;
  symbolic::Environment param_eval_map;
  for (const auto& [config_space_var, eval_path] : path) {
    const symbolic::Polynomial symbolic_path{path_.at(config_space_var)};
    for (const auto& [mu_monom, mu_var_coeff] :
         symbolic_path.monomial_to_coefficient_map()) {
      // Find the monomial with the matching degree. If it doesn't exist
      // evaluate it to 0.
      const auto evaled_monom_iter =
          eval_path.monomial_to_coefficient_map().find(mu_monom);
      const double mu_var_coeff_eval{
          evaled_monom_iter == eval_path.monomial_to_coefficient_map().end()
              ? 0
              : evaled_monom_iter->second.Evaluate()};
      param_eval_map.insert(*mu_var_coeff.GetVariables().begin(),
                            mu_var_coeff_eval);
    }
  }
  //  std::cout << "finished evaluating parameters" << std::endl;

  // Now add the separation conditions to the program
  for (const auto& condition :
       plane_geometries_on_path.positive_side_conditions) {
    condition.AddPositivityConstraintToProgram(param_eval_map, ret.prog.get());
  }
  for (const auto& condition :
       plane_geometries_on_path.negative_side_conditions) {
    condition.AddPositivityConstraintToProgram(param_eval_map, ret.prog.get());
  }
  return ret;
}

CspaceFreePath::SeparationCertificateResult
CspaceFreePath::SolveSeparationCertificateProgram(
    const CspaceFreePath::SeparationCertificateProgram& certificate_program,
    const FindSeparationCertificateGivenPathOptions& options) const {
  CspaceFreePath::SeparationCertificateResult result;
  internal::SolveSeparationCertificateProgramBase(
      certificate_program, options,
      separating_planes_[certificate_program.plane_index], &result);
  return result;
}

int CspaceFreePath::GetSeparatingPlaneIndex(
    const SortedPair<geometry::GeometryId>& pair) const {
  return (map_geometries_to_separating_planes_.count(pair) == 0)
             ? -1
             : map_geometries_to_separating_planes_.at(pair);
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
