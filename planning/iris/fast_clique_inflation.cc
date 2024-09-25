#include "drake/planning/iris/fast_clique_inflation.h"

#include <algorithm>
// #include <iostream>
#include <string>

#include <common_robotics_utilities/parallelism.hpp>

#include "drake/common/fmt_eigen.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/clarabel_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace planning {

using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::DynamicParallelForIndexLoop;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForIndexLoop;
using geometry::Meshcat;
using geometry::Sphere;
using geometry::optimization::AffineBall;
using geometry::optimization::AffineSubspace;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::VPolytope;
using math::RigidTransform;
using solvers::MathematicalProgram;

namespace {

using values_t = std::vector<double>;
using index_t = std::vector<uint8_t>;

index_t argsort(values_t const& values) {
  index_t index(values.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(), [&values](uint8_t a, uint8_t b) {
    return values[a] < values[b];
  });
  return index;
}

int unadaptive_test_samples(double p, double delta, double tau) {
  return static_cast<int>(-2 * std::log(delta) / (tau * tau * p) + 0.5);
}

}  // namespace

HPolyhedron FastCliqueInflation(const planning::CollisionChecker& checker,
                                const Eigen::MatrixXd& clique,
                                const HPolyhedron& domain,
                                const FastCliqueInflationOptions& options) {
  auto start = std::chrono::high_resolution_clock::now();
  const auto parallelism = Parallelism::Max();
  const int num_threads_to_use =
      checker.SupportsParallelChecking() && options.parallelize
          ? std::min(parallelism.num_threads(),
                     checker.num_allocated_contexts())
          : 1;
  RandomGenerator generator(options.random_seed);
  VPolytope cvxh_vpoly(clique);

  AffineBall ab;
  Eigen::MatrixXd B;
  Eigen::MatrixXd face_projector_matrix =
      Eigen::MatrixXd::Identity(clique.rows(), clique.rows());
  double rank_tol = 1e-6;
  double min_eig = 1e-3;
  if (clique.cols() == 1) {
    ab = AffineBall::MakeHypersphere(1e-2, clique.col(0));
    B = ab.B();
  } else {
    ab = AffineBall::MinimumVolumeCircumscribedEllipsoid(clique, rank_tol);
    B = ab.B();

    auto svd = B.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd sigma = svd.singularValues();

    if (sigma.minCoeff() < rank_tol || clique.cols() < clique.rows() + 1) {
      // make sure B is invertible
      for (int idx = 0; idx < sigma.size(); idx++) {
        if (sigma(idx) <= sqrt(min_eig)) {
          sigma(idx) = sqrt(min_eig);
        }
      }

      // rebuild B
      Eigen::MatrixXd U = svd.matrixU();
      Eigen::MatrixXd V = svd.matrixV();
      Eigen::MatrixXd Sigma_corrected =
          Eigen::MatrixXd::Zero(U.cols(), V.cols());
      Sigma_corrected.diagonal() = sigma;
      B = U * Sigma_corrected * V.transpose();

      AffineSubspace ah(cvxh_vpoly, rank_tol);

      // update face projector matrix
      face_projector_matrix -= ah.OrthogonalComplementBasis() *
                               ah.OrthogonalComplementBasis().transpose();
    }
  }

  const Hyperellipsoid circumscribing_ellipsoid{AffineBall(B, ab.center())};
  Eigen::VectorXd ellipsoid_center = circumscribing_ellipsoid.center();
  Eigen::MatrixXd ellipsoid_A = circumscribing_ellipsoid.A();
  const int dim = circumscribing_ellipsoid.ambient_dimension();

  int current_num_faces = domain.A().rows();

  DRAKE_THROW_UNLESS(num_threads_to_use > 0);
  DRAKE_THROW_UNLESS(domain.ambient_dimension() == dim);
  DRAKE_THROW_UNLESS(domain.IsBounded());
  DRAKE_THROW_UNLESS(domain.PointInSet(ellipsoid_center));

  DRAKE_THROW_UNLESS(domain.ambient_dimension() == clique.rows());

  // cvxh_vpoly = cvxh_vpoly.GetMinimalRepresentation();

  // log()->info("min representation of vpoly done");
  // copy to vector to allow for parallel checking
  std::vector<Eigen::VectorXd> clique_vec;
  clique_vec.reserve(clique.cols());

  for (int col = 0; col < clique.cols(); ++col) {
    Eigen::VectorXd conf = clique.col(col);
    clique_vec.emplace_back(conf);
  }

  std::vector<uint8_t> containment_point_col_free =
      checker.CheckConfigsCollisionFree(clique_vec, parallelism);
  for (const auto col_free : containment_point_col_free) {
    if (!col_free) {
      throw std::runtime_error("One or more clique points are in collision!");
    }
  }

  // handle corner case where center of ellipsoid is in collision
  if (!checker.CheckConfigCollisionFree(ellipsoid_center)) {
    log()->info("original ellipsoid center in collision\n{}",
                fmt_eigen(ellipsoid_center));
    Eigen::Index nearest_point_col;
    (clique - ellipsoid_center).colwise().norm().minCoeff(&nearest_point_col);
    ellipsoid_center = clique.col(nearest_point_col);
    log()->info("new ellipsoid center is \n{}", fmt_eigen(ellipsoid_center));
  }

  // For debugging visualization.
  Eigen::Vector3d point_to_draw = Eigen::Vector3d::Zero();
  if (options.meshcat && dim <= 3) {
    std::string path = "ellipsoid_center";
    options.meshcat->SetObject(path, Sphere(0.06),
                               geometry::Rgba(0.1, 1, 1, 1.0));
    point_to_draw.head(dim) = ellipsoid_center;
    options.meshcat->SetTransform(path, RigidTransform<double>(point_to_draw));
  }

  // upper bound on number of particles required if we hit max iterations
  double delta_min = options.delta * 6 /
                     (M_PI * M_PI * options.max_iterations_separating_planes *
                      options.max_iterations_separating_planes);

  int N_max = unadaptive_test_samples(
      options.admissible_proportion_in_collision, delta_min, options.tau);

  if (options.verbose) {
    log()->info(
        "FastCliqueInflation finding region that is {} collision free with {} "
        "certainty "
        "using {} particles.",
        options.admissible_proportion_in_collision, 1 - options.delta,
        options.num_particles);
    log()->info("FastCliqueInflation worst case test requires {} samples.",
                N_max);
  }
  Eigen::MatrixXd particles = Eigen::MatrixXd::Zero(dim, N_max);

  HPolyhedron P = domain;

  // pre-allocate memory for the polyhedron we are going to construct
  // TODO(wernerpe): find better solution than hardcoding 300
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P.A().rows() + 300, dim);
  Eigen::VectorXd b(P.A().rows() + 300);

  Eigen::MatrixXd ATA = ellipsoid_A.transpose() * ellipsoid_A;
  // // rescaling makes max step computations more stable
  // ATA = (dim / ATA.trace()) * ATA;

  // initialize polytope with domain
  A.topRows(domain.A().rows()) = domain.A();
  b.head(domain.A().rows()) = domain.b();

  // Separating Planes Step
  int num_iterations_separating_planes = 0;

  // track maximum relaxation of cspace margin
  double max_relaxation = 0;

  while (num_iterations_separating_planes <
         options.max_iterations_separating_planes) {
    int k_squared = num_iterations_separating_planes + 1;
    k_squared *= k_squared;
    double delta_k = options.delta * 6 / (M_PI * M_PI * k_squared);
    int N_k = unadaptive_test_samples(
        options.admissible_proportion_in_collision, delta_k, options.tau);

    particles.col(0) = P.UniformSample(&generator, ellipsoid_center);

    // populate particles by uniform sampling in P
    for (int i = 1; i < N_k; ++i) {
      particles.col(i) = P.UniformSample(&generator, particles.col(i - 1));
    }

    // Find all particles in collision
    std::vector<uint8_t> particle_col_free =
        checker.CheckConfigsCollisionFreeEigen(particles.leftCols(N_k),
                                               parallelism);
    int number_particles_in_collision_unadaptive_test =
        N_k -
        std::accumulate(particle_col_free.begin(), particle_col_free.end(), 0);
    int num_particles_to_optimize = std::min(
        number_particles_in_collision_unadaptive_test, options.num_particles);
    Eigen::MatrixXd particles_in_collision(dim, num_particles_to_optimize);
    int number_particles_in_collision = 0;
    for (size_t i = 0; i < particle_col_free.size(); ++i) {
      if (particle_col_free[i] == 0) {
        // only push back a maximum of num_particles for optimization of the
        // faces
        if (options.num_particles > number_particles_in_collision) {
          particles_in_collision.col(number_particles_in_collision) =
              particles.col(i);
          ++number_particles_in_collision;
        }
      }
    }
    if (options.verbose) {
      log()->info(
          "FastCliqueInflation N_k {}, N_col {}, thresh {}", N_k,
          number_particles_in_collision_unadaptive_test,
          (1 - options.tau) * options.admissible_proportion_in_collision * N_k);
    }

    // break if threshold is passed
    if (number_particles_in_collision_unadaptive_test <=
        (1 - options.tau) * options.admissible_proportion_in_collision * N_k) {
      break;
    }

    // warn user if test fails on last iteration
    if (num_iterations_separating_planes ==
        options.max_iterations_separating_planes - 1) {
      log()->warn(
          "FastCliqueInflation WARNING, separating planes hit max iterations "
          "without "
          "passing the unadaptive test, this voids the probabilistic "
          "guarantees!");
    }

    // Update particle positions
    Eigen::MatrixXd particles_in_collision_updated = particles_in_collision;

    // project all collsions onto the convex hull of the clique
    std::optional<std::pair<std::vector<double>, Eigen::MatrixXd>>
        distances_and_projections =
            cvxh_vpoly.Projection(particles_in_collision);
    if (!distances_and_projections) {
      // Projection failed
      throw std::runtime_error(
          "Projection of the particles onto the convex hull of the clique "
          "failed!");
    }
    Eigen::MatrixXd projected_particles = distances_and_projections->second;

    const auto particle_update_work =
        [&checker, &particles_in_collision_updated, &particles_in_collision,
         &projected_particles,
         &options](const int thread_num, const int64_t index) {
          const int point_idx = static_cast<int>(index);
          auto start_point = particles_in_collision.col(point_idx);

          Eigen::VectorXd current_point = start_point;

          // update particles via bisection along the gradient of dist(pt,
          // cvxhclique)
          Eigen::VectorXd grad =
              (current_point - projected_particles.col(point_idx));
          double max_distance = grad.norm();
          grad.normalize();

          Eigen::VectorXd curr_pt_lower = current_point - max_distance * grad;
          // update current point using bisection
          if (!checker.CheckConfigCollisionFree(curr_pt_lower, thread_num)) {
            // directly set to lowerbound
            current_point = curr_pt_lower;
          } else {
            // bisect to find closest point in collision
            Eigen::VectorXd curr_pt_upper = current_point;
            for (int i = 0; i < options.bisection_steps; ++i) {
              Eigen::VectorXd query = 0.5 * (curr_pt_upper + curr_pt_lower);
              if (checker.CheckConfigCollisionFree(query, thread_num)) {
                // config is collision free, increase lower bound
                curr_pt_lower = query;
              } else {
                // config is in collision, decrease upper bound
                curr_pt_upper = query;
                current_point = query;
              }
            }
          }
          //}
          particles_in_collision_updated.col(point_idx) = current_point;
        };
    // update all particles in parallel
    DynamicParallelForIndexLoop(DegreeOfParallelism(num_threads_to_use), 0,
                                number_particles_in_collision,
                                particle_update_work,
                                ParallelForBackend::BEST_AVAILABLE);

    // Resampling particles around found collisions
    // TODO(wernerpe): implement optional resampling step

    // Place Hyperplanes
    std::vector<double> particle_distances;
    particle_distances.reserve(number_particles_in_collision);
    // sort particles by their distance to the convex hull of the clique
    for (int particle_index = 0; particle_index < number_particles_in_collision;
         particle_index++) {
      // the distance to the convex hull is the norm ofupdated particle -
      // original projection
      particle_distances.emplace_back(
          (particles_in_collision_updated.col(particle_index) -
           projected_particles.col(particle_index))
              .norm());
    }

    // returned in ascending order
    auto indices_sorted = argsort(particle_distances);

    // find all particles that have distance ~0 to the convex hull
    // these are likely collisions inside of the convex hull, and need to be
    // treated separately. The procedure is the following:
    // 1. Find 0 distance particles
    // 2. sort them by the ellipsoid metric instead (this gives a sensible
    // ordering to the zero distance particles)
    // 3. put up faces tangent to the ellipsoid, while taking care to project
    // the normal vectors into the affine subspace of the clique if necessary

    // 1. finding 0 distance particles
    int num_critical_particles = 0;
    for (auto i : indices_sorted) {
      if (particle_distances.at(i) >= 1e-9) {
        break;
      }
      ++num_critical_particles;
    }
    // 2. sort by ellipsoid metric
    if (num_critical_particles) {
      std::vector<double> critical_distances(num_critical_particles);
      for (int idx = 0; idx < num_critical_particles; idx++) {
        Eigen::VectorXd critical_particle =
            particles_in_collision_updated.col(indices_sorted[idx]);
        critical_distances.emplace_back(
            (critical_particle - ellipsoid_center).transpose() * ATA *
            (critical_particle - ellipsoid_center));
      }
      auto reordering = argsort(critical_distances);
      // reorder the critical elements
      for (int idx = 0; idx < num_critical_particles; idx++) {
        indices_sorted.at(idx) = indices_sorted.at(reordering.at(idx));
      }
    }

    // bools are not threadsafe - using uint8_t instead to accomondate for
    // parallel checking
    std::vector<uint8_t> particle_is_redundant;

    for (int i = 0; i < number_particles_in_collision; ++i) {
      particle_is_redundant.push_back(0);
    }

    // add separating planes step
    int hyperplanes_added = 0;
    for (auto i : indices_sorted) {
      // add nearest face
      bool allow_relax_cspace_margin = true;
      auto nearest_particle = particles_in_collision_updated.col(i);
      if (!particle_is_redundant[i]) {
        // compute face
        // First, check if is close to projection. If there is a collision
        // inside of the convex hull r
        double dist = (nearest_particle - projected_particles.col(i)).norm();
        Eigen::VectorXd a_face;
        if (dist <= 1e-9) {
          // use ellipsoid, this is likely a collision inside of the convex hull
          log()->info(
              "FastCliqueInflation Warning! Collision inside of "
              "convex hull at \n{}.",
              fmt_eigen(nearest_particle));
          a_face = ATA * (nearest_particle - ellipsoid_center);
          a_face.normalize();
          // project face into linear subspace of the clique if necessary
          a_face = face_projector_matrix * a_face;
          a_face.normalize();

          allow_relax_cspace_margin = false;
        } else {
          // set face tangent to sublevel sets of distance function
          a_face = nearest_particle - projected_particles.col(i);
        }

        a_face.normalize();
        double b_face = a_face.transpose() * nearest_particle -
                        options.configuration_space_margin;

        // relax cspace margin to contain points
        if (allow_relax_cspace_margin) {
          Eigen::VectorXd result = a_face.transpose() * clique;
          double relaxation = result.maxCoeff() - b_face;
          if (relaxation > 0) {
            // dont allow to relax cspace margin beyond 99.9% specified value
            relaxation = std::min(relaxation,
                                  0.999 * options.configuration_space_margin);
            b_face += relaxation;
            if (max_relaxation < relaxation) max_relaxation = relaxation;
          }
        }
        A.row(current_num_faces) = a_face.transpose();
        b(current_num_faces) = b_face;
        ++current_num_faces;
        ++hyperplanes_added;

        // resize A matrix if we need more faces
        if (A.rows() <= current_num_faces) {
          A.conservativeResize(A.rows() * 2, A.cols());
          b.conservativeResize(b.rows() * 2);
        }

        // debugging visualization
        if (options.meshcat && dim <= 3) {
          std::string path =
              fmt::format("face_pt/sepit{:02}/{:03}/pt",
                          num_iterations_separating_planes, current_num_faces);
          options.meshcat->SetObject(path, Sphere(0.03),
                                     geometry::Rgba(1, 1, 0.1, 1.0));
          point_to_draw.head(dim) = nearest_particle;
          options.meshcat->SetTransform(path,
                                        RigidTransform<double>(point_to_draw));
          path =
              fmt::format("start_pt/sepit{:02}/{:03}/pt",
                          num_iterations_separating_planes, current_num_faces);
          options.meshcat->SetObject(path, Sphere(0.03),
                                     geometry::Rgba(0, 0, 0.1, 1.0));
          point_to_draw.head(dim) = particles_in_collision.col(i);
          options.meshcat->SetTransform(path,
                                        RigidTransform<double>(point_to_draw));
          Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3, 2);
          points.topLeftCorner(2, 1) = particles_in_collision.col(i);
          points.topRightCorner(2, 1) = nearest_particle;
          path =
              fmt::format("bisection/sepit{:02}/{:03}/pt",
                          num_iterations_separating_planes, current_num_faces);
          options.meshcat->SetLine(path, points, 2.0,
                                   geometry::Rgba(0, 0.1, 0, 1));
        }

        if (hyperplanes_added == options.max_separating_planes_per_iteration &&
            options.max_separating_planes_per_iteration > 0)
          break;

        // set used particle to redundant
        particle_is_redundant.at(i) = true;

// loop over remaining non-redundant particles and check for
// redundancy
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads_to_use)
#endif
        for (int particle_index = 0;
             particle_index < number_particles_in_collision; ++particle_index) {
          if (!particle_is_redundant[particle_index]) {
            if (a_face.transpose() *
                        particles_in_collision_updated.col(particle_index) -
                    b_face >=
                0) {
              particle_is_redundant[particle_index] = 1;
            }
          }
        }
      }
    }  // end adding hyperplanes

    // update current polyhedron
    P = HPolyhedron(A.topRows(current_num_faces), b.head(current_num_faces));
    if (max_relaxation > 0) {
      log()->info(fmt::format(
          "FastCliqueInflation Warning relaxing cspace margin by {:03} to "
          "ensure point containment",
          max_relaxation));
    }
    // resampling particles in current polyhedron for next iteration
    particles.col(0) = P.UniformSample(&generator);
    for (int j = 1; j < options.num_particles; ++j) {
      particles.col(j) = P.UniformSample(&generator, particles.col(j - 1));
    }
    ++num_iterations_separating_planes;
    // num_iterations_separating_planes -
    //             1 % static_cast<int>(
    //                     0.2 * options.max_iterations_separating_planes) ==
    //         0 &&
    if (options.verbose) {
      log()->info("SeparatingPlanes iteration: {} faces: {}",
                  num_iterations_separating_planes, current_num_faces);
    }
  }  // end separating planes step
  auto stop = std::chrono::high_resolution_clock::now();
  if (options.verbose) {
    log()->info(
        "FastCliqueInflation execution time : {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count());
  }
  return P;
}

}  // namespace planning
}  // namespace drake