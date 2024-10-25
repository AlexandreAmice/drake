#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/random.h"
#include "drake/multibody/tree/frame.h"
#include "drake/multibody/tree/mobilizer.h"
#include "drake/multibody/tree/multibody_element.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/multibody/tree/multibody_tree_topology.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace internal {

/* Base class for specific Mobilizer implementations with the number of
generalized positions and velocities resolved at compile time as template
parameters. This allows specific mobilizer implementations to only work on
fixed-size Eigen expressions therefore allowing for optimized operations on
fixed-size matrices. In addition, this layer discourages the proliferation
of dynamic-sized Eigen matrices that would otherwise lead to run-time
dynamic memory allocations.

Every concrete Mobilizer derived from MobilizerImpl must implement the
following (ideally inline) methods.

  // Returns X_FM(q)
  math::RigidTransform<T> calc_X_FM(const T* q) const;

  // Returns H_FM(q)⋅v
  SpatialVelocity<T> calc_V_FM(const T* q,
                               const T* v) const;

  // Returns H_FM(q)⋅vdot + Hdot_FM(q,v)⋅v
  SpatialAcceleration<T> calc_A_FM(const T* q,
                                   const T* v,
                                   const T* vdot) const;

  // Returns tau = H_FMᵀ(q)⋅F_BMo_F
  void calc_tau(const T* q, const SpatialForce<T>& F_BMo_F, T* tau) const;

The coordinate pointers are guaranteed to point to the kNq or kNv state
variables for the particular mobilizer. They are only 8-byte aligned so
be careful when interpreting them as Eigen vectors for computation purposes.

MobilizerImpl also provides a number of size specific methods to retrieve
multibody quantities of interest from caching structures. These are common
to all mobilizer implementations and therefore they live in this class.
Users should not need to interact with this class directly unless they need
to implement a custom Mobilizer class.

@tparam_default_scalar */
template <typename T, int compile_time_num_positions,
          int compile_time_num_velocities>
class MobilizerImpl : public Mobilizer<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MobilizerImpl);

  // Handy enum to grant specific implementations compile time sizes.
  // static constexpr int i = 42; discouraged.  See answer in:
  // http://stackoverflow.com/questions/37259807/static-constexpr-int-vs-old-fashioned-enum-when-and-why
  enum : int {
    kNq = compile_time_num_positions,
    kNv = compile_time_num_velocities,
    kNx = compile_time_num_positions + compile_time_num_velocities
  };
  template <typename U>
  using QVector = Eigen::Matrix<U, kNq, 1>;
  template <typename U>
  using VVector = Eigen::Matrix<U, kNv, 1>;
  template <typename U>
  using HMatrix = Eigen::Matrix<U, 6, kNv>;

  // As with Mobilizer this the only constructor available for this base class.
  // The minimum amount of information that we need to define a mobilizer is
  // provided here. Subclasses of %MobilizerImpl are therefore forced to
  // provide this information in their respective constructors.
  // TODO(sherm1) This is a bad idea -- only the base class should have to
  //  deal with these parameters that are common to all Mobilizers.
  MobilizerImpl(const SpanningForest::Mobod& mobod,
                const Frame<T>& inboard_frame, const Frame<T>& outboard_frame)
      : Mobilizer<T>(mobod, inboard_frame, outboard_frame) {}

  ~MobilizerImpl() override;

  // Sets the elements of the `state` associated with this Mobilizer to the
  // _zero_ state.  See Mobilizer::SetZeroState().
  void SetZeroState(const systems::Context<T>&,
                    systems::State<T>* state) const final {
    get_mutable_positions(state) = get_zero_position();
    get_mutable_velocities(state).setZero();
  };

  // Sets the elements of the `state` associated with this Mobilizer to the
  // _default_ state.  See Mobilizer::set_default_state().
  void set_default_state(const systems::Context<T>&,
                         systems::State<T>* state) const final {
    get_mutable_positions(&*state) = get_default_position();
    get_mutable_velocities(&*state).setZero();
  };

  // Sets the default position of this Mobilizer to be used in subsequent
  // calls to set_default_state().
  void set_default_position(
      const Eigen::Ref<const Vector<double, compile_time_num_positions>>&
          position) {
    default_position_.emplace(position);
  }

  // Sets the elements of the `state` associated with this Mobilizer to a
  // _random_ state.  If no random distribution has been set, then `state` is
  // set to the _default_ state.
  void set_random_state(const systems::Context<T>& context,
                        systems::State<T>* state,
                        RandomGenerator* generator) const override {
    if (random_state_distribution_) {
      const Vector<double, kNx> sample = Evaluate(
          *random_state_distribution_, symbolic::Environment{}, generator);
      get_mutable_positions(state) = sample.template head<kNq>();
      get_mutable_velocities(state) = sample.template tail<kNv>();
    } else {
      set_default_state(context, state);
    }
  }

  // Defines the distribution used to draw random samples from this
  // mobilizer, using a symbolic::Expression that contains random variables.
  void set_random_position_distribution(
      const Eigen::Ref<const Vector<symbolic::Expression,
                                    compile_time_num_positions>>& position) {
    if (!random_state_distribution_) {
      random_state_distribution_.emplace(
          Vector<symbolic::Expression, kNx>::Zero());
      // Note that there is no `get_zero_velocity()`, since the zero velocity
      // is simply zero for all mobilizers.  Setting the velocity elements of
      // the distribution to zero here therefore maintains the default behavior
      // for velocity.
    }

    random_state_distribution_->template head<kNq>() = position;
  }

  // Defines the distribution used to draw random samples from this
  // mobilizer, using a symbolic::Expression that contains random variables.
  void set_random_velocity_distribution(
      const Eigen::Ref<const Vector<symbolic::Expression,
                                    compile_time_num_velocities>>& velocity) {
    if (!random_state_distribution_) {
      random_state_distribution_.emplace(Vector<symbolic::Expression, kNx>());
      // Maintain the default behavior for position.
      random_state_distribution_->template head<kNq>() = get_zero_position();
    }

    random_state_distribution_->template tail<kNv>() = velocity;
  }

 protected:
  // Returns the zero configuration for the mobilizer.
  virtual Vector<double, kNq> get_zero_position() const {
    return Vector<double, kNq>::Zero();
  }

  // Returns the default configuration for the mobilizer.  The default
  // configuration is the configuration used to populate the context in
  // MultibodyPlant::SetDefaultContext().
  Vector<double, kNq> get_default_position() const {
    if (default_position_) {
      return *default_position_;
    }
    return get_zero_position();
  }

  // Returns the current distribution governing the random samples drawn
  // for this mobilizer if one has been set.
  const std::optional<Vector<symbolic::Expression, kNx>>&
  get_random_state_distribution() const {
    return random_state_distribution_;
  }

  // @name    Helper methods to retrieve entries from the Context.
  //@{
  // Helper to return a const fixed-size Eigen::VectorBlock referencing the
  // segment in the state vector corresponding to `this` mobilizer's state.
  // @pre `context` is a valid multibody system Context.
  Eigen::VectorBlock<const VectorX<T>, kNq> get_positions(
      const systems::Context<T>& context) const {
    return this->get_parent_tree().template get_state_segment<kNq>(
        context, this->position_start_in_q());
  }

  // Helper to return a mutable fixed-size Eigen::VectorBlock referencing the
  // segment in the state vector corresponding to `this` mobilizer's state.
  // Causes invalidation of at least q-dependent cache entries.
  // @pre `context` is a valid multibody system Context.
  Eigen::VectorBlock<VectorX<T>, kNq> GetMutablePositions(
      systems::Context<T>* context) const {
    return this->get_parent_tree().template GetMutableStateSegment<kNq>(
        context, this->position_start_in_q());
  }

  // Helper variant to return a const fixed-size Eigen::VectorBlock referencing
  // the segment in the `state` corresponding to `this` mobilizer's generalized
  // positions. No cache invalidation occurs.
  // @pre `state` is a valid multibody system State.
  Eigen::VectorBlock<VectorX<T>, kNq> get_mutable_positions(
      systems::State<T>* state) const {
    return this->get_parent_tree().template get_mutable_state_segment<kNq>(
        state, this->position_start_in_q());
  }

  // Helper to return a const fixed-size Eigen::VectorBlock referencing the
  // segment in the state vector corresponding to `this` mobilizer's state.
  // @pre `context` is a valid multibody system Context.
  Eigen::VectorBlock<const VectorX<T>, kNv> get_velocities(
      const systems::Context<T>& context) const {
    return this->get_parent_tree().template get_state_segment<kNv>(
        context, num_qs_in_state() + this->velocity_start_in_v());
  }

  // Helper to return a mutable fixed-size Eigen::VectorBlock referencing the
  // segment in the state vector corresponding to `this` mobilizer's state.
  // Causes invalidation of at least v-dependent cache entries.
  // @pre `context` is a valid multibody system Context.
  Eigen::VectorBlock<VectorX<T>, kNv> GetMutableVelocities(
      systems::Context<T>* context) const {
    return this->get_parent_tree().template GetMutableStateSegment<kNv>(
        context, num_qs_in_state() + this->velocity_start_in_v());
  }

  // Helper variant to return a const fixed-size Eigen::VectorBlock referencing
  // the segment in the `state` corresponding to `this` mobilizer's generalized
  // velocities. No cache invalidation occurs.
  // @pre `state` is a valid multibody system State.
  Eigen::VectorBlock<VectorX<T>, kNv> get_mutable_velocities(
      systems::State<T>* state) const {
    return this->get_parent_tree().template get_mutable_state_segment<kNv>(
        state, num_qs_in_state() + this->velocity_start_in_v());
  }
  //@}

 private:
  int num_qs_in_state() const {
    const SpanningForest& forest = this->get_parent_tree().forest();
    return forest.num_positions();
  }

  std::optional<Vector<double, kNq>> default_position_{};

  // Note: this is maintained as a concatenated vector so that the evaluation
  // method can share the sampled values of any random variables that are
  // shared between position and velocity.
  std::optional<Vector<symbolic::Expression, kNx>> random_state_distribution_{};
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
