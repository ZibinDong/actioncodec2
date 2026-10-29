// Intrinsic first- and stateful second-order SO(3) error-feedback encoders.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "parallel.hpp"

namespace py = pybind11;

namespace {

struct Quaternion {
  double x;
  double y;
  double z;
  double w;
};

Quaternion normalize(Quaternion q) {
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return q;
}

Quaternion multiply(const Quaternion& a, const Quaternion& b) {
  return normalize({a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z});
}

Quaternion conjugate(const Quaternion& q) { return {-q.x, -q.y, -q.z, q.w}; }

Quaternion from_euler(const float* euler) {
  const double roll = 0.5 * static_cast<double>(euler[0]);
  const double pitch = 0.5 * static_cast<double>(euler[1]);
  const double yaw = 0.5 * static_cast<double>(euler[2]);
  const double sr = std::sin(roll), cr = std::cos(roll);
  const double sp = std::sin(pitch), cp = std::cos(pitch);
  const double sy = std::sin(yaw), cy = std::cos(yaw);
  return normalize({sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy});
}

void to_rotvec(Quaternion q, double* output) {
  if (q.w < 0.0) {
    q.x = -q.x;
    q.y = -q.y;
    q.z = -q.z;
    q.w = -q.w;
  }
  const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  const double scale = norm < 1e-12 ? 2.0 : 2.0 * std::atan2(norm, std::max(0.0, q.w)) / norm;
  output[0] = q.x * scale;
  output[1] = q.y * scale;
  output[2] = q.z * scale;
}

Quaternion from_rotvec(const double* value) {
  const double angle = std::sqrt(value[0] * value[0] + value[1] * value[1] +
                                 value[2] * value[2]);
  const double scale = angle < 1e-12 ? 0.5 : std::sin(0.5 * angle) / angle;
  return normalize({value[0] * scale, value[1] * scale, value[2] * scale,
                    std::cos(0.5 * angle)});
}

Quaternion from_input(const float* value, bool rotation_vector) {
  if (!rotation_vector) return from_euler(value);
  const double converted[3] = {static_cast<double>(value[0]), static_cast<double>(value[1]),
                               static_cast<double>(value[2])};
  return from_rotvec(converted);
}

Quaternion residual(const Quaternion& target, const Quaternion& reconstruction, bool left) {
  return left ? multiply(target, conjugate(reconstruction))
              : multiply(conjugate(reconstruction), target);
}

Quaternion apply(const Quaternion& state, const Quaternion& increment, bool left) {
  return left ? multiply(increment, state) : multiply(state, increment);
}

std::array<double, 3> as_rotvec(const Quaternion& value) {
  std::array<double, 3> output{};
  to_rotvec(value, output.data());
  return output;
}

struct SO3Layout {
  const double* table;
  const int32_t* bins;
  int64_t width;
  double limit;
  bool smooth;
  bool rotation_vector;
  bool left;
  int32_t input_order;
  double projection_radius = std::numeric_limits<double>::infinity();
  bool lazy = false;              // momentum_lazy ranking (first order only)
  std::array<int32_t, 3> zero{};  // exact zero primitive per axis, used by ``lazy``
};

void project_primitive(double* value, double radius) {
  if (!std::isfinite(radius)) return;
  const double norm = std::sqrt(value[0]*value[0] + value[1]*value[1] + value[2]*value[2]);
  if (norm > radius) {
    const double scale = radius / norm;
    for (int j = 0; j < 3; ++j) value[j] *= scale;
  }
}

inline double primitive(const SO3Layout& layout, int axis, int32_t index) {
  return layout.table[static_cast<int64_t>(axis) * layout.width + index];
}

std::array<int32_t, 3> nearest_choice(const SO3Layout& layout,
                                      const std::array<double, 3>& request) {
  std::array<int32_t, 3> choice{};
  for (int axis = 0; axis < 3; ++axis) {
    double best = std::numeric_limits<double>::infinity();
    for (int32_t index = 0; index < layout.bins[axis]; ++index) {
      const double distance = std::fabs(request[axis] - primitive(layout, axis, index));
      if (distance < best) {
        best = distance;
        choice[axis] = index;
      }
    }
  }
  return choice;
}

std::vector<int32_t> smooth_axis(const SO3Layout& layout, int axis, double request,
                                 int32_t nearest) {
  std::vector<int32_t> output;
  bool has_nearest = false;
  const double radius = 2.0 * layout.limit + 1e-12;
  for (int32_t index = 0; index < layout.bins[axis]; ++index) {
    if (std::fabs(primitive(layout, axis, index) - request) <= radius) {
      output.push_back(index);
      has_nearest = has_nearest || index == nearest;
    }
  }
  if (!has_nearest) output.push_back(nearest);
  return output;
}

// Lexicographic selection key: (moving axes, zero distance) under ``lazy``, then error, cell.
struct ChoiceKey {
  int moving = 0;
  int distance = 0;
  double error = 0.0;
  std::array<int32_t, 3> choice{};
  bool operator<(const ChoiceKey& o) const {
    if (moving != o.moving) return moving < o.moving;
    if (distance != o.distance) return distance < o.distance;
    if (error != o.error) return error < o.error;
    return choice < o.choice;
  }
};

std::array<int32_t, 3> smooth_choice(const SO3Layout& layout,
                                     const std::array<double, 3>& request,
                                     const Quaternion& reconstruction,
                                     const Quaternion& target,
                                     const std::array<double, 3>& direction,
                                     const std::array<int32_t, 3>& nearest,
                                     const std::array<int32_t, 3>& previous = {-1, -1, -1}) {
  const auto x = smooth_axis(layout, 0, request[0], nearest[0]);
  const auto y = smooth_axis(layout, 1, request[1], nearest[1]);
  const auto z = smooth_axis(layout, 2, request[2], nearest[2]);
  ChoiceKey best_any, best_preferred;
  bool found_any = false, found_preferred = false;
  for (const int32_t ix : x) {
    for (const int32_t iy : y) {
      for (const int32_t iz : z) {
        const std::array<int32_t, 3> choice{ix, iy, iz};
        double vector[3] = {primitive(layout, 0, ix), primitive(layout, 1, iy),
                                  primitive(layout, 2, iz)};
        project_primitive(vector, layout.projection_radius);
        const Quaternion rebuilt = apply(reconstruction, from_rotvec(vector), layout.left);
        const auto remaining = as_rotvec(residual(target, rebuilt, layout.left));
        const double error = std::sqrt(remaining[0] * remaining[0] + remaining[1] * remaining[1] +
                                       remaining[2] * remaining[2]);
        if (error > layout.limit + 1e-12) continue;
        if (layout.lazy && choice == previous) return choice;
        ChoiceKey key;
        key.error = error;
        key.choice = choice;
        if (layout.lazy) {
          for (int axis = 0; axis < 3; ++axis) {
            key.moving += choice[axis] != layout.zero[axis];
            key.distance += std::abs(choice[axis] - layout.zero[axis]);
          }
        }
        if (!found_any || key < best_any) {
          best_any = key;
          found_any = true;
        }
        const double dot = vector[0] * direction[0] + vector[1] * direction[1] +
                           vector[2] * direction[2];
        if (dot >= -1e-15 && (!found_preferred || key < best_preferred)) {
          best_preferred = key;
          found_preferred = true;
        }
      }
    }
  }
  return found_preferred ? best_preferred.choice : (found_any ? best_any.choice : nearest);
}

void encode_trajectory(const SO3Layout& layout, const float* actions, int64_t steps,
                       int32_t* cells) {
  Quaternion truth{0.0, 0.0, 0.0, 1.0};
  Quaternion reconstruction = truth;
  std::array<double, 3> direction{};
  std::array<int32_t, 3> previous = layout.zero;
  for (int64_t step = 0; step < steps; ++step) {
    const Quaternion value = from_input(actions + 3 * step, layout.rotation_vector);
    truth = layout.input_order == 0 ? value : apply(truth, value, layout.left);
    const auto request = as_rotvec(residual(truth, reconstruction, layout.left));
    auto choice = nearest_choice(layout, request);
    if (layout.smooth) {
      choice = smooth_choice(layout, request, reconstruction, truth, direction, choice, previous);
    }
    previous = choice;
    const double vector[3] = {primitive(layout, 0, choice[0]),
                              primitive(layout, 1, choice[1]),
                              primitive(layout, 2, choice[2])};
    reconstruction = apply(reconstruction, from_rotvec(vector), layout.left);
    if (vector[0] != 0.0 || vector[1] != 0.0 || vector[2] != 0.0) {
      direction = {vector[0], vector[1], vector[2]};
    }
    for (int axis = 0; axis < 3; ++axis) cells[3 * step + axis] = choice[axis];
  }
}

}  // namespace

py::array_t<int32_t> so3_second_order_encode_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& table,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& bins,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& quaternions,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& previous_first_order,
    int32_t input_order, bool left, double limit, double radius, bool smooth, int threads) {
  if (table.ndim() != 2 || table.shape(0) != 3 || bins.size() != 3 ||
      quaternions.ndim() != 3 || quaternions.shape(2) != 4 ||
      previous_first_order.ndim() != 2 || previous_first_order.shape(1) != 4 ||
      previous_first_order.shape(0) != quaternions.shape(0) ||
      (input_order != 0 && input_order != 1) || !(limit > 0) || !(radius > 0)) {
    throw std::invalid_argument("invalid second-order SO(3) batch arguments");
  }
  for (int j = 0; j < 3; ++j) {
    if (bins.data()[j] < 1 || bins.data()[j] > table.shape(1))
      throw std::invalid_argument("SO(3) primitive count exceeds table width");
  }
  const SO3Layout layout{table.data(), bins.data(), table.shape(1), limit, smooth,
                         true, left, input_order, radius};
  const int64_t batch = quaternions.shape(0), steps = quaternions.shape(1);
  py::array_t<int32_t> output({batch, steps, int64_t{3}});
  const double* values = quaternions.data();
  const double* previous = previous_first_order.data();
  int32_t* result = output.mutable_data();
  const Quaternion identity{0, 0, 0, 1};
  py::gil_scoped_release release;
  ac2::parallel_for(batch, threads, [&](int64_t i) {
    Quaternion truth = identity, reconstruction = identity;
    Quaternion increment{previous[4*i], previous[4*i+1], previous[4*i+2], previous[4*i+3]};
    std::array<double, 3> direction{};
    for (int64_t t = 0; t < steps; ++t) {
      const double* raw = values + (i*steps+t)*4;
      const Quaternion value{raw[0], raw[1], raw[2], raw[3]};
      truth = input_order == 0 ? value : apply(truth, value, left);
      const auto desired = residual(truth, reconstruction, left);
      const auto correction = residual(desired, increment, left);
      const auto request = as_rotvec(correction);
      auto choice = nearest_choice(layout, request);
      if (smooth) {
        // Python scores U^-1 W for both conventions; use exactly that order.
        SO3Layout selection = layout;
        selection.left = false;
        choice = smooth_choice(selection, request, identity, correction, direction, choice);
      }
      double vector[3] = {primitive(layout, 0, choice[0]), primitive(layout, 1, choice[1]), primitive(layout, 2, choice[2])};
      project_primitive(vector, radius);
      increment = apply(increment, from_rotvec(vector), left);
      reconstruction = apply(reconstruction, increment, left);
      if (vector[0] != 0 || vector[1] != 0 || vector[2] != 0)
        direction = {vector[0], vector[1], vector[2]};
      for (int j = 0; j < 3; ++j) result[(i*steps+t)*3+j] = choice[j];
    }
  });
  return output;
}

py::array_t<int32_t> so3_encode_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& table,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& bins,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& actions,
    int32_t input_order, bool rotation_vector, bool left, double limit, int32_t selection,
    int threads) {
  if (table.ndim() != 2 || table.shape(0) != 3 || bins.size() != 3 || actions.ndim() != 3 ||
      actions.shape(2) != 3 || (input_order != 0 && input_order != 1) || limit <= 0.0 ||
      selection < 0 || selection > 2) {
    throw std::invalid_argument("invalid SO(3) batch encoder arguments");
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (bins.data()[axis] < 1 || bins.data()[axis] > table.shape(1)) {
      throw std::invalid_argument("SO(3) primitive count exceeds table width");
    }
  }
  // selection: 0 nearest, 1 smooth, 2 momentum_lazy
  SO3Layout layout{table.data(), bins.data(), table.shape(1), limit, selection != 0,
                   rotation_vector, left, input_order};
  layout.lazy = selection == 2;
  for (int axis = 0; axis < 3; ++axis) {
    double best = std::numeric_limits<double>::infinity();
    for (int32_t index = 0; index < bins.data()[axis]; ++index) {
      if (std::fabs(primitive(layout, axis, index)) < best) {
        best = std::fabs(primitive(layout, axis, index));
        layout.zero[axis] = index;
      }
    }
    if (layout.lazy && best != 0.0) {
      throw std::invalid_argument("momentum_lazy needs an exact zero primitive per axis");
    }
  }
  const int64_t batch = actions.shape(0);
  const int64_t steps = actions.shape(1);
  py::array_t<int32_t> output({batch, steps, int64_t{3}});
#ifndef _OPENMP
  (void)threads;
#endif
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t index) {
      encode_trajectory(layout, actions.data() + index * steps * 3, steps,
                        output.mutable_data() + index * steps * 3);
    });
  }
  return output;
}
