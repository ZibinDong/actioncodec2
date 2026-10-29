
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include "parallel.hpp"

namespace py = pybind11;

namespace {

struct SecondOrderAxis {
  const float* values;
  int32_t count;
  bool ordered;
  int32_t nonnegative;
  int32_t nonpositive;
};

template <bool Smooth>
inline int32_t pick_second_order_axis(const SecondOrderAxis& axis, double request,
                                     double direction, double epsilon) {
  const float* values = axis.values;
  if (axis.ordered) {
    const int32_t insertion = std::lower_bound(values, values + axis.count, request) - values;
    const int32_t left = std::max(0, insertion - 1);
    const int32_t right = std::min(insertion, axis.count - 1);
    int32_t nearest = std::fabs(request - values[left]) <= std::fabs(request - values[right])
                          ? left : right;
    // Rounded distance plateaus must keep the original first-cell tie-break.
    while (nearest > 0 && std::fabs(request - values[nearest - 1]) == std::fabs(request - values[nearest]))
      --nearest;
    if constexpr (!Smooth) return nearest;
    const int32_t preferred = direction > 0 ? std::max(nearest, axis.nonnegative)
                            : direction < 0 ? std::min(nearest, axis.nonpositive) : nearest;
    const double primitive = values[preferred];
    const double sign = primitive > 0 ? 1 : primitive < 0 ? -1 : 0;
    return sign * direction >= 0 && std::fabs(request - primitive) <= epsilon ? preferred : nearest;
  }
  // Small tables are faster to scan; arbitrary stored order also uses this path.
  int32_t nearest = 0, feasible = -1, preferred = -1;
  double nearest_distance = INFINITY, feasible_distance = INFINITY, preferred_distance = INFINITY;
  for (int32_t k = 0; k < axis.count; ++k) {
    const double primitive = values[k];
    const double distance = std::fabs(request - primitive);
    if (distance < nearest_distance) { nearest = k; nearest_distance = distance; }
    if constexpr (!Smooth) continue;
    if (distance > epsilon) continue;
    if (distance < feasible_distance) { feasible = k; feasible_distance = distance; }
    const double sign = primitive > 0 ? 1 : primitive < 0 ? -1 : 0;
    if (sign * direction >= 0 && distance < preferred_distance) {
      preferred = k;
      preferred_distance = distance;
    }
  }
  return preferred >= 0 ? preferred : feasible >= 0 ? feasible : nearest;
}

}  // namespace

py::array_t<int32_t> additive_second_order_encode_batch(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& table,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& bins,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& previous_first_order,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& epsilon,
    bool smooth, int threads) {
  if (positions.ndim() != 3 || table.ndim() != 2 || previous_first_order.ndim() != 2)
    throw std::invalid_argument("invalid second-order additive batch shapes");
  const int64_t batch = positions.shape(0), steps = positions.shape(1), dim = positions.shape(2);
  if (table.shape(0) != dim || bins.size() != dim || epsilon.size() != dim ||
      previous_first_order.shape(0) != batch || previous_first_order.shape(1) != dim)
    throw std::invalid_argument("inconsistent second-order additive batch dimensions");
  for (int64_t j = 0; j < dim; ++j) {
    if (bins.data()[j] < 1 || bins.data()[j] > table.shape(1))
      throw std::invalid_argument("primitive count exceeds table width");
  }
  py::array_t<int32_t> output({batch, steps, dim});
  const auto* targets = positions.data();
  const auto* initial = previous_first_order.data();
  const auto* axes = table.data();
  const auto* counts = bins.data();
  const auto* limits = epsilon.data();
  auto* cells = output.mutable_data();
  const int64_t width = table.shape(1);
  std::vector<SecondOrderAxis> search;
  for (int64_t j = 0; j < dim; ++j) {
    const float* begin = axes + j * width;
    const float* end = begin + counts[j];
    const bool ordered = counts[j] > 16 &&
        std::adjacent_find(begin, end, [](float a, float b) { return a >= b; }) == end;
    const int32_t nonnegative = ordered ? std::lower_bound(begin, end, 0.0f) - begin : 0;
    const int32_t nonpositive = ordered ? std::upper_bound(begin, end, 0.0f) - begin - 1 : 0;
    search.push_back({begin, counts[j], ordered, std::min(nonnegative, counts[j] - 1),
                      std::max(nonpositive, 0)});
  }
  py::gil_scoped_release release;
  ac2::parallel_for(batch, threads, [&](int64_t i) {
    std::vector<double> position(dim, 0), velocity(initial+i*dim, initial+(i+1)*dim), direction(dim, 0);
    for (int64_t t = 0; t < steps; ++t) {
      for (int64_t j = 0; j < dim; ++j) {
        const double request = targets[(i*steps+t)*dim+j] - (position[j]+velocity[j]);
        const int32_t choice = smooth
            ? pick_second_order_axis<true>(search[j], request, direction[j], limits[j])
            : pick_second_order_axis<false>(search[j], request, direction[j], limits[j]);
        const double primitive = axes[j*width+choice];
        velocity[j] += primitive;
        position[j] += velocity[j];
        if (primitive != 0) direction[j] = primitive > 0 ? 1 : -1;
        cells[(i*steps+t)*dim+j] = choice;
      }
    }
  });
  return output;
}

namespace {

struct Layout {
  int32_t dimension;
  int32_t width;                  // padded primitive-table stride
  const float* table;             // [dimension * width], +inf padding
  const int32_t* bins;            // [dimension]
  const uint8_t* binary;          // [dimension], 1 for absolute two-valued axes
  const float* lower;
  const float* upper;
  const float* epsilon;           // [dimension]; null unless `smooth` is set
  bool smooth;                    // pick by FirstOrderCodec._select_smooth
};

// Matches PhysicalCodecBase.SOURCE_TOLERANCE.
constexpr float kSourceTolerance = 8.0F * FLT_EPSILON;

inline int32_t nearest(const Layout& layout, int32_t axis, float request) {
  const float* values = layout.table + static_cast<size_t>(axis) * layout.width;
  const int32_t count = layout.bins[axis];
  int32_t best = 0;
  float best_distance = std::fabs(values[0] - request);
  for (int32_t index = 1; index < count; ++index) {
    const float distance = std::fabs(values[index] - request);
    if (distance < best_distance) {
      best_distance = distance;
      best = index;
    }
  }
  return best;
}

inline int32_t select_smooth(const Layout& layout, int32_t axis, float request, float direction) {
  const float* values = layout.table + static_cast<size_t>(axis) * layout.width;
  const int32_t count = layout.bins[axis];
  const float radius = layout.epsilon[axis] + kSourceTolerance;
  int32_t nearest_index = 0;
  float nearest_distance = std::fabs(values[0] - request);
  bool any_feasible = false;
  int32_t feasible_index = 0;
  float feasible_distance = 0.0F;
  bool any_preferred = false;
  int32_t preferred_index = 0;
  float preferred_distance = 0.0F;
  for (int32_t index = 0; index < count; ++index) {
    const float distance = std::fabs(values[index] - request);
    if (index > 0 && distance < nearest_distance) {
      nearest_distance = distance;
      nearest_index = index;
    }
    if (!(distance <= radius)) continue;
    if (!any_feasible || distance < feasible_distance) {
      any_feasible = true;
      feasible_distance = distance;
      feasible_index = index;
    }
    // np.sign(value) * direction >= 0: a zero primitive never reverses.
    const float sign = values[index] > 0.0F ? 1.0F : (values[index] < 0.0F ? -1.0F : 0.0F);
    if (!(sign * direction >= 0.0F)) continue;
    if (!any_preferred || distance < preferred_distance) {
      any_preferred = true;
      preferred_distance = distance;
      preferred_index = index;
    }
  }
  if (any_preferred) return preferred_index;
  if (any_feasible) return feasible_index;
  return nearest_index;
}

void encode_first_order(const Layout& layout, const float* positions, int64_t steps,
                        const float* initial, int32_t* cells) {
  const int32_t dimension = layout.dimension;
  std::vector<float> state(initial, initial + dimension);
  // Sign of the last non-zero primitive each coordinate emitted.
  std::vector<float> direction(dimension, 0.0F);
  for (int64_t step = 0; step < steps; ++step) {
    const float* target = positions + step * dimension;
    int32_t* row = cells + step * dimension;
    for (int32_t axis = 0; axis < dimension; ++axis) {
      const float* values = layout.table + static_cast<size_t>(axis) * layout.width;
      if (layout.binary[axis]) {
        const int32_t cell = nearest(layout, axis, target[axis]);
        state[axis] = values[cell];
        row[axis] = cell;
        continue;
      }
      const float request = target[axis] - state[axis];
      const int32_t cell = layout.smooth ? select_smooth(layout, axis, request, direction[axis])
                                         : nearest(layout, axis, request);
      const float picked = values[cell];
      state[axis] = std::min(std::max(state[axis] + picked, layout.lower[axis]),
                             layout.upper[axis]);
      if (picked != 0.0F) direction[axis] = picked > 0.0F ? 1.0F : -1.0F;
      row[axis] = cell;
    }
  }
}

void decode_first_order(const Layout& layout, const int32_t* cells, int64_t steps,
                        const float* initial, float* positions) {
  const int32_t dimension = layout.dimension;
  std::vector<float> state(initial, initial + dimension);
  for (int64_t step = 0; step < steps; ++step) {
    const int32_t* row = cells + step * dimension;
    float* out = positions + step * dimension;
    for (int32_t axis = 0; axis < dimension; ++axis) {
      const float* values = layout.table + static_cast<size_t>(axis) * layout.width;
      const float primitive = values[row[axis]];
      if (layout.binary[axis]) {
        state[axis] = primitive;
      } else {
        state[axis] = std::min(std::max(state[axis] + primitive, layout.lower[axis]),
                               layout.upper[axis]);
      }
      out[axis] = state[axis];
    }
  }
}

void encode_second_order(const Layout& layout, const float* positions, int64_t steps,
                         const float* initial, const float* initial_velocity, int32_t* cells) {
  const int32_t dimension = layout.dimension;
  std::vector<float> source(initial, initial + dimension);
  std::vector<float> source_velocity(initial_velocity, initial_velocity + dimension);
  std::vector<float> recon(initial, initial + dimension);
  std::vector<float> recon_velocity(initial_velocity, initial_velocity + dimension);
  std::vector<float> error(dimension, 0.0F);
  std::vector<float> previous(dimension, 0.0F);
  for (int64_t step = 0; step < steps; ++step) {
    const float* target = positions + step * dimension;
    int32_t* row = cells + step * dimension;
    for (int32_t axis = 0; axis < dimension; ++axis) {
      const float* values = layout.table + static_cast<size_t>(axis) * layout.width;
      if (layout.binary[axis]) {
        row[axis] = nearest(layout, axis, target[axis]);
        continue;
      }
      const float velocity = target[axis] - source[axis];
      const float request =
          2.0F * error[axis] - previous[axis] + (velocity - source_velocity[axis]);
      const int32_t cell = nearest(layout, axis, request);
      recon_velocity[axis] += values[cell];
      recon[axis] += recon_velocity[axis];
      source[axis] = target[axis];
      source_velocity[axis] = velocity;
      previous[axis] = error[axis];
      error[axis] = source[axis] - recon[axis];
      row[axis] = cell;
    }
  }
}

void decode_second_order(const Layout& layout, const int32_t* cells, int64_t steps,
                         const float* initial, const float* initial_velocity, float* positions) {
  const int32_t dimension = layout.dimension;
  std::vector<float> state(initial, initial + dimension);
  std::vector<float> velocity(initial_velocity, initial_velocity + dimension);
  for (int64_t step = 0; step < steps; ++step) {
    const int32_t* row = cells + step * dimension;
    float* out = positions + step * dimension;
    for (int32_t axis = 0; axis < dimension; ++axis) {
      const float primitive = layout.table[static_cast<size_t>(axis) * layout.width + row[axis]];
      if (layout.binary[axis]) {
        out[axis] = primitive;
        continue;
      }
      velocity[axis] += primitive;
      state[axis] += velocity[axis];
      out[axis] = state[axis];
    }
  }
}

Layout make_layout(const py::array_t<float>& table, const py::array_t<int32_t>& bins,
                   const py::array_t<uint8_t>& binary, const py::array_t<float>& lower,
                   const py::array_t<float>& upper, const py::array_t<float>& epsilon,
                   bool smooth) {
  Layout layout;
  layout.dimension = static_cast<int32_t>(table.shape(0));
  layout.width = static_cast<int32_t>(table.shape(1));
  layout.table = table.data();
  layout.bins = bins.data();
  layout.binary = binary.data();
  layout.lower = lower.data();
  layout.upper = upper.data();
  layout.epsilon = epsilon.size() == 0 ? nullptr : epsilon.data();
  layout.smooth = smooth && layout.epsilon != nullptr;
  if (smooth && layout.epsilon == nullptr) {
    throw std::invalid_argument("smooth selection needs a per-coordinate epsilon");
  }
  if (layout.epsilon != nullptr && epsilon.size() != layout.dimension) {
    throw std::invalid_argument("epsilon must have one entry per coordinate");
  }
  return layout;
}

using Array2D = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Array3D = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Cells3D = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

}  // namespace

py::array_t<float> physical_decode_batch(const py::array_t<float>& table,
                                         const py::array_t<int32_t>& bins,
                                         const py::array_t<uint8_t>& binary,
                                         const py::array_t<float>& lower,
                                         const py::array_t<float>& upper, const Cells3D& cells,
                                         const Array2D& initial,
                                         const Array2D& initial_velocity, int32_t order,
                                         int threads) {
  if (cells.ndim() != 3 || cells.shape(2) != table.shape(0) || initial.ndim() != 2 ||
      initial_velocity.ndim() != 2 || initial.shape(0) != cells.shape(0) ||
      initial_velocity.shape(0) != cells.shape(0) || initial.shape(1) != cells.shape(2) ||
      initial_velocity.shape(1) != cells.shape(2) || (order != 1 && order != 2)) {
    throw std::invalid_argument("invalid physical decode batch shapes or order");
  }
  const Layout layout =
      make_layout(table, bins, binary, lower, upper, py::array_t<float>(), false);
  const int64_t batch = cells.shape(0);
  const int64_t steps = cells.shape(1);
  const int64_t dimension = layout.dimension;
  py::array_t<float> positions({batch, steps, dimension});
#ifndef _OPENMP
  (void)threads;
#endif
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t index) {
      const int32_t* source = cells.data() + index * steps * dimension;
      float* target = positions.mutable_data() + index * steps * dimension;
      if (order == 1) {
        decode_first_order(layout, source, steps, initial.data() + index * dimension, target);
      } else {
        decode_second_order(layout, source, steps, initial.data() + index * dimension,
                            initial_velocity.data() + index * dimension, target);
      }
    });
  }
  return positions;
}

// One thread per chunk.  ``positions`` is [B, T, D] and ``initial`` is [B, D].
py::array_t<int32_t> physical_encode_batch(const py::array_t<float>& table,
                                           const py::array_t<int32_t>& bins,
                                           const py::array_t<uint8_t>& binary,
                                           const py::array_t<float>& lower,
                                           const py::array_t<float>& upper,
                                           const Array3D& positions, const Array2D& initial,
                                           const Array2D& initial_velocity, int32_t order,
                                           int threads, const py::array_t<float>& epsilon,
                                           bool smooth) {
  const Layout layout = make_layout(table, bins, binary, lower, upper, epsilon, smooth);
  const int64_t batch = positions.shape(0);
  const int64_t steps = positions.shape(1);
  const int64_t dimension = layout.dimension;
  py::array_t<int32_t> cells({batch, steps, dimension});
  const float* source = positions.data();
  const float* starts = initial.data();
  const float* velocities = initial_velocity.data();
  int32_t* output = cells.mutable_data();
#ifndef _OPENMP
  (void)threads;
#endif
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t index) {
      const float* chunk = source + index * steps * dimension;
      int32_t* target = output + index * steps * dimension;
      if (order == 1) {
        encode_first_order(layout, chunk, steps, starts + index * dimension, target);
      } else {
        encode_second_order(layout, chunk, steps, starts + index * dimension,
                            velocities + index * dimension, target);
      }
    });
  }
  return cells;
}
