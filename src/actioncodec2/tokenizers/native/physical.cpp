#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
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

    while (nearest > 0 && std::fabs(request - values[nearest - 1]) == std::fabs(request - values[nearest]))
      --nearest;
    if constexpr (!Smooth) return nearest;
    const int32_t preferred = direction > 0 ? std::max(nearest, axis.nonnegative)
                            : direction < 0 ? std::min(nearest, axis.nonpositive) : nearest;
    const double primitive = values[preferred];
    const double sign = primitive > 0 ? 1 : primitive < 0 ? -1 : 0;
    return sign * direction >= 0 && std::fabs(request - primitive) <= epsilon ? preferred : nearest;
  }

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

}

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
  int32_t width;
  const float* table;
  const int32_t* bins;
  const uint8_t* binary;
  const float* lower;
  const float* upper;
  const float* epsilon;
  bool smooth;
};

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
  if (table.ndim() != 2 || table.shape(0) < 1 || table.shape(1) < 1 ||
      table.shape(0) > std::numeric_limits<int32_t>::max() ||
      table.shape(1) > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("table must have non-empty shape [D, K]");
  }
  for (const py::array* array : {static_cast<const py::array*>(&table),
                               static_cast<const py::array*>(&bins),
                               static_cast<const py::array*>(&binary),
                               static_cast<const py::array*>(&lower),
                               static_cast<const py::array*>(&upper),
                               static_cast<const py::array*>(&epsilon)}) {
    if (!(array->flags() & py::array::c_style)) {
      throw std::invalid_argument("physical layout arrays must be C-contiguous");
    }
  }
  const auto dimension = table.shape(0);
  if (bins.ndim() != 1 || binary.ndim() != 1 || lower.ndim() != 1 || upper.ndim() != 1 ||
      bins.size() != dimension || binary.size() != dimension ||
      lower.size() != dimension || upper.size() != dimension) {
    throw std::invalid_argument("layout vectors must have shape [D]");
  }
  if ((epsilon.size() && (epsilon.ndim() != 1 || epsilon.size() != dimension)) ||
      (smooth && !epsilon.size())) {
    throw std::invalid_argument("smooth selection needs epsilon with shape [D]");
  }
  for (py::ssize_t axis = 0; axis < dimension; ++axis) {
    const int32_t count = bins.data()[axis];
    if (count < 1 || count > table.shape(1) || binary.data()[axis] > 1 ||
        !(lower.data()[axis] <= upper.data()[axis])) {
      throw std::invalid_argument("invalid physical layout bounds or primitive count");
    }
    if (epsilon.size() &&
        !(std::isfinite(epsilon.data()[axis]) && epsilon.data()[axis] >= 0)) {
      throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    for (int32_t cell = 0; cell < count; ++cell) {
      if (!std::isfinite(table.data()[axis * table.shape(1) + cell])) {
        throw std::invalid_argument("active primitives must be finite");
      }
    }
  }
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
  return layout;
}

using Array2D = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Array3D = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Cells3D = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

}

py::array_t<float> physical_decode_batch(const py::array_t<float>& table,
                                         const py::array_t<int32_t>& bins,
                                         const py::array_t<uint8_t>& binary,
                                         const py::array_t<float>& lower,
                                         const py::array_t<float>& upper, const Cells3D& cells,
                                         const Array2D& initial,
                                         const Array2D& initial_velocity, int32_t order,
                                         int threads) {
  const Layout layout =
      make_layout(table, bins, binary, lower, upper, py::array_t<float>(), false);
  if (cells.ndim() != 3 || cells.shape(1) < 1 || cells.shape(2) != layout.dimension || initial.ndim() != 2 ||
      initial_velocity.ndim() != 2 || initial.shape(0) != cells.shape(0) ||
      initial_velocity.shape(0) != cells.shape(0) || initial.shape(1) != cells.shape(2) ||
      initial_velocity.shape(1) != cells.shape(2) || (order != 1 && order != 2)) {
    throw std::invalid_argument("invalid physical decode batch shapes or order");
  }
  const int64_t batch = cells.shape(0);
  const int64_t steps = cells.shape(1);
  const int64_t dimension = layout.dimension;
  for (py::ssize_t index = 0; index < initial.size(); ++index) {
    if (!std::isfinite(initial.data()[index]) || !std::isfinite(initial_velocity.data()[index])) {
      throw std::invalid_argument("initial state and velocity must be finite");
    }
  }
  for (py::ssize_t index = 0; index < cells.size(); ++index) {
    const int32_t cell = cells.data()[index];
    if (cell < 0 || cell >= layout.bins[index % dimension]) {
      throw std::invalid_argument("cell index is outside its primitive table");
    }
  }
  py::array_t<float> positions({batch, steps, dimension});
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
  if (positions.ndim() != 3 || positions.shape(1) < 1 ||
      positions.shape(2) != layout.dimension || initial.ndim() != 2 ||
      initial_velocity.ndim() != 2 || initial.shape(0) != positions.shape(0) ||
      initial_velocity.shape(0) != positions.shape(0) ||
      initial.shape(1) != layout.dimension || initial_velocity.shape(1) != layout.dimension ||
      (order != 1 && order != 2)) {
    throw std::invalid_argument("invalid physical encode batch shapes or order");
  }
  for (py::ssize_t index = 0; index < positions.size(); ++index) {
    if (!std::isfinite(positions.data()[index])) {
      throw std::invalid_argument("positions must be finite");
    }
  }
  for (py::ssize_t index = 0; index < initial.size(); ++index) {
    if (!std::isfinite(initial.data()[index]) || !std::isfinite(initial_velocity.data()[index])) {
      throw std::invalid_argument("initial state and velocity must be finite");
    }
  }
  const int64_t batch = positions.shape(0);
  const int64_t steps = positions.shape(1);
  const int64_t dimension = layout.dimension;
  py::array_t<int32_t> cells({batch, steps, dimension});
  const float* source = positions.data();
  const float* starts = initial.data();
  const float* velocities = initial_velocity.data();
  int32_t* output = cells.mutable_data();
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
