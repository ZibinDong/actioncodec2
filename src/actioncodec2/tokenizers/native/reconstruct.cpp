#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "parallel.hpp"
#include "quaternion.hpp"

namespace py = pybind11;
namespace {
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

void validate_reconstruction(const F64& table, const I32& bins, const I64& cells,
                             const F32& previous, const F32& scale) {
  if (table.ndim() != 2 || table.shape(0) < 1 || table.shape(1) < 1 ||
      bins.ndim() != 1 || bins.size() != table.shape(0) ||
      cells.ndim() != 3 || cells.shape(0) < 1 || cells.shape(1) < 1 ||
      cells.shape(2) != table.shape(0) || previous.ndim() != 2 ||
      previous.shape(0) != cells.shape(0) || previous.shape(1) != cells.shape(2) ||
      scale.ndim() != 1 || scale.size() != cells.shape(2)) {
    throw std::invalid_argument("invalid reconstruction shapes");
  }
  for (py::ssize_t d = 0; d < bins.size(); ++d) {
    if (bins.data()[d] < 1 || bins.data()[d] > table.shape(1))
      throw std::invalid_argument("primitive count exceeds table width");
    if (!std::isfinite(scale.data()[d]) || scale.data()[d] <= 0)
      throw std::invalid_argument("scale must be positive and finite");
    for (int32_t k = 0; k < bins.data()[d]; ++k) {
      if (!std::isfinite(table.data()[d * table.shape(1) + k]))
        throw std::invalid_argument("active primitives must be finite");
    }
  }
  for (py::ssize_t i = 0; i < previous.size(); ++i) {
    if (!std::isfinite(previous.data()[i]))
      throw std::invalid_argument("previous first-order values must be finite");
  }
  for (py::ssize_t i = 0; i < cells.size(); ++i) {
    if (cells.data()[i] < 0 || cells.data()[i] >= bins.data()[i % bins.size()])
      throw std::invalid_argument("cell index is outside its primitive table");
  }
}

F32 so3_reconstruct(const F64& table, const I32& bins, const I64& cells,
                    const F32& previous, const F32& scale, int input_order,
                    int primitive_order, bool left, bool rotation_vector, double radius) {
  validate_reconstruction(table, bins, cells, previous, scale);
  if (table.shape(0) != 3 || (input_order != 0 && input_order != 1) ||
      (primitive_order != 1 && primitive_order != 2) ||
      std::isnan(radius) || radius <= 0)
    throw std::invalid_argument("invalid SO(3) reconstruction order or radius");
  const int64_t batch = cells.shape(0), steps = cells.shape(1);
  F32 output({batch, steps, int64_t{3}});
  auto* target = output.mutable_data();
  py::gil_scoped_release release;
  ac2::parallel_for(batch, 0, [&](int64_t b) {
    double boundary[3];
    for (int d = 0; d < 3; ++d)
      boundary[d] = static_cast<double>(previous.data()[3 * b + d]) * scale.data()[d];
    auto increment = rotation_vector ? ac2q::from_rotvec(boundary) : ac2q::from_euler(boundary);
    ac2q::Q state{0, 0, 0, 1};
    float accumulated[3] = {0, 0, 0};
    auto apply = [&](ac2q::Q a, ac2q::Q v) {
      return left ? ac2q::multiply(v, a) : ac2q::multiply(a, v);
    };
    for (int64_t t = 0; t < steps; ++t) {
      double v[3], represented[3];
      for (int d = 0; d < 3; ++d)
        v[d] = table.data()[d * table.shape(1) + cells.data()[(b * steps + t) * 3 + d]];
      if (primitive_order == 2) {
        const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double factor = std::min(1., radius / std::max(norm, std::numeric_limits<double>::min()));
        for (double& x : v) x *= factor;
        increment = apply(increment, ac2q::from_rotvec(v));
        state = apply(state, increment);
        auto q = input_order == 1 ? increment : state;
        if (rotation_vector) ac2q::to_rotvec(q, represented);
        else ac2q::to_euler(q, represented);
      } else if (input_order == 1) {
        if (rotation_vector) std::copy(v, v + 3, represented);
        else {
          for (double& x : v) x = static_cast<float>(x);
          ac2q::to_euler(ac2q::from_rotvec(v), represented);
        }
      } else {
        state = apply(state, ac2q::from_rotvec(v));
        if (rotation_vector) ac2q::to_rotvec(state, represented);
        else ac2q::to_euler(state, represented);
      }
      for (int d = 0; d < 3; ++d) {
        float value = static_cast<float>(represented[d] / scale.data()[d]);
        if (primitive_order == 1 && input_order == 1) {
          const float next = accumulated[d] + value;
          value = next - accumulated[d];
          accumulated[d] = next;
        }
        target[(b * steps + t) * 3 + d] = value;
      }
    }
  });
  return output;
}

F32 additive_reconstruct(const F64& table, const I32& bins, const I64& cells,
                         const F32& previous, const F32& scale, int input_order) {
  validate_reconstruction(table, bins, cells, previous, scale);
  if (input_order < 0 || input_order > 2)
    throw std::invalid_argument("invalid additive reconstruction order");
  const int64_t batch = cells.shape(0), steps = cells.shape(1), width = cells.shape(2);
  F32 output({batch, steps, width});
  auto* target = output.mutable_data();
  py::gil_scoped_release release;
  ac2::parallel_for(batch, 0, [&](int64_t b) {
    for (int64_t d = 0; d < width; ++d) {
      double change_sum = 0, position = 0;
      const float boundary = previous.data()[b * width + d] * scale.data()[d];
      for (int64_t t = 0; t < steps; ++t) {
        const int64_t index = (b * steps + t) * width + d;
        const double primitive = table.data()[d * table.shape(1) + cells.data()[index]];
        change_sum += primitive;
        const double velocity = change_sum + boundary;
        position += velocity;
        const double value = input_order == 2 ? primitive : input_order == 1 ? velocity : position;
        target[index] = static_cast<float>(value / scale.data()[d]);
      }
    }
  });
  return output;
}
}

void register_reconstruction(py::module_& module) {
  module.def("so3_reconstruct", &so3_reconstruct, py::arg("table"), py::arg("bins"),
             py::arg("cells"), py::arg("previous_first_order"), py::arg("scale"),
             py::arg("input_order"), py::arg("primitive_order"), py::arg("left"),
             py::arg("rotation_vector"), py::arg("radius"));
  module.def("additive_reconstruct", &additive_reconstruct, py::arg("table"), py::arg("bins"),
             py::arg("cells"), py::arg("previous_first_order"), py::arg("scale"),
             py::arg("input_order"));
}
