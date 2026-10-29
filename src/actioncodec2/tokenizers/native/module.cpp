#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include "parallel.hpp"

namespace py = pybind11;

py::array_t<int32_t> so3_second_order_encode_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>&,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&,
    int32_t, bool, double, double, bool, int);

py::array_t<int32_t> additive_second_order_encode_batch(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&, bool, int);

py::array_t<float> physical_decode_batch(
    const py::array_t<float>&, const py::array_t<int32_t>&, const py::array_t<uint8_t>&,
    const py::array_t<float>&, const py::array_t<float>&,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&, int32_t, int);

py::array_t<int32_t> physical_encode_batch(
    const py::array_t<float>&, const py::array_t<int32_t>&, const py::array_t<uint8_t>&,
    const py::array_t<float>&, const py::array_t<float>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&, int32_t, int,
    const py::array_t<float>&, bool);

py::array_t<int32_t> so3_encode_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>&,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&, int32_t, bool, bool,
    double, bool, int);

void register_setbpe(py::module_& module);
void register_resample(py::module_& module);
void register_reconstruction(py::module_& module);
void register_grammar(py::module_& module);

PYBIND11_MODULE(_actioncodec2_native, module) {
  module.doc() = "ActionCodec2 encode/decode kernels: physical stage, resampling, Set-BPE";
  register_setbpe(module);
  register_resample(module);
  register_reconstruction(module);
  register_grammar(module);
  module.def(
      "set_thread_budget",
      [](int threads) {
        if (threads < 0) throw std::invalid_argument("thread budget must be non-negative");
        ac2::thread_budget().store(threads);
      },
      py::arg("threads"),
      "Cap threads per call for this process; 0 uses every CPU it may run on.");
  module.def("thread_budget", [] { return ac2::thread_budget().load(); });
  module.def("plan_threads", &ac2::plan_threads, py::arg("count"), py::arg("requested"),
             py::arg("seconds_each"), "Threads a call would use for ``count`` items.");
  module.def("so3_second_order_encode_batch", &so3_second_order_encode_batch,
             py::arg("table"), py::arg("bins"), py::arg("quaternions"),
             py::arg("previous_first_order"), py::arg("input_order"), py::arg("left"),
             py::arg("limit"), py::arg("radius"), py::arg("smooth"), py::arg("threads") = 0);
  module.def("additive_second_order_encode_batch", &additive_second_order_encode_batch,
             py::arg("table"), py::arg("bins"), py::arg("positions"),
             py::arg("previous_first_order"), py::arg("epsilon"), py::arg("smooth"),
             py::arg("threads") = 0);
  module.def("physical_decode_batch", &physical_decode_batch, py::arg("table"),
             py::arg("bins"), py::arg("binary"), py::arg("lower"), py::arg("upper"),
             py::arg("cells"), py::arg("initial"), py::arg("initial_velocity"),
             py::arg("order"), py::arg("threads") = 0);
  module.def("physical_encode_batch", &physical_encode_batch, py::arg("table"), py::arg("bins"),
             py::arg("binary"), py::arg("lower"), py::arg("upper"), py::arg("positions"),
             py::arg("initial"), py::arg("initial_velocity"), py::arg("order"),
             py::arg("threads") = 0, py::arg("epsilon") = py::array_t<float>(),
             py::arg("smooth") = false);
  module.def("so3_encode_batch", &so3_encode_batch, py::arg("table"), py::arg("bins"),
             py::arg("actions"), py::arg("input_order"), py::arg("rotation_vector"),
             py::arg("left"), py::arg("limit"), py::arg("smooth"), py::arg("threads") = 0);
}
