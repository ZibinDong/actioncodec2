// pybind11 entry point for the ActionCodec2 native kernels.
//
// Every function here has an exactly equivalent Python implementation in the
// package.  The native module is a pure accelerator: if it is not built, the
// package still works, and the test suite asserts the two agree bit for bit.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

void register_resample(py::module_&);
void register_setbpe(py::module_&);
void register_inference(py::module_&);
void register_preprocessing(py::module_&);
void register_grammar(py::module_&);
void register_sparse(py::module_&);

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

py::tuple train_spatial_native(py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                               py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                               const std::vector<int32_t>&, int64_t, int);

py::tuple train_temporal_native(py::iterable, int64_t,
                                py::array_t<uint64_t, py::array::c_style | py::array::forcecast>,
                                py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                                int32_t, int32_t, int64_t, int64_t, int);

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
    double, int32_t, int);

py::tuple train_bpe_native(py::iterable, const std::vector<int32_t>&, int64_t, int64_t, int);

py::tuple spatial_encode_instances(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&,
    const std::vector<int32_t>&,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&, int);

PYBIND11_MODULE(_native, module) {
  register_resample(module);
  register_setbpe(module);
  register_inference(module);
  register_preprocessing(module);
  register_grammar(module);
  register_sparse(module);
  module.doc() = "ActionCodec2 native kernels (Set-BPE training and batched physical stage)";
  module.def("so3_second_order_encode_batch", &so3_second_order_encode_batch,
             py::arg("table"), py::arg("bins"), py::arg("quaternions"),
             py::arg("previous_first_order"), py::arg("input_order"), py::arg("left"),
             py::arg("limit"), py::arg("radius"), py::arg("smooth"), py::arg("threads") = 0);
  module.def("additive_second_order_encode_batch", &additive_second_order_encode_batch,
             py::arg("table"), py::arg("bins"), py::arg("positions"),
             py::arg("previous_first_order"), py::arg("epsilon"), py::arg("smooth"),
             py::arg("threads") = 0);

  module.def("train_spatial", &train_spatial_native, py::arg("codes"), py::arg("weights"),
             py::arg("bins"), py::arg("vocab_size"), py::arg("threads") = 0,
             "Exact greedy spatial Set-BPE; returns (merges, report).");

  module.def("train_temporal", &train_temporal_native, py::arg("instance_tokens"),
             py::arg("total_anchors"), py::arg("spatial_support"),
             py::arg("spatial_anchor"), py::arg("dimension"), py::arg("lookahead"),
             py::arg("vocab_size"), py::arg("action_count"), py::arg("threads") = 0,
             "Exact greedy rooted temporal Set-BPE; returns (merges, report).");

  module.def("train_bpe", &train_bpe_native, py::arg("cell_chunks"), py::arg("bins"),
             py::arg("vocab_size"), py::arg("max_token_cells"), py::arg("threads") = 0,
             "Exact greedy BPE over the row-major flattened cell stream; returns (merges, report).");

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
             py::arg("left"), py::arg("limit"), py::arg("selection"), py::arg("threads") = 0,
             "Batched exact SO(3) error-feedback encoding; selection 0 nearest, 1 smooth, "
             "2 momentum_lazy.");

  module.def("spatial_encode_instances", &spatial_encode_instances, py::arg("codes"),
             py::arg("row_anchors"), py::arg("bins"), py::arg("merges"),
             py::arg("threads") = 0);
}
