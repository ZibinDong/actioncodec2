#pragma once

#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac2 {
namespace py = pybind11;

// Preserve the public Python integer contract; never coerce floats or booleans.
inline std::vector<std::vector<int64_t>> checked_token_streams(
    const py::iterable& streams, int64_t vocabulary, bool routed = false) {
  const py::object numpy_integer = py::module_::import("numpy").attr("integer");
  py::object integral;
  std::vector<std::vector<int64_t>> output;
  if (PyList_Check(streams.ptr())) output.reserve(PyList_GET_SIZE(streams.ptr()));
  for (py::handle stream : streams) {
    std::vector<int64_t> row;
    if (PyList_Check(stream.ptr())) row.reserve(PyList_GET_SIZE(stream.ptr()));
    for (py::handle raw : py::reinterpret_borrow<py::iterable>(stream)) {
      bool integer = PyLong_Check(raw.ptr()) || py::isinstance(raw, numpy_integer);
      if (!integer && !routed) {
        if (!integral) integral = py::module_::import("numbers").attr("Integral");
        integer = py::isinstance(raw, integral);
      }
      if (PyBool_Check(raw.ptr()) || !integer)
        throw std::invalid_argument(routed ? "tokens[" + std::to_string(row.size()) + "] must be an integer"
                                           : "tokens must be integers");
      py::object number = py::reinterpret_steal<py::object>(PyNumber_Long(raw.ptr()));
      if (!number) throw py::error_already_set();
      int overflow = 0;
      const int64_t token = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
      if (PyErr_Occurred()) throw py::error_already_set();
      if (overflow || token < 0 || token >= vocabulary)
        throw std::invalid_argument(routed ? "tokens[" + std::to_string(row.size()) +
                                           "] is outside the routed profile vocabularies; out of range [0, " + std::to_string(vocabulary) + ")"
                                           : "token is outside the vocabulary");
      row.push_back(token);
    }
    if (row.empty()) throw std::invalid_argument("token stream must not be empty");
    output.push_back(std::move(row));
  }
  if (output.empty()) throw std::invalid_argument("streams must not be empty");
  return output;
}
}  // namespace ac2
