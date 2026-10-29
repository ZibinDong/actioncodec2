// Parallel replay of a learned Spatial Set-BPE vocabulary over action cells.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

struct Replay {
  int32_t dimension;
  int32_t vocab_size;
  std::vector<int32_t> offsets;
  std::vector<int32_t> rule;
  std::vector<int32_t> anchor;

  int32_t child(int32_t left, int32_t right) const {
    if (left > right) std::swap(left, right);
    return rule[static_cast<size_t>(left) * vocab_size + right];
  }

  std::vector<int32_t> encode(const int32_t* code) const {
    std::vector<int32_t> active;
    active.reserve(dimension);
    for (int32_t axis = 0; axis < dimension; ++axis) {
      if (code[axis] < 0) continue;
      active.push_back(offsets[axis] + code[axis]);
    }
    using Candidate = std::tuple<int32_t, int32_t, int32_t>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> heap;
    auto offer = [&](int32_t left, int32_t right) {
      const int32_t value = child(left, right);
      if (value >= 0) heap.emplace(value, std::min(left, right), std::max(left, right));
    };
    for (int32_t a = 0; a < static_cast<int32_t>(active.size()); ++a) {
      for (int32_t b = a + 1; b < static_cast<int32_t>(active.size()); ++b) {
        offer(active[a], active[b]);
      }
    }
    while (!heap.empty()) {
      const auto [child_token, left, right] = heap.top();
      heap.pop();
      auto left_it = std::find(active.begin(), active.end(), left);
      auto right_it = std::find(active.begin(), active.end(), right);
      if (left_it == active.end() || right_it == active.end()) continue;
      *left_it = child_token;
      active.erase(right_it);
      for (const int32_t other : active) {
        if (other != child_token) offer(child_token, other);
      }
    }
    std::sort(active.begin(), active.end(), [&](int32_t left, int32_t right) {
      if (anchor[left] != anchor[right]) return anchor[left] < anchor[right];
      return left < right;
    });
    return active;
  }
};

}  // namespace

py::tuple spatial_encode_instances(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& codes,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& row_anchors,
    const std::vector<int32_t>& bins,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& merges,
    int threads) {
  if (codes.ndim() != 2 || row_anchors.ndim() != 1 ||
      row_anchors.shape(0) != codes.shape(0) ||
      codes.shape(1) != static_cast<int64_t>(bins.size()) || merges.ndim() != 2 ||
      merges.shape(1) != 2) {
    throw std::invalid_argument("invalid spatial replay inputs");
  }
  Replay replay;
  replay.dimension = static_cast<int32_t>(bins.size());
  replay.offsets.assign(replay.dimension + 1, 0);
  for (int32_t axis = 0; axis < replay.dimension; ++axis) {
    replay.offsets[axis + 1] = replay.offsets[axis] + bins[axis];
  }
  const int32_t atoms = replay.offsets.back();
  replay.vocab_size = atoms + static_cast<int32_t>(merges.shape(0));
  replay.rule.assign(static_cast<size_t>(replay.vocab_size) * replay.vocab_size, -1);
  replay.anchor.assign(replay.vocab_size, -1);
  for (int32_t axis = 0; axis < replay.dimension; ++axis) {
    for (int32_t token = replay.offsets[axis]; token < replay.offsets[axis + 1]; ++token) {
      replay.anchor[token] = axis;
    }
  }
  auto rules = merges.unchecked<2>();
  for (int32_t index = 0; index < merges.shape(0); ++index) {
    const int32_t left = rules(index, 0), right = rules(index, 1), child = atoms + index;
    if (left < 0 || right < 0 || left >= child || right >= child) {
      throw std::invalid_argument("invalid spatial merge rule");
    }
    replay.rule[static_cast<size_t>(std::min(left, right)) * replay.vocab_size +
                std::max(left, right)] = child;
    replay.anchor[child] = std::min(replay.anchor[left], replay.anchor[right]);
  }

  const int64_t rows = codes.shape(0);
  std::vector<int32_t> counts(rows);
  const int32_t* source = codes.data();
  for (int64_t row = 0; row < rows; ++row) {
    int32_t active = 0;
    for (int32_t axis = 0; axis < replay.dimension; ++axis) {
      const int32_t value = source[row * replay.dimension + axis];
      if (value < -1 || value >= bins[axis]) {
        throw std::invalid_argument("a cell index is outside its coordinate's bin count");
      }
      active += value >= 0;
    }
    if (active == 0) throw std::invalid_argument("every row needs an active coordinate");
  }
#ifndef _OPENMP
  (void)threads;
#endif
  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : omp_get_max_threads())
#endif
    for (int64_t row = 0; row < rows; ++row) {
      counts[row] = static_cast<int32_t>(
          replay.encode(source + row * replay.dimension).size());
    }
  }
  std::vector<int64_t> offsets(rows + 1, 0);
  for (int64_t row = 0; row < rows; ++row) offsets[row + 1] = offsets[row] + counts[row];
  py::array_t<int32_t> tokens(offsets.back());
  py::array_t<int64_t> anchors(offsets.back());
  int32_t* output_tokens = tokens.mutable_data();
  int64_t* output_anchors = anchors.mutable_data();
  const int64_t* input_anchors = row_anchors.data();
  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : omp_get_max_threads())
#endif
    for (int64_t row = 0; row < rows; ++row) {
      const auto encoded = replay.encode(source + row * replay.dimension);
      const int64_t begin = offsets[row];
      for (int32_t index = 0; index < static_cast<int32_t>(encoded.size()); ++index) {
        output_tokens[begin + index] = encoded[index];
        output_anchors[begin + index] = input_anchors[row];
      }
    }
  }
  return py::make_tuple(tokens, anchors);
}
