#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "parallel.hpp"

namespace py = pybind11;

namespace {

int lowest_bit(uint64_t mask) { return mask ? __builtin_ctzll(mask) : -1; }

uint64_t pair_key(int64_t left, int64_t right) {
  return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
}

struct Rule {
  int64_t child;
  int64_t rank;
};

class SetBPE {
 public:
  SetBPE(std::vector<int64_t> bins, int64_t lookahead,
         const std::vector<std::pair<int64_t, int64_t>>& spatial_merges,
         const std::vector<std::tuple<int64_t, int64_t, int64_t>>& temporal_merges)
      : bins_(std::move(bins)), lookahead_(lookahead) {
    const int64_t dimension = static_cast<int64_t>(bins_.size());
    if (dimension < 1 || dimension > 63) throw std::invalid_argument("bins must have 1..63 slots");
    if (lookahead_ < 1) throw std::invalid_argument("lookahead must be positive");
    offsets_.assign(dimension + 1, 0);
    for (int64_t d = 0; d < dimension; ++d) {
      if (bins_[d] < 1) throw std::invalid_argument("every dimension needs at least one bin");
      offsets_[d + 1] = offsets_[d] + bins_[d];
    }
    atom_count_ = offsets_[dimension];
    spatial_size_ = atom_count_ + static_cast<int64_t>(spatial_merges.size());
    support_.assign(spatial_size_, 0);
    expansion_.assign(spatial_size_ * dimension, -1);
    for (int64_t d = 0; d < dimension; ++d) {
      for (int64_t v = 0; v < bins_[d]; ++v) {
        const int64_t token = offsets_[d] + v;
        support_[token] = uint64_t{1} << d;
        expansion_[token * dimension + d] = v;
      }
    }
    for (size_t rank = 0; rank < spatial_merges.size(); ++rank) {
      const auto [left, right] = spatial_merges[rank];
      const int64_t token = atom_count_ + static_cast<int64_t>(rank);
      if (left < 0 || right < 0 || left >= token || right >= token)
        throw std::invalid_argument("a merge rule must reference already-created tokens");
      if (support_[left] & support_[right])
        throw std::invalid_argument("a merge rule joins two overlapping supports");
      support_[token] = support_[left] | support_[right];
      for (int64_t d = 0; d < dimension; ++d) {
        expansion_[token * dimension + d] =
            std::max(expansion_[left * dimension + d], expansion_[right * dimension + d]);
      }
      spatial_rules_[pair_key(std::min(left, right), std::max(left, right))] =
          Rule{token, static_cast<int64_t>(rank)};
    }
    spatial_anchor_.resize(spatial_size_);
    for (int64_t token = 0; token < spatial_size_; ++token)
      spatial_anchor_[token] = lowest_bit(support_[token]);


    vocab_size_ = spatial_size_ + static_cast<int64_t>(temporal_merges.size());
    patterns_.resize(vocab_size_);
    root_.resize(vocab_size_);
    for (int64_t token = 0; token < spatial_size_; ++token) {
      patterns_[token] = {{0, token}};
      root_[token] = token;
    }
    for (size_t rank = 0; rank < temporal_merges.size(); ++rank) {
      const auto [left, right, offset] = temporal_merges[rank];
      const int64_t child = spatial_size_ + static_cast<int64_t>(rank);
      if (left < 0 || right < 0 || left >= child || right >= child)
        throw std::invalid_argument("a temporal merge must reference already-created tokens");
      if (offset < 1 || offset >= lookahead_)
        throw std::invalid_argument("a temporal merge offset must be in [1, lookahead)");
      std::vector<std::pair<int64_t, int64_t>> pattern = patterns_[left];
      for (const auto& [time, token] : patterns_[right]) pattern.emplace_back(time + offset, token);
      std::sort(pattern.begin(), pattern.end());
      const int64_t span = 1 + pattern.back().first;
      if (span > lookahead_) throw std::invalid_argument("temporal merge exceeds the lookahead horizon");
      std::vector<uint64_t> masks(span, 0);
      int roots = 0;
      for (const auto& [time, token] : pattern) {
        if (masks[time] & support_[token])
          throw std::invalid_argument("temporal merge joins overlapping spatial supports");
        masks[time] |= support_[token];
        roots += time == 0;
      }
      if (roots != 1)
        throw std::invalid_argument("a rooted temporal token must have exactly one time-zero spatial token");
      patterns_[child] = std::move(pattern);
      root_[child] = root_[left];
      const auto key = std::make_tuple(left, right, offset);
      if (temporal_rules_.count(key)) throw std::invalid_argument("duplicate temporal merge rule");
      temporal_rules_[key] = Rule{child, static_cast<int64_t>(rank)};
    }
  }

  int64_t vocab_size() const { return vocab_size_; }
  int64_t dimension() const { return static_cast<int64_t>(bins_.size()); }


  std::vector<int64_t> encode(const int64_t* cells, int64_t steps) const {
    const int64_t dimension = static_cast<int64_t>(bins_.size());
    if (steps < 1) throw std::invalid_argument("every episode must have at least one step");
    uint64_t first_support = 0;
    for (int64_t t = 0; t < steps; ++t) {
      uint64_t support = 0;
      for (int64_t d = 0; d < dimension; ++d) {
        const int64_t value = cells[t * dimension + d];
        if (value < -1 || value >= bins_[d])
          throw std::invalid_argument("episodes must contain -1 or an in-range cell index");
        if (value >= 0) support |= uint64_t{1} << d;
      }
      if (!support) throw std::invalid_argument("every timestep needs at least one active coordinate");
      if (t == 0) first_support = support;
      else if (support != first_support)
        throw std::invalid_argument("active dimensions must remain constant within an episode");
    }


    std::vector<int64_t> tokens, anchors;
    std::vector<char> active;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> at(steps);
    for (int64_t t = 0; t < steps; ++t) {
      for (int64_t token : spatial_encode(cells + t * dimension)) {
        at[t].emplace_back(token, static_cast<int64_t>(tokens.size()));
        tokens.push_back(token);
        anchors.push_back(t);
        active.push_back(1);
      }
    }


    using Offer = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    std::priority_queue<Offer, std::vector<Offer>, std::greater<Offer>> frontier;
    auto offer = [&](int64_t left_instance, int64_t right_instance) {
      const auto found = temporal_rules_.find(std::make_tuple(
          tokens[left_instance], tokens[right_instance],
          anchors[right_instance] - anchors[left_instance]));
      if (found == temporal_rules_.end()) return;
      const Rule& rule = found->second;
      frontier.emplace(rule.rank, anchors[left_instance],
                       spatial_anchor_[root_[rule.child]], left_instance, right_instance,
                       rule.child);
    };
    for (int64_t anchor = 0; anchor < steps; ++anchor) {
      for (int64_t offset = 1; offset < std::min<int64_t>(lookahead_, steps - anchor); ++offset) {
        for (const auto& [left_token, left_instance] : at[anchor]) {
          for (const auto& [right_token, right_instance] : at[anchor + offset]) {
            offer(left_instance, right_instance);
          }
        }
      }
    }
    auto erase = [&](int64_t step, int64_t token) {
      auto& items = at[step];
      for (auto it = items.begin(); it != items.end(); ++it) {
        if (it->first == token) { items.erase(it); return; }
      }
    };
    while (!frontier.empty()) {
      const auto [rank, anchor, root_anchor, left_instance, right_instance, child] = frontier.top();
      frontier.pop();
      (void)rank; (void)root_anchor;
      if (!active[left_instance] || !active[right_instance]) continue;
      if (anchors[left_instance] != anchor) continue;
      const auto resolved = temporal_rules_.find(std::make_tuple(
          tokens[left_instance], tokens[right_instance], anchors[right_instance] - anchor));
      if (resolved == temporal_rules_.end() || resolved->second.child != child) continue;
      active[left_instance] = 0;
      active[right_instance] = 0;
      erase(anchor, tokens[left_instance]);
      erase(anchors[right_instance], tokens[right_instance]);
      const int64_t new_instance = static_cast<int64_t>(tokens.size());
      tokens.push_back(child);
      anchors.push_back(anchor);
      active.push_back(1);
      at[anchor].emplace_back(child, new_instance);
      for (int64_t offset = 1; offset < lookahead_; ++offset) {
        const int64_t future = anchor + offset;
        if (future < steps) {
          for (const auto& [token, other] : at[future]) offer(new_instance, other);
        }
        const int64_t past = anchor - offset;
        if (past >= 0) {
          for (const auto& [token, other] : at[past]) offer(other, new_instance);
        }
      }
    }

    std::vector<std::tuple<int64_t, int64_t, int64_t>> ordered;
    for (int64_t anchor = 0; anchor < steps; ++anchor) {
      for (const auto& [token, instance] : at[anchor]) {
        if (active[instance]) ordered.emplace_back(anchor, spatial_anchor_[root_[token]], token);
      }
    }
    std::sort(ordered.begin(), ordered.end());
    std::vector<int64_t> stream;
    stream.reserve(ordered.size());
    for (const auto& item : ordered) stream.push_back(std::get<2>(item));
    return stream;
  }


  std::vector<int64_t> decode(const int64_t* stream, int64_t length, uint64_t full) const {
    const int64_t dimension = static_cast<int64_t>(bins_.size());
    std::vector<uint64_t> coverage{0};
    std::vector<std::vector<int64_t>> spatial_tokens(1);
    size_t front = 0;
    for (int64_t index = 0; index < length; ++index) {
      const int64_t token = stream[index];
      if (token < 0 || token >= vocab_size_)
        throw std::invalid_argument("temporal token " + std::to_string(token) + " is outside the vocabulary");
      while (front < coverage.size() && coverage[front] == full) ++front;
      if (front == coverage.size()) {
        coverage.push_back(0);
        spatial_tokens.emplace_back();
      }
      const uint64_t uncovered = full & ~coverage[front];
      if (spatial_anchor_[root_[token]] != lowest_bit(uncovered))
        throw std::invalid_argument("temporal stream violates root order at the earliest incomplete step");
      for (const auto& [offset, spatial_token] : patterns_[token]) {
        const size_t step = front + static_cast<size_t>(offset);
        while (step >= coverage.size()) {
          coverage.push_back(0);
          spatial_tokens.emplace_back();
        }
        if (coverage[step] & support_[spatial_token])
          throw std::invalid_argument("temporal stream contains overlapping spatial support");
        coverage[step] |= support_[spatial_token];
        spatial_tokens[step].push_back(spatial_token);
      }
    }
    for (uint64_t mask : coverage) {
      if (mask != full) throw std::invalid_argument("temporal stream ended with an incomplete control step");
    }
    std::vector<int64_t> cells(coverage.size() * dimension, -1);
    for (size_t step = 0; step < coverage.size(); ++step) {
      for (int64_t token : spatial_tokens[step]) {
        for (int64_t d = 0; d < dimension; ++d) {
          const int64_t value = expansion_[token * dimension + d];
          if (value >= 0) cells[step * dimension + d] = value;
        }
      }
    }
    return cells;
  }

 private:

  std::vector<int64_t> spatial_encode(const int64_t* row) const {
    const int64_t dimension = static_cast<int64_t>(bins_.size());
    std::vector<int64_t> present;
    for (int64_t d = 0; d < dimension; ++d) {
      if (row[d] >= 0) present.push_back(offsets_[d] + row[d]);
    }
    std::sort(present.begin(), present.end());
    using Candidate = std::tuple<int64_t, int64_t, int64_t, int64_t>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> heap;
    for (size_t i = 0; i < present.size(); ++i) {
      for (size_t j = i + 1; j < present.size(); ++j) {
        const auto found = spatial_rules_.find(pair_key(present[i], present[j]));
        if (found != spatial_rules_.end())
          heap.emplace(found->second.rank, found->second.child, present[i], present[j]);
      }
    }
    std::vector<int64_t> tokens = present;
    auto contains = [&](int64_t token) {
      return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
    };
    while (!heap.empty()) {
      const auto [rank, child, left, right] = heap.top();
      heap.pop();
      (void)rank;
      if (!contains(left) || !contains(right)) continue;
      tokens.erase(std::find(tokens.begin(), tokens.end(), left));
      tokens.erase(std::find(tokens.begin(), tokens.end(), right));
      for (int64_t other : tokens) {
        const int64_t low = std::min(child, other), high = std::max(child, other);
        const auto found = spatial_rules_.find(pair_key(low, high));
        if (found != spatial_rules_.end())
          heap.emplace(found->second.rank, found->second.child, low, high);
      }
      tokens.push_back(child);
    }
    std::sort(tokens.begin(), tokens.end(), [&](int64_t a, int64_t b) {
      return std::make_pair(spatial_anchor_[a], a) < std::make_pair(spatial_anchor_[b], b);
    });
    return tokens;
  }

  struct TupleHash {
    size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& key) const {
      const auto [a, b, c] = key;
      return std::hash<uint64_t>()((static_cast<uint64_t>(a) * 1000003u) ^
                                   (static_cast<uint64_t>(b) << 20) ^ static_cast<uint64_t>(c));
    }
  };

  std::vector<int64_t> bins_;
  int64_t lookahead_;
  std::vector<int64_t> offsets_;
  int64_t atom_count_ = 0, spatial_size_ = 0, vocab_size_ = 0;
  std::vector<uint64_t> support_;
  std::vector<int64_t> expansion_;
  std::vector<int> spatial_anchor_;
  std::unordered_map<uint64_t, Rule> spatial_rules_;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> patterns_;
  std::vector<int64_t> root_;
  std::unordered_map<std::tuple<int64_t, int64_t, int64_t>, Rule, TupleHash> temporal_rules_;
};

py::list encode_batch(const SetBPE& codec,
                      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& cells,
                      int threads) {
  if (cells.ndim() != 3 || cells.shape(1) < 1 || cells.shape(2) != codec.dimension())
    throw std::invalid_argument("cells must have shape [B, T, D] with the codec's dimension");
  const int64_t batch = cells.shape(0), steps = cells.shape(1), width = cells.shape(2);
  const int64_t* data = cells.data();
  std::vector<std::vector<int64_t>> streams(batch);
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t index) {
      streams[index] = codec.encode(data + index * steps * width, steps);
    });
  }
  py::list output(batch);
  for (int64_t index = 0; index < batch; ++index) output[index] = py::cast(streams[index]);
  return output;
}

py::array_t<int64_t> decode_batch(const SetBPE& codec, const std::vector<std::vector<int64_t>>& streams,
                                  int64_t width, uint64_t full, int threads) {
  const int64_t batch = static_cast<int64_t>(streams.size());
  if (batch < 1) throw std::invalid_argument("streams must not be empty");
  if (width != codec.dimension())
    throw std::invalid_argument("width must equal the codec's dimension");
  const uint64_t all = (uint64_t{1} << codec.dimension()) - 1;
  if (!full || (full & ~all))
    throw std::invalid_argument("full_support must select existing dimensions");
  std::vector<std::vector<int64_t>> cells(batch);
  {
    py::gil_scoped_release release;
    ac2::parallel_for(batch, threads, [&](int64_t index) {
      if (streams[index].empty()) throw std::invalid_argument("token stream must not be empty");
      cells[index] = codec.decode(streams[index].data(), static_cast<int64_t>(streams[index].size()), full);
    });
  }
  const int64_t steps = static_cast<int64_t>(cells[0].size()) / width;
  for (const auto& row : cells) {
    if (static_cast<int64_t>(row.size()) != steps * width)
      throw std::invalid_argument("all token streams in a batch must decode to the same length");
  }
  py::array_t<int64_t> output({batch, steps, width});
  int64_t* out = output.mutable_data();
  for (int64_t index = 0; index < batch; ++index)
    std::copy(cells[index].begin(), cells[index].end(), out + index * steps * width);
  return output;
}

}

void register_setbpe(py::module_& module) {
  py::class_<SetBPE>(module, "SetBPE")
      .def(py::init<std::vector<int64_t>, int64_t, const std::vector<std::pair<int64_t, int64_t>>&,
                    const std::vector<std::tuple<int64_t, int64_t, int64_t>>&>(),
           py::arg("bins"), py::arg("lookahead"), py::arg("spatial_merges"),
           py::arg("temporal_merges"))
      .def_property_readonly("vocab_size", &SetBPE::vocab_size)
      .def("encode_batch", &encode_batch, py::arg("cells"), py::arg("threads") = 0,
           "Dense [B, T, D] cells (-1 = absent) -> one token stream per row.")
      .def("decode_batch", &decode_batch, py::arg("streams"), py::arg("width"),
           py::arg("full_support"), py::arg("threads") = 0,
           "Equal-horizon streams -> dense [B, T, D] cells, -1 outside full_support.")
      .def("decode_checked_batch", [](const SetBPE& codec, const py::iterable& streams,
                                     int64_t width, uint64_t full, int threads) {
        const py::object numpy_integer = py::module_::import("numpy").attr("integer");
        py::object integral;
        std::vector<std::vector<int64_t>> rows;
        if (PyList_Check(streams.ptr())) rows.reserve(PyList_GET_SIZE(streams.ptr()));
        for (py::handle stream : streams) {
          std::vector<int64_t> row;
          if (PyList_Check(stream.ptr())) row.reserve(PyList_GET_SIZE(stream.ptr()));
          for (py::handle raw : py::reinterpret_borrow<py::iterable>(stream)) {
            bool integer = PyLong_Check(raw.ptr()) || py::isinstance(raw, numpy_integer);
            if (!integer) {
              if (!integral) integral = py::module_::import("numbers").attr("Integral");
              integer = py::isinstance(raw, integral);
            }
            if (PyBool_Check(raw.ptr()) || !integer)
              throw std::invalid_argument("tokens must be integers");
            py::object number = PyLong_Check(raw.ptr())
                ? py::reinterpret_borrow<py::object>(raw)
                : py::reinterpret_steal<py::object>(PyNumber_Long(raw.ptr()));
            if (!number) throw py::error_already_set();
            int overflow = 0;
            const int64_t token = PyLong_AsLongLongAndOverflow(number.ptr(), &overflow);
            if (PyErr_Occurred()) throw py::error_already_set();
            if (overflow) throw std::invalid_argument("token is outside the vocabulary");
            row.push_back(token);
          }
          rows.push_back(std::move(row));
        }
        return decode_batch(codec, rows, width, full, threads);
      }, py::arg("streams"), py::arg("width"), py::arg("full_support"), py::arg("threads") = 0);
}
