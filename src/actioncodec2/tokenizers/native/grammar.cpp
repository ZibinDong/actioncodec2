#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using State = std::vector<int64_t>;

class CoverageGrammar {
 protected:
  int64_t horizon_, vocabulary_;
  virtual void validate_state(const State&) const = 0;
  virtual bool allowed(const State&, int64_t) const = 0;
  virtual State apply(const State&, int64_t) const = 0;
 public:
  virtual ~CoverageGrammar() = default;
  State advance(const State& state, int64_t token) const {
    validate_state(state);
    if (token < 0 || token >= vocabulary_ || !allowed(state, token)) return {};
    return apply(state, token);
  }
  py::array_t<bool> mask(const State& state) const {
    validate_state(state);
    py::array_t<bool> result(vocabulary_);
    auto* output = result.mutable_data();
    py::gil_scoped_release release;
    for (int64_t token = 0; token < vocabulary_; ++token) output[token] = allowed(state, token);
    return result;
  }
};

class SetGrammar : public CoverageGrammar {
  uint64_t full_;
  std::vector<int64_t> roots_;
  std::vector<std::vector<std::pair<int64_t, uint64_t>>> patterns_;
  std::vector<bool> scope_;

  void validate_state(const State& state) const override {
    if (static_cast<int64_t>(state.size()) != horizon_)
      throw std::invalid_argument("invalid grammar state length");
    for (int64_t value : state) {
      if (value < 0 || (static_cast<uint64_t>(value) & ~full_))
        throw std::invalid_argument("invalid grammar state support");
    }
  }
  int64_t front(const State& state) const {
    for (int64_t t = 0; t < horizon_; ++t)
      if (static_cast<uint64_t>(state[t]) != full_) return t;
    return horizon_;
  }
  bool allowed(const State& state, int64_t token) const override {
    if (!scope_[token]) return false;
    const int64_t start = front(state);
    if (start == horizon_) return false;
    uint64_t missing = full_ & ~static_cast<uint64_t>(state[start]);
    int64_t root = 0;
    while (!(missing & 1)) { missing >>= 1; ++root; }
    if (roots_[token] != root) return false;
    for (auto [offset, support] : patterns_[token]) {
      if (offset >= horizon_ - start || (static_cast<uint64_t>(state[start + offset]) & support))
        return false;
    }
    return true;
  }
  State apply(const State& state, int64_t token) const override {
    State result = state;
    const int64_t start = front(state);
    for (auto [offset, support] : patterns_[token])
      result[start + offset] |= static_cast<int64_t>(support);
    return result;
  }
 public:
  SetGrammar(int64_t horizon, uint64_t full, std::vector<int64_t> roots,
             std::vector<std::vector<std::pair<int64_t, uint64_t>>> patterns)
      : full_(full), roots_(std::move(roots)), patterns_(std::move(patterns)) {
    if (horizon < 1 || !full || full > static_cast<uint64_t>(INT64_MAX) ||
        roots_.empty() || roots_.size() != patterns_.size())
      throw std::invalid_argument("invalid Set-BPE grammar metadata");
    horizon_ = horizon;
    vocabulary_ = roots_.size();
    for (int64_t token = 0; token < vocabulary_; ++token) {
      if (roots_[token] < 0 || roots_[token] >= 63 || patterns_[token].empty())
        throw std::invalid_argument("invalid grammar root or empty pattern");
      std::map<int64_t, uint64_t> coverage;
      bool scope = true;
      for (auto [offset, support] : patterns_[token]) {
        if (offset < 0 || !support || support > static_cast<uint64_t>(INT64_MAX) ||
            (coverage[offset] & support))
          throw std::invalid_argument("invalid grammar pattern offset or support");
        coverage[offset] |= support;
        scope = scope && !(support & ~full_);
      }
      const uint64_t anchor = coverage[0];
      if (!(anchor & (uint64_t{1} << roots_[token])) ||
          (anchor & ((uint64_t{1} << roots_[token]) - 1)))
        throw std::invalid_argument("grammar root must anchor the pattern");
      scope_.push_back(scope);
    }
  }
};

class PlainGrammar : public CoverageGrammar {
  int64_t width_, cells_;
  std::vector<int64_t> lengths_;
  std::vector<bool> phase_;

  void validate_state(const State& state) const override {
    if (state.size() != 1 || state[0] < 0 || state[0] > cells_)
      throw std::invalid_argument("invalid plain-BPE grammar state");
  }
  bool allowed(const State& state, int64_t token) const override {
    return lengths_[token] <= cells_ - state[0] &&
           phase_[(state[0] % width_) * vocabulary_ + token];
  }
  State apply(const State& state, int64_t token) const override {
    return {state[0] + lengths_[token]};
  }
 public:
  PlainGrammar(int64_t horizon, std::vector<int64_t> lengths,
               const py::array_t<bool, py::array::c_style | py::array::forcecast>& phase)
      : lengths_(std::move(lengths)) {
    if (horizon < 1 || lengths_.empty() || phase.ndim() != 2 || phase.shape(0) < 1 ||
        phase.shape(1) != static_cast<int64_t>(lengths_.size()) ||
        horizon > std::numeric_limits<int64_t>::max() / phase.shape(0))
      throw std::invalid_argument("invalid plain-BPE grammar metadata");
    for (int64_t length : lengths_) {
      if (length < 1) throw std::invalid_argument("token expansion must be positive");
    }
    horizon_ = horizon;
    width_ = phase.shape(0);
    cells_ = horizon * width_;
    vocabulary_ = lengths_.size();
    phase_.assign(phase.data(), phase.data() + phase.size());
  }
};
}

void register_grammar(py::module_& module) {
  py::class_<CoverageGrammar, std::shared_ptr<CoverageGrammar>>(module, "CoverageGrammar")
      .def("advance", [](const CoverageGrammar& grammar, const State& state, int64_t token) -> py::object {
        State next = grammar.advance(state, token);
        if (next.empty()) return py::none();
        py::tuple result(next.size());
        for (size_t i = 0; i < next.size(); ++i) result[i] = next[i];
        return result;
      })
      .def("mask", &CoverageGrammar::mask);
  py::class_<SetGrammar, CoverageGrammar, std::shared_ptr<SetGrammar>>(module, "SetGrammar")
      .def(py::init<int64_t, uint64_t, std::vector<int64_t>,
                    std::vector<std::vector<std::pair<int64_t, uint64_t>>>>());
  py::class_<PlainGrammar, CoverageGrammar, std::shared_ptr<PlainGrammar>>(module, "PlainGrammar")
      .def(py::init<int64_t, std::vector<int64_t>,
                    const py::array_t<bool, py::array::c_style | py::array::forcecast>&>());
}
