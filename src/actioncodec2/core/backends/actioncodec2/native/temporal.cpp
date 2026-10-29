// Native exact rooted temporal Set-BPE trainer.
//
// A temporal symbol is rooted at one token of the current control step and may
// absorb spatial tokens from a bounded number of future steps.  Training is the
// exact sparse local-graph algorithm: counters exist only for rule keys that
// actually occur inside the lookahead window, each merge updates only the edges
// incident to the instances it changed, and shifted self-rules (left == right)
// are certified by exact left-to-right maximum matching before they may win.
//
// This mirrors actioncodec2/setbpe/temporal.py merge for merge, including the
// tie-break, so the reference trainer can prove the two agree.  What differs is
// the constant factor: the Python trainer allocates a dense
// ``lookahead x V x V`` legality table, while this one derives legality from the
// per-token time-shifted support masks in O(span) with no allocation, which is
// what makes large vocabularies tractable.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

struct TemporalEntry {
  int64_t count;
  uint64_t key;
  uint32_t version;
  uint8_t certified;
};

struct TemporalOrder {
  bool operator()(const TemporalEntry& a, const TemporalEntry& b) const {
    if (a.count != b.count) return a.count < b.count;        // larger count first
    if (a.key != b.key) return a.key > b.key;                // smaller key first
    return a.certified > b.certified;                        // uncertified first
  }
};

struct InstanceChunk {
  const int32_t* tokens;
  const int64_t* anchors;
  int64_t size;
};

size_t initial_counter_capacity(int64_t instance_count, int32_t base_vocab, int32_t lookahead) {
  const size_t by_instances = static_cast<size_t>(instance_count) * 4 + 64;
  const size_t vocab = static_cast<size_t>(base_vocab);
  const size_t offsets = static_cast<size_t>(std::max(lookahead - 1, 0));
  if (vocab != 0 && offsets > (std::numeric_limits<size_t>::max() - 64) / vocab / vocab) {
    return by_instances;
  }
  const size_t by_key_universe = vocab * vocab * offsets + 64;
  return std::min(by_instances, by_key_universe);
}

inline uint64_t rule_key(int32_t left, int32_t right, int32_t offset) {
  return (static_cast<uint64_t>(left) << 40) | (static_cast<uint64_t>(right) << 16) |
         static_cast<uint64_t>(offset);
}

inline void unpack(uint64_t key, int32_t& left, int32_t& right, int32_t& offset) {
  left = static_cast<int32_t>(key >> 40);
  right = static_cast<int32_t>((key >> 16) & 0xffffffull);
  offset = static_cast<int32_t>(key & 0xffffull);
}

class Trainer {
 public:
  Trainer(int64_t total_anchors, int64_t instance_capacity_hint,
          const uint64_t* spatial_support, const int32_t* spatial_anchor, int32_t base_vocab,
          int32_t dimension, int32_t lookahead, int64_t vocab_size, int threads)
      : base_vocab_(base_vocab),
        dimension_(dimension),
        lookahead_(lookahead),
        total_anchors_(total_anchors),
        counts_(initial_counter_capacity(instance_capacity_hint, base_vocab, lookahead)) {
#ifdef _OPENMP
    // More workers eventually make the per-merge local-map reduction slower
    // than the incident scan itself.  Keep auto mode at the measured scaling
    // knee; an explicit value remains an opt-in override.
    threads_ = threads > 0 ? threads : std::min(omp_get_max_threads(), 32);
#else
    (void)threads;
    threads_ = 1;
#endif
    const size_t capacity = static_cast<size_t>(vocab_size) + 16;
    masks_.assign(capacity * lookahead_, 0);
    spans_.assign(capacity, 1);
    root_anchor_.assign(capacity, 0);
    pattern_start_.assign(capacity + 1, 0);
    for (int32_t token = 0; token < base_vocab_; ++token) {
      masks_[static_cast<size_t>(token) * lookahead_] = spatial_support[token];
      root_anchor_[token] = spatial_anchor[token];
      pattern_start_[token] = static_cast<int32_t>(pattern_.size());
      pattern_.push_back({0, token});
    }
    pattern_start_[base_vocab_] = static_cast<int32_t>(pattern_.size());
    token_count_ = base_vocab_;

    member_.assign(static_cast<size_t>(total_anchors) * dimension_, -1);
    occurrence_.resize(capacity);
    dirty_.assign(capacity, 0);
  }

  void add_chunk(const InstanceChunk& chunk) {
    for (int64_t index = 0; index < chunk.size; ++index) {
      add_instance(chunk.tokens[index], chunk.anchors[index]);
    }
    initial_count_ += chunk.size;
  }

  void finalize_instances() {
    if (initial_count_ <= 0) throw std::runtime_error("temporal training needs instances");
    active_count_ = initial_count_;
    const auto census_started = std::chrono::steady_clock::now();
    census();
    census_seconds_ = elapsed(census_started);
    counts_.for_each([&](uint64_t key, int64_t value) { heap_.push_back({value, key, 0, 0}); });
    std::make_heap(heap_.begin(), heap_.end(), TemporalOrder{});
    initial_edge_types_ = heap_.size();
  }

  // -- geometry ------------------------------------------------------------

  bool legal(int32_t left, int32_t right, int32_t offset) const {
    const int32_t right_span = spans_[right];
    if (offset + right_span > lookahead_) return false;
    const uint64_t* left_masks = &masks_[static_cast<size_t>(left) * lookahead_];
    const uint64_t* right_masks = &masks_[static_cast<size_t>(right) * lookahead_];
    const int32_t overlap = std::min(spans_[left] - offset, right_span);
    for (int32_t step = 0; step < overlap; ++step) {
      if (left_masks[offset + step] & right_masks[step]) return false;
    }
    return true;
  }

  // -- instance bookkeeping ------------------------------------------------

  int32_t add_instance(int32_t token, int64_t anchor) {
    if (token < 0 || static_cast<size_t>(token) >= occurrence_.size()) {
      throw std::runtime_error("temporal instance token " + std::to_string(token) +
                               " is outside trainer capacity " +
                               std::to_string(occurrence_.size()));
    }
    if (anchor < 0 || anchor >= total_anchors_) {
      throw std::runtime_error("temporal instance anchor is outside the corpus timeline");
    }
    int32_t* row = &member_[static_cast<size_t>(anchor) * dimension_];
    int32_t slot = 0;
    while (slot < dimension_ && row[slot] >= 0) ++slot;
    if (slot == dimension_) {
      throw std::runtime_error("temporal anchor exceeded the token-width bound");
    }
    row[slot] = token;
    occurrence_[token].push_back(anchor);
    dirty_[token] = 1;
    return slot;
  }

  // Removal touches only the row mirror.  The occurrence list is allowed to
  // keep stale anchors because ``occurs_at`` is an exact membership test, so
  // compaction can be deferred to the next time the list is actually read.
  void remove_instance(int32_t token, int64_t anchor) {
    int32_t* row = &member_[static_cast<size_t>(anchor) * dimension_];
    for (int32_t slot = 0; slot < dimension_; ++slot) {
      if (row[slot] == token) {
        row[slot] = -1;
        break;
      }
    }
    dirty_[token] = 1;
  }

  const std::vector<int64_t>& sorted_occurrence(int32_t token) {
    auto& anchors = occurrence_[token];
    if (dirty_[token]) {
      std::sort(anchors.begin(), anchors.end());
      anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
      anchors.erase(std::remove_if(anchors.begin(), anchors.end(),
                                   [&](int64_t anchor) { return !occurs_at(token, anchor); }),
                    anchors.end());
      dirty_[token] = 0;
    }
    return anchors;
  }

  bool occurs_at(int32_t token, int64_t anchor) const {
    const int32_t* row = &member_[static_cast<size_t>(anchor) * dimension_];
    for (int32_t slot = 0; slot < dimension_; ++slot) {
      if (row[slot] == token) return true;
    }
    return false;
  }

  // -- matching ------------------------------------------------------------

  std::vector<std::pair<int64_t, int64_t>> select(int32_t left, int32_t right, int32_t offset) {
    std::vector<std::pair<int64_t, int64_t>> matches;
    const std::vector<int64_t>& anchors = sorted_occurrence(left);
    if (left != right) {
#ifdef _OPENMP
      if (threads_ > 1 && anchors.size() >= 65536) {
        const int workers = std::min<int64_t>(threads_, anchors.size() / 32768);
        std::vector<std::vector<std::pair<int64_t, int64_t>>> local(workers);
#pragma omp parallel num_threads(workers)
        {
          const int worker = omp_get_thread_num();
          const size_t start = anchors.size() * static_cast<size_t>(worker) / workers;
          const size_t stop = anchors.size() * static_cast<size_t>(worker + 1) / workers;
          auto& output = local[worker];
          output.reserve(stop - start);
          for (size_t index = start; index < stop; ++index) {
            const int64_t anchor = anchors[index];
            const int64_t target = anchor + offset;
            if (target < total_anchors_ && occurs_at(right, target)) {
              output.emplace_back(anchor, target);
            }
          }
        }
        size_t count = 0;
        for (const auto& output : local) count += output.size();
        matches.reserve(count);
        for (auto& output : local) {
          matches.insert(matches.end(), output.begin(), output.end());
        }
        return matches;
      }
#endif
      for (const int64_t anchor : anchors) {
        const int64_t target = anchor + offset;
        if (target < total_anchors_ && occurs_at(right, target)) {
          matches.emplace_back(anchor, target);
        }
      }
      return matches;
    }
    // Shifted self-rule: an instance cannot be both the left and the right end
    // of two different merges, so the true gain is a maximum matching, which
    // for a fixed shift on a line is exactly greedy left to right.
    std::deque<int64_t> consumed_targets;
    for (const int64_t anchor : anchors) {
      while (!consumed_targets.empty() && consumed_targets.front() < anchor) {
        consumed_targets.pop_front();
      }
      if (!consumed_targets.empty() && consumed_targets.front() == anchor) {
        consumed_targets.pop_front();
        continue;
      }
      const int64_t wide = anchor + offset;
      if (wide >= total_anchors_) continue;
      const int64_t target = wide;
      if (!occurs_at(left, target)) continue;
      matches.emplace_back(anchor, target);
      consumed_targets.push_back(target);
    }
    return matches;
  }

  // -- search --------------------------------------------------------------

  bool best(int32_t& left, int32_t& right, int32_t& offset,
            std::vector<std::pair<int64_t, int64_t>>& matches) {
    while (!heap_.empty()) {
      std::pop_heap(heap_.begin(), heap_.end(), TemporalOrder{});
      const TemporalEntry top = heap_.back();
      heap_.pop_back();
      ++heap_pops_;
      const int64_t current = counts_.get(top.key);
      if (current <= 0) continue;
      unpack(top.key, left, right, offset);
      const bool self_rule = left == right;
      const uint32_t version = self_rule ? self_version(top.key) : 0;
      const bool stale = self_rule ? version != top.version : current != top.count;
      if (stale) {
        heap_.push_back({current, top.key, version, 0});
        std::push_heap(heap_.begin(), heap_.end(), TemporalOrder{});
        continue;
      }
      if (self_rule && !top.certified) {
        // Raw edge count only upper-bounds the achievable gain here; replace
        // the priority with the exact matching size and re-queue.
        auto certified = select(left, right, offset);
        self_cache_[top.key] = {version, certified};
        heap_.push_back({static_cast<int64_t>(certified.size()), top.key, version, 1});
        std::push_heap(heap_.begin(), heap_.end(), TemporalOrder{});
        ++self_certifications_;
        continue;
      }
      if (top.certified) {
        auto cached = self_cache_.find(top.key);
        if (cached == self_cache_.end() || cached->second.first != version) continue;
        matches = cached->second.second;
      } else {
        matches = select(left, right, offset);
      }
      if (static_cast<int64_t>(matches.size()) != top.count) {
        throw std::runtime_error("temporal counter disagrees with exact matching");
      }
      return !matches.empty();
    }
    return false;
  }

  // -- commit --------------------------------------------------------------

  void commit(int32_t left, int32_t right, int32_t offset,
              const std::vector<std::pair<int64_t, int64_t>>& matches) {
    const int32_t child = token_count_++;
    // The child's pattern is the left pattern plus the right pattern shifted.
    const int32_t start = static_cast<int32_t>(pattern_.size());
    for (int32_t index = pattern_start_[left]; index < pattern_start_[left + 1]; ++index) {
      pattern_.push_back(pattern_[index]);
    }
    for (int32_t index = pattern_start_[right]; index < pattern_start_[right + 1]; ++index) {
      pattern_.push_back({pattern_[index].first + offset, pattern_[index].second});
    }
    std::sort(pattern_.begin() + start, pattern_.end());
    pattern_start_[child] = start;
    pattern_start_[child + 1] = static_cast<int32_t>(pattern_.size());

    int32_t span = 0;
    int32_t roots = 0;
    uint64_t* child_masks = &masks_[static_cast<size_t>(child) * lookahead_];
    for (int32_t index = start; index < static_cast<int32_t>(pattern_.size()); ++index) {
      const int32_t time = pattern_[index].first;
      const uint64_t support = masks_[static_cast<size_t>(pattern_[index].second) * lookahead_];
      if (time >= lookahead_) throw std::runtime_error("temporal merge exceeds the lookahead");
      if (child_masks[time] & support) {
        throw std::runtime_error("temporal merge joins overlapping spatial supports");
      }
      child_masks[time] |= support;
      span = std::max(span, time + 1);
      roots += (time == 0);
    }
    if (roots != 1) throw std::runtime_error("a rooted temporal token needs one time-zero token");
    spans_[child] = span;
    root_anchor_[child] = root_anchor_[left];

    ac2::CounterMap deltas(
        std::min(matches.size() * 8 + 64, incident_key_capacity(token_count_)));
    std::vector<std::pair<int32_t, int64_t>> removed;  // (token, anchor), sorted by anchor
    removed.reserve(matches.size() * 2);
    size_t left_index = 0;
    size_t right_index = 0;
    while (left_index < matches.size() || right_index < matches.size()) {
      const int64_t left_anchor =
          left_index < matches.size() ? matches[left_index].first : total_anchors_;
      const int64_t right_anchor =
          right_index < matches.size() ? matches[right_index].second : total_anchors_;
      if (left_anchor <= right_anchor) {
        removed.emplace_back(left, left_anchor);
        ++left_index;
      } else {
        removed.emplace_back(right, right_anchor);
        ++right_index;
      }
    }
    accumulate_incident(removed, -1, deltas);
    correct_internal(removed, +1, deltas);

    for (const auto& instance : removed) remove_instance(instance.first, instance.second);
    std::vector<std::pair<int32_t, int64_t>> children;
    children.reserve(matches.size());
    for (const auto& match : matches) {
      add_instance(child, match.first);
      children.emplace_back(child, match.first);
    }
    accumulate_incident(children, +1, deltas);
    correct_internal(children, -1, deltas);

    apply(deltas, rule_key(left, right, offset));
    active_count_ -= static_cast<int64_t>(matches.size());
    merges_.push_back({left, right, offset});
  }

  // -- accessors -----------------------------------------------------------

  const std::vector<std::array<int32_t, 3>>& merges() const { return merges_; }
  int64_t active_count() const { return active_count_; }
  int64_t initial_count() const { return initial_count_; }
  int64_t heap_pops() const { return heap_pops_; }
  int64_t counter_updates() const { return counter_updates_; }
  int64_t self_certifications() const { return self_certifications_; }
  size_t initial_edge_types() const { return initial_edge_types_; }
  size_t live_edge_types() const { return counts_.live_entries(); }
  size_t counter_buckets() const { return counts_.buckets(); }
  int threads() const { return threads_; }
  double census_seconds() const { return census_seconds_; }
  double incident_seconds() const { return incident_seconds_; }
  double incident_parallel_seconds() const { return incident_parallel_seconds_; }
  double incident_merge_seconds() const { return incident_merge_seconds_; }
  double correction_seconds() const { return correction_seconds_; }
  double correction_parallel_seconds() const { return correction_parallel_seconds_; }
  double apply_seconds() const { return apply_seconds_; }

 private:
  static double elapsed(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  }

  size_t incident_key_capacity(int32_t vocabulary) const {
    return static_cast<size_t>(6) * static_cast<size_t>(vocabulary) *
               static_cast<size_t>(std::max(lookahead_ - 1, 0)) +
           64;
  }

  void census_anchor(int64_t anchor, ac2::CounterMap& counter) const {
    const int32_t* row = &member_[static_cast<size_t>(anchor) * dimension_];
    for (int32_t slot = 0; slot < dimension_; ++slot) {
      const int32_t left = row[slot];
      if (left < 0) continue;
      for (int32_t offset = 1; offset < lookahead_; ++offset) {
        const int64_t target = anchor + offset;
        if (target >= total_anchors_) break;
        const int32_t* other_row = &member_[static_cast<size_t>(target) * dimension_];
        for (int32_t other_slot = 0; other_slot < dimension_; ++other_slot) {
          const int32_t right = other_row[other_slot];
          if (right >= 0 && legal(left, right, offset)) {
            counter.add(rule_key(left, right, offset), 1);
          }
        }
      }
    }
  }

  void census() {
    const size_t key_universe = initial_counter_capacity(total_anchors_, base_vocab_, lookahead_);
#ifdef _OPENMP
    if (threads_ > 1 && total_anchors_ >= 4096) {
      const int workers = std::min<int64_t>(threads_, total_anchors_ / 2048);
      // Small rule universes fit comfortably in one local map per worker.
      // For a large spatial vocabulary, each long-running worker would
      // eventually discover almost the whole universe and multiply memory by
      // the thread count.  Merge fixed-size anchor batches instead: this keeps
      // local memory bounded while the global counter remains exact.
      const bool batched = key_universe > (1u << 20);
      const int64_t anchors_per_worker = batched ? 8192 : total_anchors_;
      const int64_t batch_size = anchors_per_worker * workers;
      const size_t local_hint = batched
                                    ? std::min<size_t>(key_universe,
                                                       anchors_per_worker * 32 + 64)
                                    : key_universe;
      std::vector<std::unique_ptr<ac2::CounterMap>> local;
      local.reserve(workers);
      for (int worker = 0; worker < workers; ++worker) {
        local.push_back(std::make_unique<ac2::CounterMap>(local_hint));
      }
      for (int64_t batch_start = 0; batch_start < total_anchors_; batch_start += batch_size) {
        const int64_t batch_end = std::min(total_anchors_, batch_start + batch_size);
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int64_t anchor = batch_start; anchor < batch_end; ++anchor) {
          census_anchor(anchor, *local[omp_get_thread_num()]);
        }
        for (const auto& counter : local) {
          counter->for_each([&](uint64_t key, int64_t value) { counts_.add(key, value); });
          if (batched) counter->clear();
        }
      }
      return;
    }
#endif
    for (int64_t anchor = 0; anchor < total_anchors_; ++anchor) census_anchor(anchor, counts_);
  }

  uint32_t self_version(uint64_t key) const {
    auto found = self_versions_.find(key);
    return found == self_versions_.end() ? 0u : found->second;
  }

  void accumulate_incident(const std::vector<std::pair<int32_t, int64_t>>& instances,
                           int64_t multiplier, ac2::CounterMap& deltas) {
    const auto started = std::chrono::steady_clock::now();
#ifdef _OPENMP
    if (threads_ > 1 && instances.size() >= 1024) {
      const int workers = std::min<int>(threads_, static_cast<int>(instances.size() / 512));
      const size_t hint = std::min(
          instances.size() * 8 / static_cast<size_t>(workers) + 64,
          incident_key_capacity(token_count_));
      std::vector<std::unique_ptr<ac2::CounterMap>> local;
      local.reserve(workers);
      for (int worker = 0; worker < workers; ++worker) {
        local.push_back(std::make_unique<ac2::CounterMap>(hint));
      }
      const auto parallel_started = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(workers)
      for (int64_t index = 0; index < static_cast<int64_t>(instances.size()); ++index) {
        accumulate_one(instances[static_cast<size_t>(index)], multiplier,
                       *local[omp_get_thread_num()]);
      }
      incident_parallel_seconds_ += elapsed(parallel_started);
      const auto merge_started = std::chrono::steady_clock::now();
      for (const auto& counter : local) {
        counter->for_each([&](uint64_t key, int64_t value) { deltas.add(key, value); });
      }
      incident_merge_seconds_ += elapsed(merge_started);
      incident_seconds_ += elapsed(started);
      return;
    }
#endif
    for (const auto& instance : instances) {
      accumulate_one(instance, multiplier, deltas);
    }
    incident_seconds_ += elapsed(started);
  }

  void accumulate_one(const std::pair<int32_t, int64_t>& instance, int64_t multiplier,
                      ac2::CounterMap& deltas) const {
    const int32_t token = instance.first;
    const int64_t anchor = instance.second;
    for (int32_t offset = 1; offset < lookahead_; ++offset) {
      const int64_t forward = anchor + offset;
      if (forward < total_anchors_) {
        const int32_t* row = &member_[static_cast<size_t>(forward) * dimension_];
        for (int32_t slot = 0; slot < dimension_; ++slot) {
          const int32_t other = row[slot];
          if (other >= 0 && legal(token, other, offset)) {
            deltas.add(rule_key(token, other, offset), multiplier);
          }
        }
      }
      const int64_t backward = anchor - offset;
      if (backward >= 0) {
        const int32_t* row = &member_[static_cast<size_t>(backward) * dimension_];
        for (int32_t slot = 0; slot < dimension_; ++slot) {
          const int32_t other = row[slot];
          if (other >= 0 && legal(other, token, offset)) {
            deltas.add(rule_key(other, token, offset), multiplier);
          }
        }
      }
    }
  }

  // Incident aggregation sees a changed-to-changed edge once from each end.
  void correct_internal(const std::vector<std::pair<int32_t, int64_t>>& instances,
                        int64_t correction, ac2::CounterMap& deltas) {
    const auto started = std::chrono::steady_clock::now();
    if (instances.empty()) return;
    std::vector<size_t> groups{0};
    groups.reserve(instances.size());
    for (size_t index = 1; index < instances.size(); ++index) {
      if (instances[index].second < instances[index - 1].second) {
        throw std::runtime_error("changed temporal instances must be anchor-sorted");
      }
      if (instances[index].second != instances[index - 1].second) groups.push_back(index);
    }
    groups.push_back(instances.size());
    const size_t group_count = groups.size() - 1;
#ifdef _OPENMP
    if (threads_ > 1 && group_count >= 1024) {
      const int workers = std::min<int>(threads_, static_cast<int>(group_count / 512));
      const size_t hint = std::min(group_count * 2 / static_cast<size_t>(workers) + 64,
                                   incident_key_capacity(token_count_));
      std::vector<std::unique_ptr<ac2::CounterMap>> local;
      local.reserve(workers);
      for (int worker = 0; worker < workers; ++worker) {
        local.push_back(std::make_unique<ac2::CounterMap>(hint));
      }
      const auto parallel_started = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(workers)
      for (int64_t index = 0; index < static_cast<int64_t>(group_count); ++index) {
        correct_group(static_cast<size_t>(index), groups, instances, correction,
                      *local[omp_get_thread_num()]);
      }
      correction_parallel_seconds_ += elapsed(parallel_started);
      for (const auto& counter : local) {
        counter->for_each([&](uint64_t key, int64_t value) { deltas.add(key, value); });
      }
    } else
#endif
    {
      for (size_t group = 0; group < group_count; ++group) {
        correct_group(group, groups, instances, correction, deltas);
      }
    }
    correction_seconds_ += elapsed(started);
  }

  void correct_group(size_t group, const std::vector<size_t>& groups,
                     const std::vector<std::pair<int32_t, int64_t>>& instances,
                     int64_t correction, ac2::CounterMap& deltas) const {
    const size_t left_start = groups[group];
    const size_t left_stop = groups[group + 1];
    const int64_t anchor = instances[left_start].second;
    for (size_t other = group + 1; other + 1 < groups.size(); ++other) {
      const size_t right_start = groups[other];
      const int64_t distance = instances[right_start].second - anchor;
      if (distance >= lookahead_) break;
      const int32_t offset = static_cast<int32_t>(distance);
      for (size_t left_index = left_start; left_index < left_stop; ++left_index) {
        const int32_t left = instances[left_index].first;
        for (size_t right_index = right_start; right_index < groups[other + 1]; ++right_index) {
          const int32_t right = instances[right_index].first;
          if (legal(left, right, offset)) {
            deltas.add(rule_key(left, right, offset), correction);
          }
        }
      }
    }
  }

  void apply(const ac2::CounterMap& deltas, uint64_t selected) {
    const auto started = std::chrono::steady_clock::now();
    deltas.for_each([&](uint64_t key, int64_t delta) {
      const auto [previous, count] = counts_.add_with_previous(key, delta);
      if (count < 0) {
        int32_t bl = 0, br = 0, bo = 0;
        unpack(key, bl, br, bo);
        throw std::runtime_error("temporal edge counter became negative: rule=(" +
                                 std::to_string(bl) + "," + std::to_string(br) + "," +
                                 std::to_string(bo) + ") previous=" + std::to_string(previous) +
                                 " delta=" + std::to_string(delta) + " at merge " +
                                 std::to_string(merges_.size()));
      }
      counter_updates_ += delta < 0 ? -delta : delta;
      int32_t left = 0;
      int32_t right = 0;
      int32_t offset = 0;
      unpack(key, left, right, offset);
      uint32_t version = 0;
      if (left == right) {
        version = self_version(key) + 1;
        self_versions_[key] = version;
        self_cache_.erase(key);
      }
      // Every edge a commit adds contains the freshly minted child token, so a
      // pre-existing key can only lose mass; its stale heap priority stays an
      // admissible upper bound and is repaired when popped.  New keys and the
      // key just consumed need an entry right away.
      if (count > 0 && (previous == 0 || key == selected)) {
        heap_.push_back({count, key, version, 0});
        std::push_heap(heap_.begin(), heap_.end(), TemporalOrder{});
      }
    });
    apply_seconds_ += elapsed(started);
  }

  int32_t base_vocab_;
  int32_t dimension_;
  int32_t lookahead_;
  int64_t total_anchors_;
  int threads_ = 1;
  int32_t token_count_ = 0;

  std::vector<uint64_t> masks_;
  std::vector<int32_t> spans_;
  std::vector<int32_t> root_anchor_;
  std::vector<std::pair<int32_t, int32_t>> pattern_;  // (time, spatial token)
  std::vector<int32_t> pattern_start_;

  std::vector<int32_t> member_;
  std::vector<std::vector<int64_t>> occurrence_;
  std::vector<uint8_t> dirty_;

  ac2::CounterMap counts_;
  std::vector<TemporalEntry> heap_;
  std::unordered_map<uint64_t, uint32_t> self_versions_;
  std::unordered_map<uint64_t, std::pair<uint32_t, std::vector<std::pair<int64_t, int64_t>>>>
      self_cache_;

  std::vector<std::array<int32_t, 3>> merges_;
  int64_t active_count_ = 0;
  int64_t initial_count_ = 0;
  int64_t heap_pops_ = 0;
  int64_t counter_updates_ = 0;
  int64_t self_certifications_ = 0;
  size_t initial_edge_types_ = 0;
  double census_seconds_ = 0.0;
  double incident_seconds_ = 0.0;
  double incident_parallel_seconds_ = 0.0;
  double incident_merge_seconds_ = 0.0;
  double correction_seconds_ = 0.0;
  double correction_parallel_seconds_ = 0.0;
  double apply_seconds_ = 0.0;
};

}  // namespace

py::tuple train_temporal_native(
    py::iterable instance_chunks, int64_t total_anchors,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> spatial_support,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> spatial_anchor,
    int32_t dimension, int32_t lookahead, int64_t vocab_size, int64_t action_count, int threads) {
  if (lookahead < 1) throw std::invalid_argument("lookahead must be positive");
  const int32_t base_vocab = static_cast<int32_t>(spatial_support.size());
  if (vocab_size < base_vocab) {
    throw std::invalid_argument("vocab_size is below the spatial vocabulary size");
  }
  std::unique_ptr<Trainer> trainer;
  {
    py::gil_scoped_release release;
    trainer = std::make_unique<Trainer>(
        total_anchors, action_count, spatial_support.data(),
        spatial_anchor.data(), base_vocab, dimension, lookahead, vocab_size, threads);
  }
  int64_t instance_count = 0;
  for (py::handle raw_chunk : instance_chunks) {
    py::tuple chunk = py::cast<py::tuple>(raw_chunk);
    if (chunk.size() != 2) {
      throw std::invalid_argument("every temporal instance chunk must be a token/anchor pair");
    }
    auto tokens = py::cast<
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(chunk[0]);
    auto anchors = py::cast<
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>(chunk[1]);
    if (tokens.ndim() != 1 || anchors.ndim() != 1 || tokens.size() != anchors.size()) {
      throw std::invalid_argument("every temporal token/anchor chunk must be an aligned vector");
    }
    for (py::ssize_t index = 0; index < tokens.size(); ++index) {
      if (tokens.data()[index] < 0 || tokens.data()[index] >= base_vocab) {
        throw std::invalid_argument("input temporal token is outside the spatial vocabulary");
      }
    }
    const InstanceChunk view{tokens.data(), anchors.data(), tokens.size()};
    {
      py::gil_scoped_release release;
      trainer->add_chunk(view);
    }
    instance_count += tokens.size();
  }
  if (instance_count <= 0) {
    throw std::invalid_argument("temporal instance chunks must not be empty");
  }
  int32_t left = 0;
  int32_t right = 0;
  int32_t offset = 0;
  std::vector<std::pair<int64_t, int64_t>> matches;
  double selection_seconds = 0.0;
  double commit_seconds = 0.0;
  {
    py::gil_scoped_release release;
    trainer->finalize_instances();
    while (static_cast<int64_t>(trainer->merges().size()) + base_vocab < vocab_size) {
      auto started = std::chrono::steady_clock::now();
      const bool found = trainer->best(left, right, offset, matches);
      selection_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      if (!found) break;
      started = std::chrono::steady_clock::now();
      trainer->commit(left, right, offset, matches);
      commit_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
  }

  py::list merge_list;
  for (const auto& merge : trainer->merges()) {
    merge_list.append(py::make_tuple(merge[0], merge[1], merge[2]));
  }
  py::dict report;
  report["lookahead"] = lookahead;
  report["spatial_vocab_size"] = base_vocab;
  report["merges"] = static_cast<int64_t>(trainer->merges().size());
  report["vocab_size"] = base_vocab + static_cast<int64_t>(trainer->merges().size());
  report["actions"] = action_count;
  report["tokens_per_action_initial"] =
      static_cast<float>(trainer->initial_count()) / static_cast<float>(action_count);
  report["tokens_per_action"] =
      static_cast<float>(trainer->active_count()) / static_cast<float>(action_count);
  report["compression"] =
      static_cast<float>(trainer->initial_count()) / static_cast<float>(trainer->active_count());
  report["initial_edge_types"] = static_cast<int64_t>(trainer->initial_edge_types());
  report["live_edge_types"] = static_cast<int64_t>(trainer->live_edge_types());
  report["counter_buckets"] = static_cast<int64_t>(trainer->counter_buckets());
  report["census_seconds"] = trainer->census_seconds();
  report["incident_seconds"] = trainer->incident_seconds();
  report["incident_parallel_seconds"] = trainer->incident_parallel_seconds();
  report["incident_merge_seconds"] = trainer->incident_merge_seconds();
  report["correction_seconds"] = trainer->correction_seconds();
  report["correction_parallel_seconds"] = trainer->correction_parallel_seconds();
  report["apply_seconds"] = trainer->apply_seconds();
  report["selection_seconds"] = selection_seconds;
  report["commit_seconds"] = commit_seconds;
  report["heap_pops"] = trainer->heap_pops();
  report["counter_updates"] = trainer->counter_updates();
  report["self_certifications"] = trainer->self_certifications();
  report["backend"] = "native_sparse_exact";
  report["threads"] = trainer->threads();
  report["threads_requested"] = threads;
  return py::make_tuple(merge_list, report);
}
