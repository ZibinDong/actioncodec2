// Native BPE trainer -- the FAST-style ablation baseline.
//
// Set-BPE treats one control step as an unordered set of primitive cells and
// merges spatially before merging across time.  Plain BPE instead serializes
// the quantized ``[T, D]`` cell matrix row-major into one symbol sequence and
// runs textbook byte-pair encoding on it, which is what FAST does after its
// DCT stage.  Both trainers consume the identical quantized corpus,
// so a paired run isolates the tokenizer structure and nothing else.
//
// Correctness is identical to the Python reference trainer in
// ``actioncodec2/bpe_training.py``, including the tie-break (largest count,
// then lexicographically smallest ordered ``(left, right)`` pair) and the
// left-to-right application order, so the two emit the same merge list.
//
// Where the time goes, and what is done about it:
//   * occurrences of every observed ordered pair are indexed once, so a merge
//     touches only the positions that actually carry it;
//   * the corpus is a doubly linked list over one flat symbol array, so
//     collapsing a pair is O(1) and never rewrites the tail;
//   * counters live in the same flat linear-probing table as the Set-BPE
//     kernels (common.hpp), and the heap is lazy with per-merge deduplication
//     of the keys a merge could have changed.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

namespace py = pybind11;

namespace {

// BPE pairs are ordered, so the key keeps ``left`` in the high word.  The
// heap tie-break in common.hpp prefers the smallest key, which is therefore the
// lexicographically smallest ``(left, right)`` -- the Python reference rule.
inline uint64_t ordered_key(int32_t left, int32_t right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(right));
}

inline int32_t key_left(uint64_t key) { return static_cast<int32_t>(key >> 32); }
inline int32_t key_right(uint64_t key) { return static_cast<int32_t>(key & 0xffffffffull); }

// Per-merge set of pair keys whose counters moved.  A merge can only touch
// pairs incident to its two parents or its child, so the live population is
// bounded by a small multiple of the vocabulary; clearing is a generation bump
// rather than a table sweep.
class StampSet {
 public:
  explicit StampSet(size_t capacity_hint) {
    size_t target = 1024;
    while (target < capacity_hint * 4) target <<= 1;
    keys_.assign(target, 0);
    stamps_.assign(target, 0);
    mask_ = target - 1;
  }

  void reset() {
    ++generation_;
    members_.clear();
    if (generation_ != 0) return;
    std::fill(stamps_.begin(), stamps_.end(), 0);
    generation_ = 1;
  }

  void insert(uint64_t key) {
    size_t slot = mix(key) & mask_;
    while (stamps_[slot] == generation_) {
      if (keys_[slot] == key) return;
      slot = (slot + 1) & mask_;
    }
    if ((members_.size() + 1) * 4 > (mask_ + 1) * 3) {
      grow();
      slot = mix(key) & mask_;
      while (stamps_[slot] == generation_) {
        if (keys_[slot] == key) return;
        slot = (slot + 1) & mask_;
      }
    }
    keys_[slot] = key;
    stamps_[slot] = generation_;
    members_.push_back(key);
  }

  const std::vector<uint64_t>& members() const { return members_; }

 private:
  static uint64_t mix(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
  }

  void grow() {
    const size_t buckets = (mask_ + 1) << 1;
    keys_.assign(buckets, 0);
    stamps_.assign(buckets, 0);
    mask_ = buckets - 1;
    generation_ = 1;
    for (const uint64_t key : members_) {
      size_t slot = mix(key) & mask_;
      while (stamps_[slot] == generation_) slot = (slot + 1) & mask_;
      keys_[slot] = key;
      stamps_[slot] = generation_;
    }
  }

  std::vector<uint64_t> keys_;
  std::vector<uint32_t> stamps_;
  std::vector<uint64_t> members_;
  size_t mask_ = 0;
  uint32_t generation_ = 1;
};

// Occurrence index.  ``slots`` maps a pair key to its 1-based row in
// ``lists``; a value of zero means "no row yet", which is exactly the absent
// value CounterMap already reports, so no separate presence bit is needed.
struct OccurrenceIndex {
  ac2::CounterMap slots{1 << 16};
  std::vector<std::vector<int32_t>> lists;
  std::vector<uint64_t> keys;
  int64_t stored = 0;

  int64_t row_for(uint64_t key) {
    const int64_t existing = slots.get(key);
    if (existing > 0) return existing - 1;
    const int64_t row = static_cast<int64_t>(lists.size());
    lists.emplace_back();
    keys.push_back(key);
    slots.add(key, row + 1);
    return row;
  }

  void append(uint64_t key, int32_t position) {
    lists[static_cast<size_t>(row_for(key))].push_back(position);
    ++stored;
  }
};

struct Corpus {
  std::vector<int32_t> symbol;  // live token id per slot, -1 once absorbed
  std::vector<int32_t> prev;
  std::vector<int32_t> next;
  int64_t actions = 0;
  int64_t episodes = 0;
};

void ingest_chunk(Corpus& corpus, const int32_t* cells, int64_t rows, int32_t dimension,
                  const int32_t* offsets, const int32_t* bins, const int64_t* episode_offsets,
                  int64_t episode_count) {
  for (int64_t episode = 0; episode < episode_count; ++episode) {
    const int64_t start = episode_offsets[episode];
    const int64_t stop = episode_offsets[episode + 1];
    if (stop <= start || start < 0 || stop > rows) {
      throw std::invalid_argument("BPE episode offsets must be increasing and in range");
    }
    const size_t first = corpus.symbol.size();
    const int32_t* initial = cells + start * dimension;
    for (int64_t row = start; row < stop; ++row) {
      const int32_t* values = cells + row * dimension;
      for (int32_t dim = 0; dim < dimension; ++dim) {
        const int32_t value = values[dim];
        if ((value >= 0) != (initial[dim] >= 0)) {
          throw std::invalid_argument("active dimensions must remain constant within an episode");
        }
        if (value < 0) {
          if (value != -1) throw std::invalid_argument("cells must contain -1 or a valid index");
          continue;
        }
        if (value >= bins[dim]) throw std::invalid_argument("cell index exceeds its bin count");
        corpus.symbol.push_back(offsets[dim] + value);
      }
    }
    const size_t last = corpus.symbol.size();
    if (last == first) throw std::invalid_argument("every control step needs an active coordinate");
    if (last > static_cast<size_t>(INT32_MAX)) {
      throw std::invalid_argument(
          "BPE corpus exceeds 2^31 primitive cells; lower the fitting coverage");
    }
    corpus.prev.resize(last);
    corpus.next.resize(last);
    for (size_t index = first; index < last; ++index) {
      corpus.prev[index] = index == first ? -1 : static_cast<int32_t>(index - 1);
      corpus.next[index] = index + 1 == last ? -1 : static_cast<int32_t>(index + 1);
    }
    corpus.actions += stop - start;
    ++corpus.episodes;
  }
}

}  // namespace

py::tuple train_bpe_native(py::iterable cell_chunks, const std::vector<int32_t>& bins,
                            int64_t vocab_size, int64_t max_token_cells, int threads) {
  const int32_t dimension = static_cast<int32_t>(bins.size());
  if (dimension < 1) throw std::invalid_argument("bins must not be empty");
  std::vector<int32_t> offsets(dimension + 1, 0);
  for (int32_t index = 0; index < dimension; ++index) {
    if (bins[index] < 1) throw std::invalid_argument("every dimension needs at least one bin");
    offsets[index + 1] = offsets[index] + bins[index];
  }
  const int32_t atom_count = offsets[dimension];
  if (vocab_size < atom_count) {
    throw std::invalid_argument("vocab_size is below the number of primitive atoms");
  }
  if (max_token_cells < 1) throw std::invalid_argument("max_token_cells must be positive");

  // Greedy merge updates are serial. Preserve the requested option for the
  // shared trainer API without changing the calling thread's OpenMP settings.
  const int thread_count = 1;

  Corpus corpus;
  for (py::handle raw_chunk : cell_chunks) {
    py::tuple chunk = py::cast<py::tuple>(raw_chunk);
    if (chunk.size() != 2) throw std::invalid_argument("each chunk must be (cells, offsets)");
    auto cells = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(chunk[0]);
    auto bounds =
        py::cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>(chunk[1]);
    auto cells_view = cells.unchecked<2>();
    auto bounds_view = bounds.unchecked<1>();
    if (cells_view.shape(1) != dimension) {
      throw std::invalid_argument("cells must have one column per bin count");
    }
    if (bounds_view.shape(0) < 2) throw std::invalid_argument("offsets must bound one episode");
    {
      py::gil_scoped_release release;
      ingest_chunk(corpus, cells_view.data(0, 0), cells_view.shape(0), dimension, offsets.data(),
                   bins.data(), bounds_view.data(0), bounds_view.shape(0) - 1);
    }
  }
  if (corpus.symbol.empty()) throw std::invalid_argument("BPE training needs one episode");

  const int64_t initial_length = static_cast<int64_t>(corpus.symbol.size());
  int64_t length = initial_length;
  std::vector<int32_t>& symbol = corpus.symbol;
  std::vector<int32_t>& prev = corpus.prev;
  std::vector<int32_t>& next = corpus.next;

  ac2::CounterMap counts(1 << 16);
  OccurrenceIndex index;
  std::vector<std::pair<int32_t, int32_t>> merges;
  std::vector<int32_t> token_cells(static_cast<size_t>(atom_count), 1);
  int64_t initial_pair_types = 0;
  int64_t heap_pops = 0;
  int64_t counter_updates = 0;
  int64_t compactions = 0;
  int64_t peak_stored = 0;
  int64_t oversized_pairs = 0;
  int32_t observed_cells = 1;

  {
  py::gil_scoped_release release;

  // One pass builds ascending occurrence lists; the counter is their length.
  for (int64_t position = 0; position < initial_length; ++position) {
    const int32_t follower = next[position];
    if (follower < 0) continue;
    index.append(ordered_key(symbol[position], symbol[follower]),
                 static_cast<int32_t>(position));
  }
  for (size_t row = 0; row < index.lists.size(); ++row) {
    counts.add(index.keys[row], static_cast<int64_t>(index.lists[row].size()));
  }
  initial_pair_types = static_cast<int64_t>(index.lists.size());
  peak_stored = index.stored;

  std::vector<ac2::HeapEntry> heap;
  heap.reserve(index.lists.size());
  counts.for_each([&](uint64_t key, int64_t value) { heap.push_back({value, key, 0}); });
  std::make_heap(heap.begin(), heap.end(), ac2::HeapOrder{});

  token_cells.reserve(static_cast<size_t>(vocab_size));
  merges.reserve(static_cast<size_t>(vocab_size - atom_count));

  ac2::CounterMap oversized(1 << 12);
  StampSet touched(static_cast<size_t>(vocab_size));
  std::vector<int32_t> occurrences;

  while (static_cast<int64_t>(atom_count + merges.size()) < vocab_size && !heap.empty()) {
    const ac2::HeapEntry top = ac2::heap_pop(heap);
    ++heap_pops;
    const uint64_t key = top.key;
    const int64_t frequency = counts.get(key);
    if (frequency <= 0 || frequency != top.count) continue;
    if (oversized.get(key) != 0) continue;
    const int32_t left = key_left(key);
    const int32_t right = key_right(key);
    const int64_t span = static_cast<int64_t>(token_cells[left]) + token_cells[right];
    if (span > max_token_cells) {
      oversized.add(key, 1);
      continue;
    }

    const int32_t child = atom_count + static_cast<int32_t>(merges.size());
    token_cells.push_back(static_cast<int32_t>(span));
    observed_cells = std::max(observed_cells, static_cast<int32_t>(span));

    occurrences.clear();
    occurrences.swap(index.lists[static_cast<size_t>(index.slots.get(key) - 1)]);
    index.stored -= static_cast<int64_t>(occurrences.size());
    // Left-to-right application is what makes overlapping runs deterministic;
    // later appends leave the list unsorted, so restore the order first.
    std::sort(occurrences.begin(), occurrences.end());
    touched.reset();
    touched.insert(key);

    int64_t applied = 0;
    for (const int32_t position : occurrences) {
      if (symbol[position] != left) continue;
      const int32_t follower = next[position];
      if (follower < 0 || symbol[follower] != right) continue;
      const int32_t before = prev[position];
      const int32_t after = next[follower];
      if (before >= 0) {
        const uint64_t stale = ordered_key(symbol[before], left);
        counts.add(stale, -1);
        touched.insert(stale);
      }
      if (after >= 0) {
        const uint64_t stale = ordered_key(right, symbol[after]);
        counts.add(stale, -1);
        touched.insert(stale);
      }
      symbol[position] = child;
      symbol[follower] = -1;
      next[position] = after;
      if (after >= 0) prev[after] = position;
      prev[follower] = -1;
      next[follower] = -1;
      if (before >= 0) {
        const uint64_t fresh = ordered_key(symbol[before], child);
        counts.add(fresh, 1);
        index.append(fresh, before);
        touched.insert(fresh);
      }
      if (after >= 0) {
        const uint64_t fresh = ordered_key(child, symbol[after]);
        counts.add(fresh, 1);
        index.append(fresh, position);
        touched.insert(fresh);
      }
      ++applied;
    }
    counts.add(key, -applied);
    counter_updates += 4 * applied;
    length -= applied;
    merges.emplace_back(left, right);

    for (const uint64_t member : touched.members()) {
      const int64_t value = counts.get(member);
      if (value > 0) ac2::heap_push(heap, {value, member, 0});
    }
    peak_stored = std::max(peak_stored, index.stored);

    // Superseded positions stay in the lists of pairs that never won.  Sweep
    // them out once they outnumber the live corpus, so memory tracks the
    // corpus rather than the merge history.
    if (index.stored > 3 * length) {
      int64_t retained = 0;
      for (size_t row = 0; row < index.lists.size(); ++row) {
        auto& list = index.lists[row];
        if (list.empty()) continue;
        const int32_t expected_left = key_left(index.keys[row]);
        const int32_t expected_right = key_right(index.keys[row]);
        size_t write = 0;
        for (const int32_t position : list) {
          const int32_t follower = next[position];
          if (symbol[position] != expected_left) continue;
          if (follower < 0 || symbol[follower] != expected_right) continue;
          list[write++] = position;
        }
        list.resize(write);
        if (write * 2 < list.capacity()) list.shrink_to_fit();
        retained += static_cast<int64_t>(write);
      }
      index.stored = retained;
      ++compactions;
    }

    if (heap.size() > 4 * counts.size() + 65536) {
      heap.clear();
      counts.for_each([&](uint64_t member, int64_t value) { heap.push_back({value, member, 0}); });
      std::make_heap(heap.begin(), heap.end(), ac2::HeapOrder{});
    }
  }

  oversized_pairs = static_cast<int64_t>(oversized.live_entries());
  }

  py::list merge_list;
  for (const auto& merge : merges) {
    merge_list.append(py::make_tuple(merge.first, merge.second));
  }
  py::dict report;
  report["episodes"] = corpus.episodes;
  report["actions"] = corpus.actions;
  report["atoms"] = static_cast<int64_t>(atom_count);
  report["merges"] = static_cast<int64_t>(merges.size());
  report["vocab_size"] = static_cast<int64_t>(atom_count + merges.size());
  report["cells"] = initial_length;
  report["tokens_per_action_initial"] =
      static_cast<double>(initial_length) / static_cast<double>(corpus.actions);
  report["tokens_per_action"] =
      static_cast<double>(length) / static_cast<double>(corpus.actions);
  report["compression"] = static_cast<double>(initial_length) / static_cast<double>(length);
  report["max_token_cells"] = max_token_cells;
  report["max_token_cells_observed"] = static_cast<int64_t>(observed_cells);
  report["oversized_pairs_skipped"] = oversized_pairs;
  report["initial_pair_types"] = initial_pair_types;
  report["pair_types_peak"] = static_cast<int64_t>(index.lists.size());
  report["occurrence_slots_peak"] = peak_stored;
  report["occurrence_compactions"] = compactions;
  report["heap_pops"] = heap_pops;
  report["counter_updates"] = counter_updates;
  report["threads"] = thread_count;
  report["requested_threads"] = threads;
  report["trainer"] = "native_sparse_exact";
  return py::make_tuple(merge_list, report);
}
