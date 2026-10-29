
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common.hpp"

namespace py = pybind11;

namespace {

inline uint64_t pair_key(int32_t left, int32_t right) {
  uint32_t low = static_cast<uint32_t>(left < right ? left : right);
  uint32_t high = static_cast<uint32_t>(left < right ? right : left);
  return (static_cast<uint64_t>(low) << 32) | high;
}

struct Corpus {
  int64_t records = 0;
  int32_t width = 0;                 // slots per record == action dimension
  std::vector<int32_t> member;       // [records * width], -1 when free
  std::vector<int64_t> weights;      // [records]
};

}  // namespace

py::tuple train_spatial_native(py::array_t<int32_t, py::array::c_style | py::array::forcecast> codes,
                               py::array_t<int64_t, py::array::c_style | py::array::forcecast> weights,
                               const std::vector<int32_t>& bins, int64_t vocab_size,
                               int threads) {
  auto codes_view = codes.unchecked<2>();
  auto weights_view = weights.unchecked<1>();
  const int64_t records = codes_view.shape(0);
  const int32_t dimension = static_cast<int32_t>(codes_view.shape(1));
  if (dimension != static_cast<int32_t>(bins.size())) {
    throw std::invalid_argument("codes must have one column per bin count");
  }
  if (weights_view.shape(0) != records) {
    throw std::invalid_argument("weights must have one entry per record");
  }

  std::vector<int32_t> offsets(dimension + 1, 0);
  for (int32_t index = 0; index < dimension; ++index) offsets[index + 1] = offsets[index] + bins[index];
  const int32_t atom_count = offsets[dimension];
  if (vocab_size < atom_count) {
    throw std::invalid_argument("vocab_size is below the number of primitive atoms");
  }

#ifdef _OPENMP
  const int thread_count = threads > 0 ? threads : omp_get_max_threads();
#else
  (void)threads;
  const int thread_count = 1;
#endif

  Corpus corpus;
  corpus.records = records;
  corpus.width = dimension;
  corpus.member.resize(static_cast<size_t>(records) * dimension);
  corpus.weights.resize(records);
  int64_t total_weight = 0;
  int64_t initial_length = 0;
  for (int64_t record = 0; record < records; ++record) {
    corpus.weights[record] = weights_view(record);
    total_weight += weights_view(record);
    int32_t active = 0;
    for (int32_t slot = 0; slot < dimension; ++slot) {
      const int32_t value = codes_view(record, slot);
      if (value < -1 || value >= bins[slot]) {
        throw std::invalid_argument("a cell index is outside its coordinate's bin count");
      }
      if (value < 0) {
        corpus.member[static_cast<size_t>(record) * dimension + slot] = -1;
      } else {
        corpus.member[static_cast<size_t>(record) * dimension + slot] = offsets[slot] + value;
        ++active;
      }
    }
    if (active == 0) throw std::invalid_argument("every record needs an active coordinate");
    initial_length += weights_view(record) * active;
  }

  // Occurrence sets, kept sorted so intersections and differences are linear.
  std::vector<std::vector<int32_t>> occurrence(atom_count);
  for (int32_t slot = 0; slot < dimension; ++slot) {
    for (int64_t record = 0; record < records; ++record) {
      const int32_t token = corpus.member[static_cast<size_t>(record) * dimension + slot];
      if (token >= 0) occurrence[token].push_back(static_cast<int32_t>(record));
    }
  }
  for (auto& members : occurrence) std::sort(members.begin(), members.end());

  // --- initial census -------------------------------------------------------
  ac2::CounterMap counts(static_cast<size_t>(records) * 4 + 64);
  {
    std::vector<ac2::CounterMap> shards;
    shards.reserve(thread_count);
    for (int index = 0; index < thread_count; ++index) shards.emplace_back(1024);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count)
#endif
    {
#ifdef _OPENMP
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      ac2::CounterMap& local = shards[thread];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int64_t record = 0; record < records; ++record) {
        const int32_t* row = &corpus.member[static_cast<size_t>(record) * dimension];
        const int64_t weight = corpus.weights[record];
        for (int32_t a = 0; a < dimension; ++a) {
          if (row[a] < 0) continue;
          for (int32_t b = a + 1; b < dimension; ++b) {
            if (row[b] < 0) continue;
            local.add(pair_key(row[a], row[b]), weight);
          }
        }
      }
    }
    for (auto& shard : shards) {
      shard.for_each([&](uint64_t key, int64_t value) { counts.add(key, value); });
    }
  }

  std::vector<ac2::HeapEntry> heap;
  heap.reserve(counts.size());
  counts.for_each([&](uint64_t key, int64_t value) { heap.push_back({value, key, 0}); });
  std::make_heap(heap.begin(), heap.end(), ac2::HeapOrder{});
  const size_t initial_pair_types = heap.size();

  // --- greedy merging -------------------------------------------------------
  std::vector<std::pair<int32_t, int32_t>> merges;
  merges.reserve(static_cast<size_t>(vocab_size - atom_count));
  int64_t length = initial_length;  // total active tokens over the corpus
  int64_t heap_pops = 0;
  int64_t counter_updates = 0;

  std::vector<int32_t> matched;
  std::vector<int32_t> remainder;
  std::vector<int32_t> affected;
  std::vector<int32_t> stamp(static_cast<size_t>(vocab_size) + 16, 0);
  int32_t generation = 0;
  std::vector<ac2::CounterMap> deltas;
  deltas.reserve(thread_count);
  for (int index = 0; index < thread_count; ++index) deltas.emplace_back(1024);

  while (static_cast<int64_t>(merges.size()) + atom_count < vocab_size) {
    int32_t left = -1;
    int32_t right = -1;
    int64_t frequency = 0;
    while (!heap.empty()) {
      const ac2::HeapEntry top = ac2::heap_pop(heap);
      ++heap_pops;
      const int64_t current = counts.get(top.key);
      if (current <= 0) continue;
      if (current != top.count) continue;  // a fresher entry for this key exists
      left = static_cast<int32_t>(top.key >> 32);
      right = static_cast<int32_t>(top.key & 0xffffffffu);
      frequency = current;
      break;
    }
    if (left < 0) break;  // every record has collapsed to a single token

    matched.clear();
    std::set_intersection(occurrence[left].begin(), occurrence[left].end(),
                          occurrence[right].begin(), occurrence[right].end(),
                          std::back_inserter(matched));
    if (matched.empty()) {  // cannot happen while counters are exact
      counts.add(pair_key(left, right), -frequency);
      continue;
    }

    const int32_t child = static_cast<int32_t>(occurrence.size());
    occurrence.emplace_back();

    const bool parallel = matched.size() > 4096 && thread_count > 1;
    if (parallel) {
      for (auto& shard : deltas) shard = ac2::CounterMap(1024);
    }
#ifdef _OPENMP
#pragma omp parallel if (parallel) num_threads(thread_count)
#endif
    {
#ifdef _OPENMP
      const int thread = parallel ? omp_get_thread_num() : 0;
#else
      const int thread = 0;
#endif
      ac2::CounterMap& local = parallel ? deltas[thread] : counts;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int64_t index = 0; index < static_cast<int64_t>(matched.size()); ++index) {
        const int32_t record = matched[index];
        const int32_t* row = &corpus.member[static_cast<size_t>(record) * dimension];
        const int64_t weight = corpus.weights[record];
        for (int32_t slot = 0; slot < dimension; ++slot) {
          const int32_t other = row[slot];
          if (other < 0 || other == left || other == right) continue;
          local.add(pair_key(left, other), -weight);
          local.add(pair_key(right, other), -weight);
          local.add(pair_key(child, other), weight);
        }
      }
    }
    if (parallel) {
      for (auto& shard : deltas) {
        shard.for_each([&](uint64_t key, int64_t value) { counts.add(key, value); });
      }
    }

    // Rewrite the touched rows: the child takes the left parent's slot.
    for (const int32_t record : matched) {
      int32_t* row = &corpus.member[static_cast<size_t>(record) * dimension];
      for (int32_t slot = 0; slot < dimension; ++slot) {
        if (row[slot] == left) {
          row[slot] = child;
        } else if (row[slot] == right) {
          row[slot] = -1;
        }
      }
    }

    counts.add(pair_key(left, right), -frequency);
    counter_updates += static_cast<int64_t>(matched.size()) * dimension;

    occurrence[child] = matched;
    for (const int32_t parent : {left, right}) {
      remainder.clear();
      std::set_difference(occurrence[parent].begin(), occurrence[parent].end(), matched.begin(),
                          matched.end(), std::back_inserter(remainder));
      occurrence[parent].swap(remainder);
    }

    affected.clear();
    if (static_cast<size_t>(child) >= stamp.size()) stamp.resize(child + 1024, 0);
    ++generation;
    for (const int32_t record : matched) {
      const int32_t* row = &corpus.member[static_cast<size_t>(record) * dimension];
      for (int32_t slot = 0; slot < dimension; ++slot) {
        const int32_t other = row[slot];
        if (other < 0 || other == child) continue;
        if (stamp[other] == generation) continue;
        stamp[other] = generation;
        affected.push_back(other);
      }
    }
    for (const int32_t other : affected) {
      for (const int32_t parent : {left, right, child}) {
        const uint64_t key = pair_key(parent, other);
        const int64_t value = counts.get(key);
        if (value > 0) ac2::heap_push(heap, {value, key, 0});
      }
    }

    merges.emplace_back(left, right);
    length -= frequency;

    if (heap.size() > 4 * counts.size() + 65536) {
      heap.clear();
      counts.for_each([&](uint64_t key, int64_t value) { heap.push_back({value, key, 0}); });
      std::make_heap(heap.begin(), heap.end(), ac2::HeapOrder{});
    }
  }

  py::list merge_list;
  for (const auto& merge : merges) {
    merge_list.append(py::make_tuple(merge.first, merge.second));
  }
  py::dict report;
  report["records"] = records;
  report["weight"] = total_weight;
  report["atoms"] = atom_count;
  report["merges"] = static_cast<int64_t>(merges.size());
  report["vocab_size"] = atom_count + static_cast<int64_t>(merges.size());
  report["tokens_per_action_initial"] = static_cast<float>(initial_length) / total_weight;
  report["tokens_per_action"] = static_cast<float>(length) / static_cast<float>(total_weight);
  report["compression"] = static_cast<float>(initial_length) / static_cast<float>(length);
  report["initial_pair_types"] = static_cast<int64_t>(initial_pair_types);
  report["heap_pops"] = heap_pops;
  report["counter_updates"] = counter_updates;
  report["threads"] = thread_count;
  report["backend"] = "native_sparse_exact";
  return py::make_tuple(merge_list, report);
}
