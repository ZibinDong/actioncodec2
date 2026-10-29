// Shared primitives for the ActionCodec2 native kernels.
#pragma once

#include <cstdint>
#include <algorithm>
#include <utility>
#include <vector>

namespace ac2 {

class CounterMap {
 public:
  static constexpr uint64_t kEmpty = ~0ull;

  explicit CounterMap(size_t capacity_hint = 1024) { reserve(capacity_hint); }

  void reserve(size_t entries) {
    size_t target = 16;
    while (target * 3 < entries * 4) target <<= 1;  // keep load factor <= 0.75
    if (target <= mask_ + 1 && !keys_.empty()) return;
    rehash(target);
  }

  size_t size() const { return size_; }
  size_t buckets() const { return mask_ + 1; }

  void clear() {
    std::fill(keys_.begin(), keys_.end(), kEmpty);
    size_ = 0;
  }

  int64_t get(uint64_t key) const {
    size_t slot = probe(key);
    return keys_[slot] == key ? values_[slot] : 0;
  }

  int64_t add(uint64_t key, int64_t delta) {
    return add_with_previous(key, delta).second;
  }

  // Update and expose both values with one hash-table probe. This is useful
  // when a caller must distinguish a newly observed key from a stale one.
  std::pair<int64_t, int64_t> add_with_previous(uint64_t key, int64_t delta) {
    size_t slot = probe(key);
    if (keys_[slot] != key) {
      if ((size_ + 1) * 4 > (mask_ + 1) * 3) {
        rehash((mask_ + 1) << 1);
        slot = probe(key);
      }
      keys_[slot] = key;
      values_[slot] = 0;
      ++size_;
    }
    const int64_t previous = values_[slot];
    values_[slot] += delta;
    return {previous, values_[slot]};
  }

  template <typename Fn>
  void for_each(Fn&& fn) const {
    for (size_t slot = 0; slot <= mask_; ++slot) {
      if (keys_[slot] != kEmpty && values_[slot] != 0) fn(keys_[slot], values_[slot]);
    }
  }

  size_t live_entries() const {
    size_t live = 0;
    for (size_t slot = 0; slot <= mask_; ++slot) {
      if (keys_[slot] != kEmpty && values_[slot] != 0) ++live;
    }
    return live;
  }

 private:
  static uint64_t mix(uint64_t value) {  // splitmix64 finalizer
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
  }

  size_t probe(uint64_t key) const {
    size_t slot = mix(key) & mask_;
    while (keys_[slot] != kEmpty && keys_[slot] != key) slot = (slot + 1) & mask_;
    return slot;
  }

  void rehash(size_t buckets) {
    std::vector<uint64_t> old_keys;
    std::vector<int64_t> old_values;
    old_keys.swap(keys_);
    old_values.swap(values_);
    keys_.assign(buckets, kEmpty);
    values_.assign(buckets, 0);
    mask_ = buckets - 1;
    size_ = 0;
    for (size_t slot = 0; slot < old_keys.size(); ++slot) {
      if (old_keys[slot] != kEmpty && old_values[slot] != 0) {
        size_t target = probe(old_keys[slot]);
        keys_[target] = old_keys[slot];
        values_[target] = old_values[slot];
        ++size_;
      }
    }
  }

  std::vector<uint64_t> keys_;
  std::vector<int64_t> values_;
  size_t mask_ = 0;
  size_t size_ = 0;
};

struct HeapEntry {
  int64_t count;
  uint64_t key;
  uint32_t version;  // only shifted temporal self-rules use this
};

struct HeapOrder {
  bool operator()(const HeapEntry& a, const HeapEntry& b) const {
    if (a.count != b.count) return a.count < b.count;  // larger count first
    return a.key > b.key;                              // smaller key first
  }
};

inline void heap_push(std::vector<HeapEntry>& heap, const HeapEntry& entry) {
  heap.push_back(entry);
  std::push_heap(heap.begin(), heap.end(), HeapOrder{});
}

inline HeapEntry heap_pop(std::vector<HeapEntry>& heap) {
  std::pop_heap(heap.begin(), heap.end(), HeapOrder{});
  HeapEntry top = heap.back();
  heap.pop_back();
  return top;
}

}  // namespace ac2
