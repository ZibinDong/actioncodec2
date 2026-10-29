
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "parallel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#define AC2_PARALLEL_SORT __gnu_parallel::sort
#else
#define AC2_PARALLEL_SORT std::sort
#endif

namespace py = pybind11;

namespace {

constexpr int kMaxLookahead = 32;
constexpr size_t kCertifyBatch = 32;
constexpr size_t kApplyGrain = 256;  // minimum candidate pairs per parallel apply group

constexpr size_t kShards = 64;  // rule-table shards updated concurrently after a merge

inline uint64_t mix(uint64_t key) { return key * 0x9E3779B97F4A7C15ull; }
inline size_t shard_of(uint64_t key) { return static_cast<size_t>(mix(key) >> 58); }

// Small open-addressing accumulator for rule-count deltas; clearing costs O(used).
class DeltaMap {
 public:
  void add(uint64_t key, int64_t delta) {
    if ((filled_.size() + 1) * 2 > keys_.size()) grow();
    size_t h = slot(key);
    while (keys_[h] != kEmpty && keys_[h] != key) h = (h + 1) & (keys_.size() - 1);
    if (keys_[h] == kEmpty) {
      keys_[h] = key;
      filled_.push_back(static_cast<uint32_t>(h));
    }
    values_[h] += delta;
  }
  template <class F>
  void each(F&& visit) const {
    for (uint32_t h : filled_) visit(keys_[h], values_[h]);
  }
  void clear() {
    for (uint32_t h : filled_) {
      keys_[h] = kEmpty;
      values_[h] = 0;
    }
    filled_.clear();
  }

 private:
  static constexpr uint64_t kEmpty = ~0ull;
  size_t slot(uint64_t key) const {
    return static_cast<size_t>(mix(key) >> 20) & (keys_.size() - 1);
  }
  void grow() {
    std::vector<uint64_t> keys = std::move(keys_);
    std::vector<int64_t> values = std::move(values_);
    std::vector<uint32_t> filled = std::move(filled_);
    const size_t capacity = std::max<size_t>(64, keys.size() * 2);
    keys_.assign(capacity, kEmpty);
    values_.assign(capacity, 0);
    filled_.clear();
    for (uint32_t h : filled) add(keys[h], values[h]);
  }
  std::vector<uint64_t> keys_;
  std::vector<int64_t> values_;
  std::vector<uint32_t> filled_;
};

inline uint64_t rule_key(int32_t left, int32_t right, int32_t offset) {
  return (static_cast<uint64_t>(left) << 40) | (static_cast<uint64_t>(right) << 16) |
         static_cast<uint64_t>(offset);
}

inline void unpack(uint64_t key, int32_t& left, int32_t& right, int32_t& offset) {
  left = static_cast<int32_t>(key >> 40);
  right = static_cast<int32_t>((key >> 16) & 0xffffffull);
  offset = static_cast<int32_t>(key & 0xffffull);
}

// Immutable-per-token geometry shared by the trainer, encoder and decoder.
struct Tables {
  int32_t dims = 0;       // physical slots including E
  int32_t e_dim = 0;      // index of E
  int32_t lookahead = 0;  // maximum token span in frames
  std::vector<uint32_t> masks;                 // token * lookahead -> support mask
  std::vector<uint32_t> emask;                 // token -> bit per offset holding E
  std::vector<int8_t> first_from;              // token * lookahead + k -> first support offset >= k
  std::vector<int32_t> span, rdim;
  std::vector<std::vector<std::pair<int32_t, int32_t>>> pattern;  // (offset, spatial token)
  int32_t base = 0;

  void init(const std::vector<uint32_t>& spatial_support, int32_t dims_, int32_t lookahead_) {
    dims = dims_;
    e_dim = dims_ - 1;
    lookahead = lookahead_;
    if (lookahead < 1 || lookahead > kMaxLookahead) throw std::invalid_argument("lookahead");
    if (dims < 2 || dims > 16) throw std::invalid_argument("dims must be in [2, 16]");
    base = static_cast<int32_t>(spatial_support.size());
    for (int32_t t = 0; t < base; ++t) {
      uint32_t s = spatial_support[t];
      if (!s) throw std::invalid_argument("spatial token without support");
      push({{0, t}}, std::vector<uint32_t>{s});
    }
  }

  void push(std::vector<std::pair<int32_t, int32_t>> pat, const std::vector<uint32_t>& m) {
    const size_t token = span.size();
    masks.resize((token + 1) * lookahead, 0);
    uint32_t e = 0;
    int32_t sp = 0;
    for (size_t o = 0; o < m.size(); ++o) {
      masks[token * lookahead + o] = m[o];
      if (m[o]) sp = static_cast<int32_t>(o) + 1;
      if (m[o] >> e_dim & 1u) e |= 1u << o;
    }
    emask.push_back(e);
    first_from.resize((token + 1) * lookahead, -1);
    for (int32_t k = sp - 1, next = -1; k >= 0; --k) {
      if (masks[token * lookahead + k]) next = k;
      first_from[token * lookahead + k] = static_cast<int8_t>(next);
    }
    span.push_back(sp);
    rdim.push_back(__builtin_ctz(m[0]));
    pattern.push_back(std::move(pat));
  }

  int32_t size() const { return static_cast<int32_t>(span.size()); }
  uint32_t mask(int32_t token, int32_t o) const {
    return masks[static_cast<size_t>(token) * lookahead + o];
  }

  // Merge geometry; returns false when the union would exceed the lookahead or overlap.
  bool compose(int32_t a, int32_t b, int32_t off, std::vector<uint32_t>& out) const {
    if (off < 0 || off >= lookahead) return false;
    const int32_t sp = std::max(span[a], off + span[b]);
    if (sp > lookahead) return false;
    out.assign(sp, 0);
    for (int32_t o = 0; o < span[a]; ++o) out[o] = mask(a, o);
    for (int32_t o = 0; o < span[b]; ++o) {
      if (out[o + off] & mask(b, o)) return false;
      out[o + off] |= mask(b, o);
    }
    return true;
  }

  int32_t add_merge(int32_t a, int32_t b, int32_t off) {
    std::vector<uint32_t> m;
    if (!compose(a, b, off, m)) throw std::invalid_argument("illegal temporal merge");
    auto pat = pattern[a];
    for (auto [o, t] : pattern[b]) pat.push_back({o + off, t});
    std::sort(pat.begin(), pat.end());
    push(std::move(pat), m);
    return size() - 1;
  }
};

using Key = std::tuple<int64_t, int32_t, int64_t, int32_t>;
using InstanceId = uint32_t;
constexpr InstanceId kNoInstance = UINT32_MAX;

// Mutable instance state over a global frame axis.  Used per chunk by the encoder and
// globally by the trainer.
struct Instances {
  const Tables* tb = nullptr;
  int64_t frames = 0;
  std::vector<int32_t> tok;
  std::vector<int64_t> root;
  std::vector<uint8_t> alive;
  std::vector<InstanceId> member;  // frame * dims + dim -> instance

  void reset(const Tables* tables, int64_t frame_count) {
    tb = tables;
    frames = frame_count;
    tok.clear();
    root.clear();
    alive.clear();
    member.assign(static_cast<size_t>(frame_count) * tables->dims, kNoInstance);
  }

  InstanceId cell(int64_t f, int32_t d) const {
    return member[static_cast<size_t>(f) * tb->dims + d];
  }

  Key key(InstanceId i) const {
    const uint32_t e = tb->emask[tok[i]];
    return Key(root[i], e ? 1 : 0, e ? root[i] + __builtin_ctz(e) : 0, tb->rdim[tok[i]]);
  }

  InstanceId add(int32_t token, int64_t r) {
    if (tok.size() >= kNoInstance) throw std::length_error("instance ids");
    const InstanceId i = static_cast<InstanceId>(tok.size());
    tok.push_back(token);
    root.push_back(r);
    alive.push_back(1);
    place(i);
    return i;
  }

  // Reserves dead slots that disjoint frame ranges then fill concurrently via ``place``.
  InstanceId reserve(size_t count) {
    const size_t base = tok.size();
    if (base + count > static_cast<size_t>(kNoInstance)) throw std::length_error("instance ids");
    tok.resize(base + count, 0);
    root.resize(base + count, 0);
    alive.resize(base + count, 0);
    return static_cast<InstanceId>(base);
  }

  void place(InstanceId i) {
    const int32_t token = tok[i];
    const int64_t r = root[i];
    alive[i] = 1;
    for (int32_t o = 0; o < tb->span[token]; ++o) {
      uint32_t m = tb->mask(token, o);
      while (m) {
        const int32_t d = __builtin_ctz(m);
        m &= m - 1;
        InstanceId& slot = member[static_cast<size_t>(r + o) * tb->dims + d];
        if (slot != kNoInstance) throw std::runtime_error("instance overlap");
        slot = i;
      }
    }
  }

  void kill(InstanceId i) {
    alive[i] = 0;
    const int32_t token = tok[i];
    for (int32_t o = 0; o < tb->span[token]; ++o) {
      uint32_t m = tb->mask(token, o);
      while (m) {
        const int32_t d = __builtin_ctz(m);
        m &= m - 1;
        member[static_cast<size_t>(root[i] + o) * tb->dims + d] = kNoInstance;
      }
    }
  }

  bool first(InstanceId a, InstanceId b) const {
    return std::make_pair(root[a], tb->rdim[tok[a]]) < std::make_pair(root[b], tb->rdim[tok[b]]);
  }

  // Ordered candidate pairs (a, b, offset) involving i inside the lookahead window.
  // Each neighbour is emitted once, at its first support cell inside the window.
  template <class F>
  void partners(InstanceId i, bool allow_zero, F&& emit) const {
    const int64_t r = root[i];
    const int32_t L = tb->lookahead;
    const int64_t lo = std::max<int64_t>(0, r - L + 1), hi = std::min<int64_t>(frames - 1, r + L - 1);
    for (int64_t f = lo; f <= hi; ++f) {
      for (int32_t d = 0; d < tb->dims; ++d) {
        const InstanceId j = cell(f, d);
        if (j == kNoInstance || j == i) continue;
        const int32_t tj = tok[j];
        const int64_t o = f - root[j];
        const int64_t k = std::max<int64_t>(0, lo - root[j]);
        if (tb->first_from[static_cast<size_t>(tj) * L + k] != o) continue;
        if (__builtin_ctz(tb->mask(tj, static_cast<int32_t>(o))) != d) continue;
        const bool ij = first(i, j);
        const InstanceId a = ij ? i : j, b = ij ? j : i;
        const int64_t off = root[b] - root[a];
        if (off >= L || (off == 0 && !allow_zero)) continue;
        if (std::max<int64_t>(tb->span[tok[a]], off + tb->span[tok[b]]) > L) continue;
        emit(a, b, static_cast<int32_t>(off));
      }
    }
  }

  // Would merging a (rooted first) with b keep the stream decodable?
  bool valid(InstanceId a, InstanceId b, int32_t off) const {
    const int32_t ta = tok[a], tb_ = tok[b];
    const int32_t sp = std::max(tb->span[ta], off + tb->span[tb_]);
    uint32_t emask = tb->emask[ta] | (tb->emask[tb_] << off);
    const Key km(root[a], emask ? 1 : 0, emask ? root[a] + __builtin_ctz(emask) : 0,
                 tb->rdim[ta]);
    for (int32_t o = 0; o < sp; ++o) {
      uint32_t m = (o < tb->span[ta] ? tb->mask(ta, o) : 0) |
                   (o >= off && o - off < tb->span[tb_] ? tb->mask(tb_, o - off) : 0);
      if (!m) continue;
      const int64_t u = root[a] + o;
      const InstanceId h = cell(u, tb->e_dim);
      if (h != kNoInstance && h != a && h != b && !(key(h) > km)) return false;
      if (emask >> o & 1u) {
        for (int32_t d = 0; d < tb->dims; ++d) {
          const InstanceId j = cell(u, d);
          if (j != kNoInstance && j != a && j != b && !(key(j) < km)) return false;
        }
      }
    }
    return true;
  }

  // Right partner of left instance a under rule (left, right, off), or -1.
  InstanceId right_of(InstanceId a, int32_t right, int32_t off) const {
    const int64_t f = root[a] + off;
    if (f >= frames) return kNoInstance;
    const InstanceId b = cell(f, tb->rdim[right]);
    if (b == kNoInstance || b == a || !alive[b] || tok[b] != right || root[b] != f) return kNoInstance;
    return b;
  }

  // Left partner of right instance b under rule (left, right, off), or -1.
  InstanceId left_of(InstanceId b, int32_t left, int32_t off) const {
    const int64_t f = root[b] - off;
    if (f < 0) return kNoInstance;
    const InstanceId a = cell(f, tb->rdim[left]);
    if (a == kNoInstance || a == b || !alive[a] || tok[a] != left || root[a] != f) return kNoInstance;
    return a;
  }

  std::vector<int32_t> serialize() const {
    std::vector<InstanceId> live;
    for (size_t i = 0; i < tok.size(); ++i)
      if (alive[i]) live.push_back(static_cast<InstanceId>(i));
    std::sort(live.begin(), live.end(), [&](InstanceId x, InstanceId y) { return key(x) < key(y); });
    std::vector<int32_t> out;
    out.reserve(live.size());
    for (InstanceId i : live) out.push_back(tok[i]);
    return out;
  }
};

struct Stat {
  bool retired = false;  // a rule is learned at most once; leftovers stay unmerged
  int64_t raw = 0;
  uint32_t version = 0;
  int64_t exact = -1;  // matching- and validity-certified count at exact_version
  uint32_t exact_version = 0;
  int64_t added = 0;  // cumulative pair insertions; bounds growth since certification
  int64_t added_at_exact = 0;
  uint32_t dirty_epoch = 0;  // merge batch that last queued this key for re-push
  bool live = false;         // a heap entry (queued_version, queued_count) represents the key
  uint32_t queued_version = 0;
  int64_t queued_count = 0;
};

// Open-addressing rule table; references are invalidated only by inserting a new key.
class StatMap {
 public:
  Stat* find(uint64_t key) {
    if (keys_.empty()) return nullptr;
    for (size_t h = slot(key);; h = (h + 1) & (keys_.size() - 1)) {
      if (keys_[h] == key) return &values_[h];
      if (keys_[h] == kEmpty) return nullptr;
    }
  }
  Stat& operator[](uint64_t key) {
    if ((used_ + 1) * 2 > keys_.size()) grow();
    size_t h = slot(key);
    while (keys_[h] != kEmpty && keys_[h] != key) h = (h + 1) & (keys_.size() - 1);
    if (keys_[h] == kEmpty) {
      keys_[h] = key;
      ++used_;
    }
    return values_[h];
  }
  size_t size() const { return used_; }
  template <class F>
  void each(F&& visit) const {
    for (size_t h = 0; h < keys_.size(); ++h)
      if (keys_[h] != kEmpty) visit(keys_[h], values_[h]);
  }

 private:
  static constexpr uint64_t kEmpty = ~0ull;
  size_t slot(uint64_t key) const {
    return static_cast<size_t>(mix(key) >> 20) & (keys_.size() - 1);
  }
  void grow() {
    std::vector<uint64_t> keys = std::move(keys_);
    std::vector<Stat> values = std::move(values_);
    const size_t capacity = std::max<size_t>(1 << 16, keys.size() * 2);
    keys_.assign(capacity, kEmpty);
    values_.assign(capacity, Stat{});
    used_ = 0;
    for (size_t h = 0; h < keys.size(); ++h)
      if (keys[h] != kEmpty) (*this)[keys[h]] = values[h];
  }
  std::vector<uint64_t> keys_;
  std::vector<Stat> values_;
  size_t used_ = 0;
};

// Rule table split by key hash so disjoint shards can be updated concurrently.
class ShardedStats {
 public:
  Stat* find(uint64_t key) { return shards_[shard_of(key)].find(key); }
  Stat& operator[](uint64_t key) { return shards_[shard_of(key)][key]; }
  StatMap& shard(size_t p) { return shards_[p]; }
  size_t size() const {
    size_t n = 0;
    for (const auto& s : shards_) n += s.size();
    return n;
  }
  template <class F>
  void each(F&& visit) const {
    for (const auto& s : shards_) s.each(visit);
  }

 private:
  StatMap shards_[kShards];
};

struct Entry {
  int64_t count;
  uint64_t key;
  uint32_t version;
  bool operator<(const Entry& o) const {
    if (count != o.count) return count < o.count;
    return key > o.key;
  }
};

class Trainer {
 public:
  Trainer(std::vector<uint32_t> spatial_support, int32_t dims, int32_t lookahead,
          int64_t vocab_size, bool allow_zero, int64_t min_count, int threads,
          bool dense_census = false, int64_t max_instances = UINT32_MAX)
      : vocab_size_(vocab_size), allow_zero_(allow_zero), dense_census_(dense_census),
        min_count_(min_count), max_instances_(max_instances) {
    if (max_instances_ < 1 || max_instances_ > UINT32_MAX)
      throw std::invalid_argument("max_instances");
    tables_.init(spatial_support, dims, lookahead);
#ifdef _OPENMP
    threads_ = threads > 0 ? threads : std::min(omp_get_max_threads(), 32);
#else
    (void)threads;
    threads_ = 1;
#endif
  }

  // anchors: global frame index per instance (chunks already separated by >= lookahead).
  void set_instances(py::array_t<int64_t, py::array::c_style> anchors,
                     py::array_t<int32_t, py::array::c_style> tokens, int64_t frames) {
    auto a = anchors.unchecked<1>();
    auto t = tokens.unchecked<1>();
    if (a.shape(0) != t.shape(0)) throw std::invalid_argument("anchors/tokens length");
    if (a.shape(0) > max_instances_)
      throw std::length_error("live instance capacity exceeded before training");
    st_.reset(&tables_, frames);
    // Pre-sized so references into the outer vector stay valid while children are added.
    occurrence_.assign(static_cast<size_t>(vocab_size_) + 16, {});
    dead_.assign(static_cast<size_t>(vocab_size_) + 16, 0);
    for (py::ssize_t i = 0; i < a.shape(0); ++i) {
      const int32_t tk = t(i);
      if (tk < 0 || tk >= tables_.base) throw std::invalid_argument("token outside spatial vocab");
      occurrence_[tk].push_back(st_.add(tk, a(i)));
    }
    for (auto& occ : occurrence_)
      std::sort(occ.begin(), occ.end(), [&](InstanceId x, InstanceId y) { return st_.root[x] < st_.root[y]; });
    initial_instances_ = a.shape(0);
    // Supports never change, so frames preceded by >= lookahead empty frames split the
    // axis into ranges whose merges never read or write each other's cells.
    safe_.clear();
    int64_t empty = tables_.lookahead;
    for (int64_t f = 0; f < frames; ++f) {
      bool occupied = false;
      for (int32_t d = 0; d < tables_.dims && !occupied; ++d) occupied = st_.cell(f, d) != kNoInstance;
      if (!occupied) {
        ++empty;
        continue;
      }
      if (empty >= tables_.lookahead) safe_.push_back(f);
      empty = 0;
    }
  }

  py::dict train(bool progress) {
    const auto t0 = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      census();
      std::vector<uint64_t> keys;
      keys.reserve(stats_.size());
      stats_.each([&](uint64_t k, const Stat&) { keys.push_back(k); });
      for (uint64_t k : keys) queue(k, stats_[k], stats_[k].raw);
      initial_types_ = stats_.size();
      census_seconds_ = seconds_since(t0);
      run(progress);
    }
    py::dict report;
    report["initial_instances"] = initial_instances_;
    report["final_instances"] = live_;
    report["initial_rule_types"] = initial_types_;
    report["merges"] = static_cast<int64_t>(merges_.size());
    report["dry_runs"] = dry_runs_;
    report["rejected_by_validity"] = rejected_;
    report["applied"] = applied_;
    report["census_seconds"] = census_seconds_;
    report["dry_run_seconds"] = dry_seconds_;
    report["apply_seconds"] = apply_seconds_;
    report["collect_seconds"] = collect_seconds_;
    report["parallel_seconds"] = parallel_seconds_;
    report["serial_seconds"] = serial_seconds_;
    report["merge_seconds"] = merge_seconds_;
    report["changes"] = changes_;
    report["compactions"] = compactions_;
    report["total_seconds"] = seconds_since(t0);
    return report;
  }

  std::vector<std::tuple<int32_t, int32_t, int32_t>> merges() const { return merges_; }

  // Count every forward pair once: blockwise parallel key generation, parallel sort, RLE.
  void census() {
    const int64_t n = static_cast<int64_t>(st_.tok.size());
    const int64_t base = tables_.base;
    const int64_t universe = base * base * tables_.lookahead;
    if (dense_census_ && universe <= 4000000) {
      std::vector<std::vector<int64_t>> local(
          threads_, std::vector<int64_t>(static_cast<size_t>(universe), 0));
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
      {
#ifdef _OPENMP
        auto& counts = local[omp_get_thread_num()];
#else
        auto& counts = local[0];
#endif
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t i = 0; i < n; ++i) {
          const InstanceId ii = static_cast<InstanceId>(i);
          st_.partners(ii, allow_zero_, [&](InstanceId a, InstanceId b, int32_t off) {
            if (a == ii) {
              const int64_t key = (static_cast<int64_t>(st_.tok[a]) * base + st_.tok[b]) *
                                      tables_.lookahead + off;
              ++counts[static_cast<size_t>(key)];
            }
          });
        }
      }
      for (int64_t key = 0; key < universe; ++key) {
        int64_t count = 0;
        for (const auto& counts : local) count += counts[static_cast<size_t>(key)];
        if (!count) continue;
        const int32_t offset = static_cast<int32_t>(key % tables_.lookahead);
        const int64_t pair = key / tables_.lookahead;
        const int32_t left = static_cast<int32_t>(pair / base);
        const int32_t right = static_cast<int32_t>(pair % base);
        stats_[rule_key(left, right, offset)].raw += count;
      }
      return;
    }
    const int64_t block = int64_t{1} << 22;
    std::vector<std::vector<uint64_t>> local(threads_);
    std::vector<uint64_t> keys;
    for (int64_t start = 0; start < n; start += block) {
      const int64_t stop = std::min(n, start + block);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
      {
#ifdef _OPENMP
        auto& out = local[omp_get_thread_num()];
#else
        auto& out = local[0];
#endif
        out.clear();
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t i = start; i < stop; ++i) {
          const InstanceId ii = static_cast<InstanceId>(i);
          st_.partners(ii, allow_zero_, [&](InstanceId a, InstanceId b, int32_t off) {
            if (a == ii) out.push_back(rule_key(st_.tok[a], st_.tok[b], off));
          });
        }
      }
      keys.clear();
      for (auto& v : local) keys.insert(keys.end(), v.begin(), v.end());
      AC2_PARALLEL_SORT(keys.begin(), keys.end());
      for (size_t k = 0; k < keys.size();) {
        size_t e = k;
        while (e < keys.size() && keys[e] == keys[k]) ++e;
        stats_[keys[k]].raw += static_cast<int64_t>(e - k);
        k = e;
      }
    }
  }

 private:
  // Reclaim dead slots without changing token order, coverage, or candidate statistics.
  void compact_instances() {
    std::vector<InstanceId> remap(st_.tok.size(), kNoInstance);
    std::vector<int32_t> tokens;
    std::vector<int64_t> roots;
    std::vector<uint8_t> alive;
    tokens.reserve(static_cast<size_t>(live_));
    roots.reserve(static_cast<size_t>(live_));
    alive.reserve(static_cast<size_t>(live_));
    for (size_t old = 0; old < st_.tok.size(); ++old) {
      if (!st_.alive[old]) continue;
      remap[old] = static_cast<InstanceId>(tokens.size());
      tokens.push_back(st_.tok[old]);
      roots.push_back(st_.root[old]);
      alive.push_back(1);
    }
    if (tokens.size() != static_cast<size_t>(live_))
      throw std::runtime_error("instance compaction live count mismatch");
    for (auto& occ : occurrence_) {
      size_t next = 0;
      for (InstanceId old : occ) {
        const InstanceId mapped = remap[static_cast<size_t>(old)];
        if (mapped != kNoInstance) occ[next++] = mapped;
      }
      occ.resize(next);
    }
    for (auto& slot : st_.member)
      if (slot != kNoInstance) slot = remap[static_cast<size_t>(slot)];
    st_.tok.swap(tokens);
    st_.root.swap(roots);
    st_.alive.swap(alive);
    std::fill(dead_.begin(), dead_.end(), 0);
    ++compactions_;
  }

  void change(uint64_t k, int64_t delta, StatMap& table, std::vector<uint64_t>& dirty) {
    Stat& s = table[k];
    s.raw += delta;
    if (delta > 0) s.added += delta;
    ++s.version;
    if (s.dirty_epoch != epoch_) {
      s.dirty_epoch = epoch_;
      dirty.push_back(k);
    }
  }

  std::vector<InstanceId>& lefts(int32_t left) {
    auto& occ = occurrence_[left];
    // Lists are root-sorted by construction: atoms are sorted once, and every child list
    // is appended in the root order of its single merge batch.
    if (dead_[left] * 4 > static_cast<int64_t>(occ.size())) {
      occ.erase(std::remove_if(occ.begin(), occ.end(), [&](InstanceId i) { return !st_.alive[i]; }),
                occ.end());
      dead_[left] = 0;
    }
    return occ;
  }

  static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  // Candidate (a, b) pairs are visited in increasing root order of a by scanning the
  // shorter occurrence list; the partner is one cell lookup away in either direction.
  const std::vector<InstanceId>& scanned(int32_t left, int32_t right, bool& forward) const {
    const auto& lo = occurrence_[left];
    const auto& ro = occurrence_[right];
    forward = left == right || lo.size() <= ro.size();
    return forward ? lo : ro;
  }

  template <class F>
  void pairs_in(int32_t left, int32_t right, int32_t off, size_t begin, size_t end,
                F&& visit) const {
    bool forward;
    const auto& scan = scanned(left, right, forward);
    for (size_t i = begin; i < end; ++i) {
      const InstanceId x = scan[i];
      if (!st_.alive[x]) continue;
      if (forward) {
        const InstanceId b = st_.right_of(x, right, off);
        if (b != kNoInstance) visit(x, b);
      } else {
        const InstanceId a = st_.left_of(x, left, off);
        if (a != kNoInstance) visit(a, x);
      }
    }
  }

  // Cuts of a root-sorted list into blocks of about ``grain`` entries at safe frames, so
  // per-block replay (self-rule chains included) equals one sequential pass.
  std::vector<size_t> blocks(const std::vector<InstanceId>& scan, size_t grain) const {
    std::vector<size_t> cuts{0};
    for (size_t next = grain; next < scan.size();) {
      auto frame = std::upper_bound(safe_.begin(), safe_.end(), st_.root[scan[next]]);
      if (frame == safe_.end()) break;
      const size_t at = static_cast<size_t>(
          std::lower_bound(scan.begin() + next, scan.end(), *frame,
                           [&](InstanceId i, int64_t f) { return st_.root[i] < f; }) -
          scan.begin());
      if (at >= scan.size()) break;
      cuts.push_back(at);
      next = at + grain;
    }
    cuts.push_back(scan.size());
    return cuts;
  }

  // Candidate pairs in scan order, gathered by blocks in parallel.
  std::vector<std::pair<InstanceId, InstanceId>> collect(int32_t left, int32_t right, int32_t off) const {
    bool forward;
    const std::vector<size_t> cuts = blocks(scanned(left, right, forward), 1 << 16);
    std::vector<std::vector<std::pair<InstanceId, InstanceId>>> parts(cuts.size() - 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_) if (parts.size() > 1)
#endif
    for (size_t k = 0; k < parts.size(); ++k)
      pairs_in(left, right, off, cuts[k], cuts[k + 1],
               [&](InstanceId a, InstanceId b) { parts[k].emplace_back(a, b); });
    std::vector<std::pair<InstanceId, InstanceId>> pairs;
    for (auto& part : parts) pairs.insert(pairs.end(), part.begin(), part.end());
    return pairs;
  }

  // Exact count of one block under the real merge's matching and validity tests.
  int64_t count_exact(int32_t left, int32_t right, int32_t off, size_t begin, size_t end) const {
    int64_t n = 0;
    // Self rules chain: a may be the right half of a recent match (roots increase).
    std::vector<InstanceId> recent;
    size_t head = 0;
    pairs_in(left, right, off, begin, end, [&](InstanceId a, InstanceId b) {
      if (left == right) {
        while (head < recent.size() && st_.root[recent[head]] < st_.root[a]) ++head;
        if (std::find(recent.begin() + head, recent.end(), a) != recent.end()) return;
      }
      if (!st_.valid(a, b, off)) return;
      if (left == right) recent.push_back(b);
      ++n;
    });
    return n;
  }

  // Certifies a batch of keys; every key is split into blocks so one large list does
  // not serialize the batch.  Read-only on shared state.
  void certify(std::vector<Entry>& batch) {
    struct Timer {
      double& sink;
      std::chrono::steady_clock::time_point t;
      ~Timer() { sink += seconds_since(t); }
    } timer{dry_seconds_, std::chrono::steady_clock::now()};
    dry_runs_ += static_cast<int64_t>(batch.size());
    struct Task {
      size_t key, begin, end;
    };
    std::vector<Task> tasks;
    for (size_t k = 0; k < batch.size(); ++k) {
      int32_t l, r, o;
      unpack(batch[k].key, l, r, o);
      lefts(l);  // compaction mutates lists: do it before the parallel region
      lefts(r);
      bool forward;
      const std::vector<size_t> cuts = blocks(scanned(l, r, forward), 1 << 15);
      for (size_t c = 0; c + 1 < cuts.size(); ++c) tasks.push_back({k, cuts[c], cuts[c + 1]});
    }
    std::vector<int64_t> partial(tasks.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_)
#endif
    for (size_t t = 0; t < tasks.size(); ++t) {
      int32_t l, r, o;
      unpack(batch[tasks[t].key].key, l, r, o);
      partial[t] = count_exact(l, r, o, tasks[t].begin, tasks[t].end);
    }
    std::vector<int64_t> exact(batch.size(), 0);
    for (size_t t = 0; t < tasks.size(); ++t) exact[tasks[t].key] += partial[t];
    for (size_t k = 0; k < batch.size(); ++k) {
      Stat& s = stats_[batch[k].key];
      s.exact = exact[k];
      s.exact_version = s.version;
      s.added_at_exact = s.added;
      if (s.exact >= min_count_) queue(batch[k].key, s, s.exact);
    }
  }

  // Pairs are applied in root order; ranges split at safe frames are independent, so
  // each is replayed sequentially by one thread with exactly the serial outcome.
  void apply(int32_t left, int32_t right, int32_t off, int32_t child) {
    struct Timer {
      double& sink;
      std::chrono::steady_clock::time_point t;
      ~Timer() { sink += seconds_since(t); }
    } timer{apply_seconds_, std::chrono::steady_clock::now()};
    lefts(left);
    lefts(right);
    // Candidates are collected first: applying merges kills instances in both lists.
    const auto collect_start = std::chrono::steady_clock::now();
    std::vector<std::pair<InstanceId, InstanceId>> pairs = collect(left, right, off);
    if (st_.tok.size() + pairs.size() > static_cast<size_t>(max_instances_)) {
      pairs.clear();
      pairs.shrink_to_fit();
      compact_instances();
      pairs = collect(left, right, off);
      if (st_.tok.size() + pairs.size() > static_cast<size_t>(max_instances_))
        throw std::length_error("live instance capacity exceeded after compaction");
    }
    collect_seconds_ += seconds_since(collect_start);
    const std::vector<size_t> cuts = split(pairs);
    const size_t groups = cuts.size() - 1;
    const InstanceId base = st_.reserve(pairs.size());
    if (pool_.size() < groups) pool_.resize(groups, std::vector<DeltaMap>(kShards));
    dirty_shards_.resize(kShards);
    std::vector<std::vector<InstanceId>> born(groups);
    std::vector<int64_t> applied(groups, 0), rejected(groups, 0);
    const auto loop_start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(std::min<size_t>(threads_, groups))
#endif
    for (size_t g = 0; g < groups; ++g) {
      auto& delta = pool_[g];
      auto add = [&](uint64_t key, int64_t d) { delta[shard_of(key)].add(key, d); };
      for (size_t k = cuts[g]; k < cuts[g + 1]; ++k) {
        const InstanceId a = pairs[k].first, b = pairs[k].second;
        if (!st_.alive[a] || !st_.alive[b]) continue;
        if (!st_.valid(a, b, off)) {
          ++rejected[g];
          continue;
        }
        st_.partners(a, allow_zero_, [&](InstanceId x, InstanceId y, int32_t o) {
          add(rule_key(st_.tok[x], st_.tok[y], o), -1);
        });
        st_.partners(b, allow_zero_, [&](InstanceId x, InstanceId y, int32_t o) {
          if (x != a && y != a) add(rule_key(st_.tok[x], st_.tok[y], o), -1);
        });
        const int64_t r = st_.root[a];
        st_.kill(a);
        st_.kill(b);
        const InstanceId m = base + static_cast<InstanceId>(k);
        st_.tok[m] = child;
        st_.root[m] = r;
        st_.place(m);
        born[g].push_back(m);
        st_.partners(m, allow_zero_, [&](InstanceId x, InstanceId y, int32_t o) {
          add(rule_key(st_.tok[x], st_.tok[y], o), +1);
        });
        ++applied[g];
      }
    }
    (groups > 1 ? parallel_seconds_ : serial_seconds_) += seconds_since(loop_start);
    const auto merge_start = std::chrono::steady_clock::now();
    std::vector<int64_t> changes(kShards, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_) if (groups > 1)
#endif
    for (size_t p = 0; p < kShards; ++p) {
      StatMap& table = stats_.shard(p);
      for (size_t g = 0; g < groups; ++g) {
        pool_[g][p].each([&](uint64_t k, int64_t d) {
          change(k, d, table, dirty_shards_[p]);
          ++changes[p];
        });
        pool_[g][p].clear();
      }
    }
    for (size_t p = 0; p < kShards; ++p) {
      dirty_.insert(dirty_.end(), dirty_shards_[p].begin(), dirty_shards_[p].end());
      dirty_shards_[p].clear();
      changes_ += changes[p];
    }
    for (size_t g = 0; g < groups; ++g) {
      auto& occ = occurrence_[child];
      occ.insert(occ.end(), born[g].begin(), born[g].end());
      dead_[left] += applied[g];
      dead_[right] += applied[g];
      applied_ += applied[g];
      rejected_ += rejected[g];
      live_ -= applied[g];
    }
    merge_seconds_ += seconds_since(merge_start);
  }

  // Cut root-ordered pairs into contiguous groups at safe frames.
  std::vector<size_t> split(const std::vector<std::pair<InstanceId, InstanceId>>& pairs) const {
    std::vector<size_t> cuts{0};
    const size_t n = pairs.size();
    const size_t groups = std::max<size_t>(1, std::min<size_t>(threads_, n / kApplyGrain));
    for (size_t g = 1; g < groups; ++g) {
      const size_t target = std::max(cuts.back(), n * g / groups);
      if (target >= n) break;
      auto frame = std::upper_bound(safe_.begin(), safe_.end(), st_.root[pairs[target].first]);
      if (frame == safe_.end()) break;
      const size_t cut = static_cast<size_t>(
          std::lower_bound(pairs.begin() + target, pairs.end(), *frame,
                           [&](const std::pair<InstanceId, InstanceId>& p, int64_t f) {
                             return st_.root[p.first] < f;
                           }) -
          pairs.begin());
      if (cut > cuts.back() && cut < n) cuts.push_back(cut);
    }
    cuts.push_back(n);
    return cuts;
  }

  static bool certified(const Stat& s) { return s.exact >= 0 && s.exact_version == s.version; }

  static int64_t bound(const Stat& s) {
    if (certified(s)) return s.exact;
    return s.exact >= 0 ? std::min(s.raw, s.exact + s.added - s.added_at_exact) : s.raw;
  }

  void queue(uint64_t key, Stat& s, int64_t count) {
    heap_.push({count, key, s.version});
    s.live = true;
    s.queued_version = s.version;
    s.queued_count = count;
  }

  bool take(Entry& e, Stat*& s) {
    e = heap_.top();
    heap_.pop();
    s = stats_.find(e.key);
    if (!s || s->retired || !s->live || e.version != s->queued_version) return false;
    s->live = false;
    if (e.version == s->version) return true;
    const int64_t b = bound(*s);
    if (b > 0) queue(e.key, *s, b);
    return false;
  }

  bool current_certified(const Entry& e) {
    const Stat* s = stats_.find(e.key);
    return s && !s->retired && s->live && e.version == s->queued_version &&
           e.version == s->version && certified(*s);
  }

  void run(bool progress) {
    live_ = initial_instances_;
    while (tables_.size() < vocab_size_ && !heap_.empty()) {
      Entry top;
      Stat* found;
      if (!take(top, found)) continue;
      Stat& s = *found;
      if (!certified(s)) {
        std::vector<Entry> batch{top};
        while (!heap_.empty() && batch.size() < kCertifyBatch) {
          if (current_certified(heap_.top())) break;
          Entry e;
          Stat* next;
          if (take(e, next)) batch.push_back(e);
        }
        certify(batch);
        continue;
      }
      if (top.count != s.exact) continue;
      if (s.exact < min_count_) continue;
      int32_t l, r, o;
      unpack(top.key, l, r, o);
      std::vector<uint32_t> m;
      if (!tables_.compose(l, r, o, m)) {
        s.exact = 0;
        continue;
      }
      const int32_t child = tables_.add_merge(l, r, o);
      merges_.emplace_back(l, r, o);
      s.retired = true;
      dirty_.clear();
      ++epoch_;
      apply(l, r, o, child);
      // Lazy decrease-key: only a grown bound needs a new entry now.
      for (uint64_t k : dirty_) {
        Stat& d = stats_[k];
        if (d.retired) continue;
        const int64_t b = bound(d);
        if (b > 0 && (!d.live || b > d.queued_count)) queue(k, d, b);
      }
      dirty_.clear();
      if (progress && merges_.size() % 256 == 0) {
        py::gil_scoped_acquire acquire;
        py::print("sparse temporal merges", merges_.size(), "live", live_, py::arg("flush") = true);
      }
    }
  }

  Tables tables_;
  Instances st_;
  int64_t vocab_size_;
  bool allow_zero_;
  bool dense_census_;
  int64_t min_count_;
  int64_t max_instances_;
  std::vector<std::vector<InstanceId>> occurrence_;
  ShardedStats stats_;
  std::vector<std::vector<DeltaMap>> pool_;            // [apply group][shard]
  std::vector<std::vector<uint64_t>> dirty_shards_;  // [shard]
  std::priority_queue<Entry> heap_;
  std::vector<uint64_t> dirty_;
  std::vector<std::tuple<int32_t, int32_t, int32_t>> merges_;
  std::vector<int64_t> dead_;
  std::vector<int64_t> safe_;
  uint32_t epoch_ = 0;
  int threads_ = 1;
  int64_t initial_instances_ = 0, live_ = 0, dry_runs_ = 0, rejected_ = 0, applied_ = 0;
  int64_t compactions_ = 0;
  size_t initial_types_ = 0;
  int64_t changes_ = 0;
  double census_seconds_ = 0, dry_seconds_ = 0, apply_seconds_ = 0, collect_seconds_ = 0,
         parallel_seconds_ = 0, serial_seconds_ = 0, merge_seconds_ = 0;
};

struct Spatial {
  std::vector<int64_t> offsets;
  std::vector<uint32_t> support;
  std::vector<std::vector<int32_t>> expansion;  // token -> value per sparse slot (-1 absent)
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> rules;  // (lo, hi) -> (child, rank)
  int32_t dims = 0;

  static uint64_t pair_key(int32_t a, int32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
  }

  void init(const std::vector<int64_t>& bins, const std::vector<std::pair<int32_t, int32_t>>& merges) {
    dims = static_cast<int32_t>(bins.size());
    offsets.assign(1, 0);
    for (int64_t b : bins) {
      if (b < 1) throw std::invalid_argument("every sparse slot needs at least one value");
      offsets.push_back(offsets.back() + b);
    }
    for (int32_t d = 0; d < dims; ++d)
      for (int64_t v = 0; v < bins[d]; ++v) {
        support.push_back(1u << d);
        std::vector<int32_t> e(dims, -1);
        e[d] = static_cast<int32_t>(v);
        expansion.push_back(std::move(e));
      }
    const int32_t atoms = static_cast<int32_t>(support.size());
    for (size_t r = 0; r < merges.size(); ++r) {
      const auto [a, b] = merges[r];
      const int32_t child = atoms + static_cast<int32_t>(r);
      if (a < 0 || b < 0 || a >= child || b >= child || (support[a] & support[b]))
        throw std::invalid_argument("invalid spatial merge");
      support.push_back(support[a] | support[b]);
      std::vector<int32_t> e(dims);
      for (int32_t d = 0; d < dims; ++d) e[d] = std::max(expansion[a][d], expansion[b][d]);
      expansion.push_back(std::move(e));
      rules.emplace(pair_key(a, b), std::make_pair(child, static_cast<int32_t>(r)));
    }
  }

  std::vector<int32_t> encode(const int32_t* row) const {
    std::set<int32_t> tokens;
    for (int32_t d = 0; d < dims; ++d)
      if (row[d] >= 0) tokens.insert(static_cast<int32_t>(offsets[d] + row[d]));
    using Item = std::tuple<int32_t, int32_t, int32_t, int32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    auto offer = [&](int32_t a, int32_t b) {
      auto it = rules.find(pair_key(a, b));
      if (it != rules.end())
        heap.emplace(it->second.second, it->second.first, std::min(a, b), std::max(a, b));
    };
    for (auto i = tokens.begin(); i != tokens.end(); ++i)
      for (auto j = std::next(i); j != tokens.end(); ++j) offer(*i, *j);
    while (!heap.empty()) {
      const auto [rank, child, left, right] = heap.top();
      heap.pop();
      if (!tokens.count(left) || !tokens.count(right)) continue;
      tokens.erase(left);
      tokens.erase(right);
      for (int32_t other : tokens) offer(child, other);
      tokens.insert(child);
    }
    return {tokens.begin(), tokens.end()};
  }
};

// Runtime codec over dense primitive cells ``(T, D)`` with ``-1`` for inactive slots.
class SparseCodec {
 public:
  SparseCodec(std::vector<int64_t> bins, std::vector<int64_t> zero,
              std::vector<std::pair<int32_t, int32_t>> spatial_merges, int32_t lookahead,
              std::vector<std::tuple<int32_t, int32_t, int32_t>> temporal_merges)
      : bins_(std::move(bins)), zero_(std::move(zero)) {
    if (bins_.empty() || bins_.size() != zero_.size())
      throw std::invalid_argument("bins and zero must have one entry per slot");
    dims_ = static_cast<int32_t>(bins_.size());
    std::vector<int64_t> sparse;
    for (int32_t d = 0; d < dims_; ++d) {
      // -2 retains every dense value; -1 change-codes an absolute slot.
      if (zero_[d] >= bins_[d] || zero_[d] < -2) throw std::invalid_argument("zero index");
      sparse.push_back(bins_[d] - (zero_[d] >= 0));
    }
    sparse.push_back(1);  // E
    spatial_.init(sparse, spatial_merges);
    tables_.init(spatial_.support, dims_ + 1, lookahead);
    for (size_t r = 0; r < temporal_merges.size(); ++r) {
      auto [a, b, o] = temporal_merges[r];
      if (a < 0 || b < 0 || a >= tables_.size() || b >= tables_.size())
        throw std::invalid_argument("merge order");
      tables_.add_merge(a, b, o);
      if (!rank_.emplace(rule_key(a, b, o), static_cast<int32_t>(r)).second)
        throw std::invalid_argument("duplicate temporal rule");
    }
  }

  int32_t vocab_size() const { return tables_.size(); }
  int32_t spatial_vocab_size() const { return static_cast<int32_t>(spatial_.support.size()); }

  std::vector<std::vector<int32_t>> encode_batch(
      const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& cells,
      int threads) const {
    if (cells.ndim() != 3 || cells.shape(2) != dims_ || cells.shape(0) < 1 || cells.shape(1) < 1)
      throw std::invalid_argument("cells must have shape (B, T, D) with B, T >= 1");
    const int64_t batch = cells.shape(0), steps = cells.shape(1);
    const int32_t* data = cells.data();
    std::vector<std::vector<int32_t>> out(batch);
    {
      py::gil_scoped_release release;
      ac2::parallel_for(batch, threads, [&](int64_t b) {
        out[b] = encode_one(data + b * steps * dims_, steps);
      });
    }
    return out;
  }

  py::array_t<int32_t> decode_batch(const std::vector<std::vector<int32_t>>& rows,
                                    int64_t horizon, const std::vector<bool>& active) const {
    if (horizon < 1) throw std::invalid_argument("horizon must be positive");
    if (static_cast<int32_t>(active.size()) != dims_)
      throw std::invalid_argument("active must have one flag per slot");
    py::array_t<int32_t> out({static_cast<int64_t>(rows.size()), horizon, int64_t{dims_}});
    int32_t* data = out.mutable_data();
    for (size_t k = 0; k < rows.size(); ++k) decode_one(rows[k], horizon, active, data + k * horizon * dims_);
    return out;
  }

  // Spatial tokens of sparse frames ``(N, D + 1)``: per-frame counts and flat tokens.
  py::tuple spatial_encode(
      const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& frames) const {
    if (frames.ndim() != 2 || frames.shape(1) != dims_ + 1)
      throw std::invalid_argument("frames must have shape (N, D + 1)");
    const int64_t n = frames.shape(0);
    py::array_t<int64_t> counts(n);
    std::vector<int32_t> flat;
    for (int64_t i = 0; i < n; ++i) {
      const int32_t* row = frames.data() + i * (dims_ + 1);
      for (int32_t d = 0; d <= dims_; ++d)
        if (row[d] >= static_cast<int32_t>(spatial_.offsets[d + 1] - spatial_.offsets[d]))
          throw std::invalid_argument("sparse value out of range");
      const auto tokens = spatial_.encode(row);
      counts.mutable_data()[i] = static_cast<int64_t>(tokens.size());
      flat.insert(flat.end(), tokens.begin(), tokens.end());
    }
    return py::make_tuple(counts, py::array_t<int32_t>(flat.size(), flat.data()));
  }

  std::vector<std::vector<std::pair<int32_t, int32_t>>> patterns() const { return tables_.pattern; }

 private:
  // Dense chunk -> sparse frames (T, D + 1); validates slot activity and ranges.
  std::vector<int32_t> sparsify(const int32_t* cells, int64_t steps) const {
    const int32_t width = dims_ + 1;
    std::vector<int32_t> sparse(static_cast<size_t>(steps) * width, -1);
    int64_t last = -1;
    for (int32_t d = 0; d < dims_; ++d) {
      const bool active = cells[d] >= 0;
      for (int64_t t = 0; t < steps; ++t) {
        const int32_t v = cells[t * dims_ + d];
        if ((v >= 0) != active)
          throw std::invalid_argument("a slot must be active for the whole chunk or not at all");
        if (v >= bins_[d]) throw std::invalid_argument("cell outside the primitive range");
        if (!active) continue;
        int32_t s = -1;
        if (zero_[d] == -2) {
          s = v;
        } else if (zero_[d] < 0) {
          if (t == 0 || v != cells[(t - 1) * dims_ + d]) s = v;
        } else if (v != zero_[d]) {
          s = v - (v > zero_[d]);
        }
        sparse[t * width + d] = s;
        if (s >= 0) last = std::max(last, t);
      }
    }
    for (int64_t t = 0; t < last; ++t) sparse[t * width + dims_] = 0;
    return sparse;
  }

  std::vector<int32_t> encode_one(const int32_t* cells, int64_t steps) const {
    const std::vector<int32_t> sparse = sparsify(cells, steps);
    Instances s;
    s.reset(&tables_, steps + tables_.lookahead);
    for (int64_t t = 0; t < steps; ++t)
      for (int32_t token : spatial_.encode(sparse.data() + t * (dims_ + 1))) s.add(token, t);
    using Cand = std::tuple<int32_t, int64_t, int32_t, int64_t, InstanceId, InstanceId, int32_t>;
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> heap;
    auto offer = [&](InstanceId i) {
      s.partners(i, true, [&](InstanceId a, InstanceId b, int32_t off) {
        auto it = rank_.find(rule_key(s.tok[a], s.tok[b], off));
        if (it == rank_.end()) return;
        heap.emplace(it->second, s.root[a], tables_.rdim[s.tok[a]], s.root[b], a, b, off);
      });
    };
    for (size_t i = 0; i < s.tok.size(); ++i) offer(static_cast<InstanceId>(i));
    while (!heap.empty()) {
      auto [r, ra, da, rb, a, b, off] = heap.top();
      heap.pop();
      if (!s.alive[a] || !s.alive[b] || !s.valid(a, b, off)) continue;
      const int64_t root = s.root[a];
      s.kill(a);
      s.kill(b);
      offer(s.add(tables_.base + r, root));
    }
    return s.serialize();
  }

  void decode_one(const std::vector<int32_t>& row, int64_t horizon,
                  const std::vector<bool>& active, int32_t* cells) const {
    const int32_t L = tables_.lookahead, width = dims_ + 1;
    uint32_t allowed = 1u << dims_;
    for (int32_t d = 0; d < dims_; ++d)
      if (active[d]) allowed |= 1u << d;
    std::vector<uint32_t> cover(horizon + L, 0);
    std::vector<int32_t> sparse(static_cast<size_t>(horizon) * width, -1);
    int64_t front = 0;
    std::vector<uint32_t> own(L, 0);
    std::vector<int32_t> frames;
    for (int32_t token : row) {
      if (token < 0 || token >= tables_.size()) throw std::invalid_argument("token outside vocab");
      if (front >= horizon) throw std::invalid_argument("token after the last open frame");
      frames.clear();
      for (auto [o, t] : tables_.pattern[token]) {
        const int64_t u = front + o;
        const uint32_t m = spatial_.support[t];
        if (u >= horizon) throw std::invalid_argument("token exceeds horizon");
        if (m & ~allowed) throw std::invalid_argument("token touches an inactive slot");
        if (m & own[o]) throw std::invalid_argument("overlapping token supports");
        if ((m >> dims_ & 1u) && u == horizon - 1)
          throw std::invalid_argument("end-of-frame atom on the last frame");
        if (!own[o]) frames.push_back(o);
        own[o] |= m;
        for (int32_t d = 0; d < width; ++d)
          if (spatial_.expansion[t][d] >= 0) sparse[u * width + d] = spatial_.expansion[t][d];
      }
      for (int32_t o : frames) {
        const int64_t u = front + o;
        if (cover[u] & own[o]) throw std::invalid_argument("overlapping token supports");
        if (cover[u] >> dims_ & 1u) throw std::invalid_argument("token touches a closed frame");
        cover[u] |= own[o];
        own[o] = 0;
      }
      while (front < horizon && (cover[front] >> dims_ & 1u)) ++front;
    }
    for (int32_t d = 0; d < dims_; ++d) {
      int32_t carried = -1;
      for (int64_t t = 0; t < horizon; ++t) {
        const int32_t v = sparse[t * width + d];
        int32_t& out = cells[t * dims_ + d];
        if (!active[d]) {
          out = -1;
        } else if (zero_[d] == -2) {
          if (v < 0) throw std::invalid_argument("dense active slot is missing");
          out = v;
        } else if (zero_[d] < 0) {
          if (v >= 0) carried = v;
          if (carried < 0) throw std::invalid_argument("an absolute slot must be stated on frame 0");
          out = carried;
        } else {
          out = v < 0 ? static_cast<int32_t>(zero_[d]) : v + (v >= zero_[d]);
        }
      }
    }
  }

  std::vector<int64_t> bins_, zero_;
  int32_t dims_ = 0;
  Spatial spatial_;
  Tables tables_;
  std::unordered_map<uint64_t, int32_t> rank_;
};

}  // namespace

void register_sparse(py::module_& m) {
  py::class_<Trainer>(m, "SparseTrainer")
      .def(py::init<std::vector<uint32_t>, int32_t, int32_t, int64_t, bool, int64_t, int, bool, int64_t>(),
           py::arg("spatial_support"), py::arg("dims"), py::arg("lookahead"),
           py::arg("vocab_size"), py::arg("allow_zero_offset"), py::arg("min_count") = 2,
           py::arg("threads") = 0, py::arg("dense_census") = false,
           py::arg("max_instances") = UINT32_MAX)
      .def("set_instances", &Trainer::set_instances)
      .def("train", &Trainer::train, py::arg("progress") = false)
      .def("merges", &Trainer::merges);
  py::class_<SparseCodec>(m, "SparseCodec")
      .def(py::init<std::vector<int64_t>, std::vector<int64_t>,
                    std::vector<std::pair<int32_t, int32_t>>, int32_t,
                    std::vector<std::tuple<int32_t, int32_t, int32_t>>>(),
           py::arg("bins"), py::arg("zero"), py::arg("spatial_merges"), py::arg("lookahead"),
           py::arg("temporal_merges"))
      .def_property_readonly("vocab_size", &SparseCodec::vocab_size)
      .def_property_readonly("spatial_vocab_size", &SparseCodec::spatial_vocab_size)
      .def("encode_batch", &SparseCodec::encode_batch, py::arg("cells"), py::arg("threads") = 0)
      .def("decode_batch", &SparseCodec::decode_batch, py::arg("rows"), py::arg("horizon"),
           py::arg("active"))
      .def("spatial_encode", &SparseCodec::spatial_encode, py::arg("frames"))
      .def("patterns", &SparseCodec::patterns);
}
