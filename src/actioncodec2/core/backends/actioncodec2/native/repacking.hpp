// Optional lossless fixed-vocabulary packing for the native Set-BPE encoder.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ac2pack {
using Instance = std::tuple<int64_t, int64_t, int64_t>;
using Cover = std::array<uint64_t, 4>;

struct CoverHash {
  size_t operator()(const Cover& value) const {
    size_t hash = 0;
    for (uint64_t word : value) hash ^= std::hash<uint64_t>{}(word) +
        0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct Placement {
  Cover cells{};
  int64_t token;
  int anchor, count;
};

struct Requirement { int offset; uint64_t mask, key; };
using MatchCache = std::array<std::unordered_map<uint64_t, std::vector<int64_t>>, 8>;

inline int first_free(const Cover& cover) {
  for (int word = 0; word < 4; ++word)
    if (~cover[word]) return word * 64 + __builtin_ctzll(~cover[word]);
  return 256;
}

inline int free_count(const Cover& cover) {
  int count = 0;
  for (uint64_t word : cover) count += __builtin_popcountll(~word);
  return count;
}

inline bool overlaps(const Cover& a, const Cover& b) {
  for (int word = 0; word < 4; ++word) if (a[word] & b[word]) return true;
  return false;
}

inline Cover unite(Cover a, const Cover& b) {
  for (int word = 0; word < 4; ++word) a[word] |= b[word];
  return a;
}

class Search {
 public:
  Search(const std::vector<int64_t>& bins, const std::vector<uint64_t>& support,
         const std::vector<int64_t>& expansion,
         const std::vector<std::vector<std::pair<int64_t, int64_t>>>& patterns,
         const std::vector<int64_t>& roots)
      : bins_(bins), width_(bins.size()), spatial_size_(support.size()),
        support_(support), patterns_(patterns) {
    if (width_ > 8) throw std::invalid_argument("packing search supports at most eight slots");
    for (int64_t count : bins)
      if (count > 255) throw std::invalid_argument("packing search supports at most 255 bins per slot");
    keys_.resize(spatial_size_);
    anchors_.resize(patterns.size());
    spans_.resize(patterns.size(), 1);
    counts_.resize(patterns.size());
    for (int64_t id = 0; id < spatial_size_; ++id) {
      for (int d = 0; d < width_; ++d) {
        const int64_t value = expansion[id * width_ + d];
        if (value >= 0) keys_[id] |= uint64_t(value + 1) << (8 * d);
      }
      lookup_.emplace(keys_[id], id);
    }
    temporal_by_root_.resize(spatial_size_);
    for (size_t id = 0; id < patterns.size(); ++id) {
      anchors_[id] = __builtin_ctzll(support[roots[id]]);
      int count = 0;
      for (auto [time, sid] : patterns[id]) {
        count += __builtin_popcountll(support[sid]);
        spans_[id] = std::max(spans_[id], int(time + 1));
      }
      max_cells_ = std::max(max_cells_, count);
      counts_[id] = count;
      if (id >= static_cast<size_t>(spatial_size_))
        temporal_by_root_[lookup_.at(keys_[roots[id]])].push_back(id);
    }
    byte_masks_.resize(1 << width_);
    for (int mask = 1; mask < (1 << width_); ++mask) {
      const int d = __builtin_ctz(mask);
      byte_masks_[mask] = byte_masks_[mask ^ (1 << d)] | (uint64_t{255} << (8 * d));
    }
    prepare_spatial_trie();
    prepare_patterns();
  }

  std::vector<int64_t> apply(const int64_t* cells, int64_t steps,
      const std::vector<int64_t>& original, bool spatial, bool temporal, int window, int beam) const {
    if (!temporal) return spatial ? repack(cells, steps, original) : original;
    auto result = local_repack(cells, steps, original, window, beam);
    return spatial ? repack(cells, steps, result) : result;
  }

 private:
  struct TrieNode { std::vector<int32_t> children; int64_t token = -1; };

  int32_t trie_node(int dimension) {
    const auto index = static_cast<int32_t>(trie_.size());
    trie_.emplace_back();
    if (dimension < width_) trie_.back().children.assign(bins_[dimension] + 1, -1);
    return index;
  }

  void prepare_spatial_trie() {
    for (int d = 0; d < width_; ++d) trie_roots_[d].assign(bins_[d] + 1, -1);
    for (int64_t sid = 0; sid < spatial_size_; ++sid) {
      if (lookup_.at(keys_[sid]) != sid) continue;
      const int root = anchors_[sid], value = (keys_[sid] >> (8 * root)) & 255;
      int32_t node = trie_roots_[root][value];
      if (node < 0) {
        node = trie_node(root + 1);
        trie_roots_[root][value] = node;
      }
      for (int d = root + 1; d < width_; ++d) {
        const int cell = (keys_[sid] >> (8 * d)) & 255;
        int32_t child = trie_[node].children[cell];
        if (child < 0) {
          child = trie_node(d + 1);
          trie_[node].children[cell] = child;
        }
        node = child;
      }
      trie_[node].token = sid;
    }
  }

  void query_spatial_trie(int32_t node, int dimension, uint64_t row,
                          std::vector<int64_t>& result) const {
    if (node < 0) return;
    if (dimension == width_) { result.push_back(trie_[node].token); return; }
    query_spatial_trie(trie_[node].children[0], dimension + 1, row, result);
    const int value = (row >> (8 * dimension)) & 255;
    if (value) query_spatial_trie(trie_[node].children[value], dimension + 1, row, result);
  }

  void prepare_patterns() {
    relative_.resize(patterns_.size());
    requirements_.resize(patterns_.size());
    for (size_t token = 0; token < patterns_.size(); ++token) {
      if (spans_[token] * width_ > 256) continue;
      std::vector<Requirement> rows(spans_[token], Requirement{0, 0, 0});
      for (auto [offset, sid] : patterns_[token]) {
        rows[offset].offset = offset;
        rows[offset].mask |= byte_masks_[support_[sid]];
        rows[offset].key |= keys_[sid];
        uint64_t support = support_[sid];
        while (support) {
          const int d = __builtin_ctzll(support); support &= support - 1;
          const int bit = offset * width_ + d;
          relative_[token][bit / 64] |= uint64_t{1} << (bit % 64);
        }
      }
      for (const auto& row : rows) if (row.mask) requirements_[token].push_back(row);
    }
  }

  uint64_t row_key(const int64_t* row) const {
    uint64_t key = 0;
    for (int d = 0; d < width_; ++d)
      if (row[d] >= 0) key |= uint64_t(row[d] + 1) << (8 * d);
    return key;
  }

  std::vector<int64_t> matching_spatial(uint64_t row, int dim) const {
    std::vector<int64_t> result;
    const int value = (row >> (8 * dim)) & 255;
    if (value) query_spatial_trie(trie_roots_[dim][value], dim + 1, row, result);
    return result;
  }

  bool placement(int64_t token, int anchor, const std::vector<uint64_t>& rows,
                 const Cover& fixed, Placement& out) const {
    if (anchor + spans_[token] > static_cast<int>(rows.size())) return false;
    for (const auto& req : requirements_[token])
      if ((rows[anchor + req.offset] & req.mask) != req.key) return false;
    out = Placement{}; out.token = token; out.anchor = anchor; out.count = counts_[token];
    const int start = anchor * width_, word = start / 64, shift = start % 64;
    const int words = (spans_[token] * width_ + 63) / 64;
    for (int i = 0; i < words && word + i < 4; ++i) {
      out.cells[word + i] |= relative_[token][i] << shift;
      if (shift && word + i + 1 < 4)
        out.cells[word + i + 1] |= relative_[token][i] >> (64 - shift);
    }
    return !overlaps(out.cells, fixed);
  }

  std::vector<Placement> candidates(
      int root, const std::vector<uint64_t>& rows, const Cover& fixed,
      MatchCache& cache) const {
    std::vector<Placement> result;
    const int time = root / width_, dim = root % width_;
    auto& table = cache[dim];
    auto found = table.find(rows[time]);
    if (found == table.end())
      found = table.emplace(rows[time], matching_spatial(rows[time], dim)).first;
    for (int64_t sid : found->second) {
      Placement offer;
      if (placement(sid, time, rows, fixed, offer)) result.push_back(offer);
      for (int64_t token : temporal_by_root_[sid])
        if (placement(token, time, rows, fixed, offer)) result.push_back(offer);
    }
    std::sort(result.begin(), result.end(), [](const Placement& a, const Placement& b) {
      return std::make_pair(-a.count, a.token) < std::make_pair(-b.count, b.token);
    });
    return result;
  }

  using Options = std::function<const std::vector<Placement>&(int)>;

  std::vector<Instance> greedy_cover(const Options& options, Cover occupied, int bound) const {
    std::vector<Instance> result;
    while (first_free(occupied) != 256 && static_cast<int>(result.size()) + 1 < bound) {
      const auto& offers = options(first_free(occupied));
      auto choice = std::find_if(offers.begin(), offers.end(), [&](const Placement& offer) {
        return !overlaps(occupied, offer.cells);
      });
      if (choice == offers.end()) return {};
      result.emplace_back(choice->anchor, anchors_[choice->token], choice->token);
      occupied = unite(occupied, choice->cells);
    }
    return first_free(occupied) == 256 ? result : std::vector<Instance>{};
  }

  std::vector<Instance> beam_cover(const Options& options,
                                   const Cover& fixed, int bound, int beam) const {
    if (beam == 1) return greedy_cover(options, fixed, bound);
    struct Node { Cover covered; int parent; int64_t token; int anchor, remaining, root; };
    std::vector<Node> arena{{fixed, -1, -1, -1, free_count(fixed), first_free(fixed)}};
    arena.reserve(beam * bound + 1);
    std::vector<int> frontier{0};
    frontier.reserve(beam);
    std::array<Node, 64> next;
    auto better = [](const Node& a, const Node& b) {
      if (a.remaining != b.remaining) return a.remaining < b.remaining;
      return a.root != b.root ? a.root > b.root : a.covered < b.covered;
    };
    for (int depth = 0; depth + 1 < bound && !frontier.empty(); ++depth) {
      int keep = 0;
      for (int index : frontier) {
        for (const auto& offer : options(arena[index].root)) {
          if (overlaps(arena[index].covered, offer.cells)) continue;
          Cover covered = unite(arena[index].covered, offer.cells);
          const int remaining = arena[index].remaining - offer.count;
          if (!remaining) {
            std::vector<Instance> result{{offer.anchor, anchors_[offer.token], offer.token}};
            for (int p = index; arena[p].parent >= 0; p = arena[p].parent)
              result.emplace_back(arena[p].anchor, anchors_[arena[p].token], arena[p].token);
            std::reverse(result.begin(), result.end());
            return result;
          }
          if (keep == beam && remaining > next[keep - 1].remaining) continue;
          Node node{covered, index, offer.token, offer.anchor, remaining, first_free(covered)};
          if (keep == beam && !better(node, next[keep - 1])) continue;
          bool duplicate = false;
          for (int i = 0; i < keep; ++i) if (covered == next[i].covered) { duplicate = true; break; }
          if (duplicate) continue;
          int position = keep < beam ? keep++ : keep - 1;
          while (position && better(node, next[position - 1])) {
            next[position] = next[position - 1]; --position;
          }
          next[position] = node;
        }
      }
      frontier.clear();
      for (int i = 0; i < keep; ++i) {
        frontier.push_back(arena.size());
        arena.push_back(std::move(next[i]));
      }
    }
    return {};
  }

  std::vector<int64_t> local_repack(const int64_t* cells, int64_t steps,
                                   const std::vector<int64_t>& stream, int window, int beam) const {
    uint64_t full = 0;
    for (int d = 0; d < width_; ++d) if (cells[d] >= 0) full |= uint64_t{1} << d;
    std::vector<uint64_t> coverage(steps, 0), fixed(steps, 0);
    std::vector<std::vector<Instance>> blocks((steps + window - 1) / window);
    std::vector<Instance> result;
    int64_t front = 0;
    for (int64_t token : stream) {
      while (front < steps && coverage[front] == full) ++front;
      const bool crossing = front / window != (front + spans_[token] - 1) / window;
      Instance instance{front, anchors_[token], token};
      if (crossing) result.push_back(instance);
      else blocks[front / window].push_back(instance);
      for (auto [offset, sid] : patterns_[token]) {
        coverage[front + offset] |= support_[sid];
        if (crossing) fixed[front + offset] |= support_[sid];
      }
    }
    MatchCache cache;
    for (size_t block = 0; block < blocks.size(); ++block) {
      const int start = block * window, length = std::min<int64_t>(window, steps - start);
      Cover occupied; occupied.fill(~uint64_t{0});
      std::vector<uint64_t> rows(length);
      for (int t = 0; t < length; ++t) {
        rows[t] = row_key(cells + (start + t) * width_);
        for (int d = 0; d < width_; ++d) {
          const int bit = t * width_ + d;
          if ((full & (uint64_t{1} << d)) && !(fixed[start + t] & (uint64_t{1} << d)))
            occupied[bit / 64] &= ~(uint64_t{1} << (bit % 64));
        }
      }
      std::vector<Instance> improved;
      const int bound = blocks[block].size();
      if ((free_count(occupied) + max_cells_ - 1) / max_cells_ < bound) {
        std::vector<std::vector<Placement>> options(length * width_);
        std::vector<char> ready(length * width_, 0);
        auto get_options = [&](int root) -> const std::vector<Placement>& {
          if (!ready[root]) {
            options[root] = candidates(root, rows, occupied, cache);
            ready[root] = 1;
          }
          return options[root];
        };
        improved = beam_cover(get_options, occupied, bound, beam);
      }
      if (improved.empty()) result.insert(result.end(), blocks[block].begin(), blocks[block].end());
      else for (auto [time, root, token] : improved) result.emplace_back(start + time, root, token);
      for (auto& table : cache) if (table.size() > 4096) table.clear();
    }
    std::sort(result.begin(), result.end());
    std::vector<int64_t> tokens;
    for (auto [time, root, token] : result) tokens.push_back(token);
    if (tokens.size() > stream.size()) throw std::runtime_error("local packing regressed");
    return tokens;
  }

  std::vector<int64_t> optimal(uint64_t row, int mask,
                               const std::vector<int64_t>& incumbent) const {
    if (incumbent.size() < 2) return incumbent;
    std::array<int64_t, 256> matched;
    matched.fill(-2);
    auto match = [&](int subset) {
      int64_t& token = matched[subset];
      if (token == -2) {
        const auto found = lookup_.find(row & byte_masks_[subset]);
        token = found == lookup_.end() ? -1 : found->second;
      }
      return token;
    };
    if (int64_t token = match(mask); token >= 0) return {token};
    if (incumbent.size() == 2) return incumbent;
    const int bit = mask & -mask;
    for (int sub = (mask - 1) & mask; sub; sub = (sub - 1) & mask) {
      if (!(sub & bit)) continue;
      const int64_t left = match(sub), right = match(mask ^ sub);
      if (left >= 0 && right >= 0) return {left, right};
    }
    if (incumbent.size() == 3) return incumbent;
    std::array<int, 256> costs, choices;
    costs.fill(99); choices.fill(0); costs[0] = 0;
    std::function<int(int)> solve = [&](int remaining) {
      if (costs[remaining] != 99) return costs[remaining];
      int best = 98;
      const int first = remaining & -remaining;
      for (int sub = remaining; sub; sub = (sub - 1) & remaining) {
        if (!(sub & first) || match(sub) < 0) continue;
        const int cost = 1 + solve(remaining ^ sub);
        if (cost < best) { best = cost; choices[remaining] = sub; }
      }
      return costs[remaining] = best;
    };
    if (solve(mask) >= static_cast<int>(incumbent.size())) return incumbent;
    std::vector<int64_t> result;
    for (int remaining = mask; remaining; remaining ^= choices[remaining])
      result.push_back(match(choices[remaining]));
    return result;
  }

  std::vector<int64_t> repack(const int64_t* cells, int64_t steps,
                            const std::vector<int64_t>& stream) const {
    uint64_t full = 0;
    for (int d = 0; d < width_; ++d) if (cells[d] >= 0) full |= uint64_t{1} << d;
    std::vector<uint64_t> covered(steps, 0), fixed(steps, 0);
    std::vector<std::vector<int64_t>> old_spatial(steps), temporal(steps);
    int64_t front = 0;
    for (int64_t token : stream) {
      while (front < steps && covered[front] == full) ++front;
      if (front == steps || token < 0 || token >= static_cast<int64_t>(patterns_.size()) ||
          anchors_[token] != __builtin_ctzll(full & ~covered[front]))
        throw std::invalid_argument("invalid original token stream");
      const bool frozen = token >= spatial_size_;
      (frozen ? temporal[front] : old_spatial[front]).push_back(token);
      for (auto [offset, id] : patterns_[token]) {
        if (front + offset >= steps || (covered[front + offset] & support_[id]))
          throw std::invalid_argument("invalid original token coverage");
        covered[front + offset] |= support_[id];
        if (frozen) fixed[front + offset] |= support_[id];
      }
    }
    std::vector<int64_t> result;
    result.reserve(stream.size());
    for (int64_t time = 0; time < steps; ++time) {
      if (covered[time] != full) throw std::invalid_argument("incomplete original stream");
      auto tokens = optimal(row_key(cells + time * width_), full & ~fixed[time], old_spatial[time]);
      tokens.insert(tokens.end(), temporal[time].begin(), temporal[time].end());
      std::sort(tokens.begin(), tokens.end(), [&](int64_t a, int64_t b) {
        return std::make_pair(anchors_[a], a) < std::make_pair(anchors_[b], b);
      });
      result.insert(result.end(), tokens.begin(), tokens.end());
    }
    if (result.size() > stream.size()) throw std::runtime_error("packing regressed");
    return result;
  }

  const std::vector<int64_t>& bins_;
  int width_;
  int64_t spatial_size_;
  const std::vector<uint64_t>& support_;
  const std::vector<std::vector<std::pair<int64_t, int64_t>>>& patterns_;
  std::vector<uint64_t> keys_, byte_masks_;
  std::vector<int> anchors_, spans_, counts_;
  std::vector<Cover> relative_;
  std::vector<std::vector<Requirement>> requirements_;
  std::vector<std::vector<int64_t>> temporal_by_root_;
  std::vector<TrieNode> trie_;
  std::array<std::vector<int32_t>, 8> trie_roots_;
  int max_cells_ = 1;
  std::unordered_map<uint64_t, int64_t> lookup_;
};
}  // namespace ac2pack
