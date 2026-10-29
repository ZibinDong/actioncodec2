#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace ac2 {

// Process-wide cap on threads per call; 0 means every CPU the process may use.
inline std::atomic<int>& thread_budget() {
  static std::atomic<int> value{0};
  return value;
}

inline int available_cpus() {
#ifdef __linux__
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
    return std::max(1, CPU_COUNT(&allowed));
  }
#endif
  return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

// Measured cost of starting and joining one thread (seconds).
constexpr double kThreadStartSeconds = 30e-6;

// Threads for ``count`` iterations of ``seconds_each``: at most ``requested``
// when positive, at most the process budget, and at most ``sqrt(W / s)``.
inline int64_t plan_threads(int64_t count, int requested, double seconds_each) {
  const int budget = thread_budget().load();
  int64_t threads = budget > 0 ? budget : available_cpus();
  if (requested > 0) threads = std::min<int64_t>(threads, requested);
  const double by_work = std::sqrt(count * seconds_each / kThreadStartSeconds);
  if (by_work < static_cast<double>(threads)) threads = static_cast<int64_t>(by_work);
  return std::max<int64_t>(1, std::min(threads, count));
}

template <class Body>
void parallel_for(int64_t count, int threads, Body&& body) {
  if (count <= 0) return;
  const auto start = std::chrono::steady_clock::now();
  body(0);
  const double seconds_each =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const int64_t rest = count - 1;
  if (rest == 0) return;
  const int64_t workers = plan_threads(rest, threads, seconds_each);
  auto run = [&](int64_t worker) {
    const int64_t begin = 1 + rest * worker / workers;
    const int64_t end = 1 + rest * (worker + 1) / workers;
    for (int64_t index = begin; index < end; ++index) body(index);
  };
  if (workers == 1) {
    run(0);
    return;
  }
  std::exception_ptr failure;
  std::mutex failure_lock;
  auto guarded = [&](int64_t worker) {
    try {
      run(worker);
    } catch (...) {
      std::lock_guard<std::mutex> hold(failure_lock);
      if (!failure) failure = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers - 1));
  try {
    for (int64_t worker = 1; worker < workers; ++worker) pool.emplace_back(guarded, worker);
  } catch (...) {
    for (auto& thread : pool) thread.join();
    throw;
  }
  guarded(0);
  for (auto& thread : pool) thread.join();
  if (failure) std::rethrow_exception(failure);
}

}  // namespace ac2
