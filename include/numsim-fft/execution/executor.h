#ifndef NUMSIM_FFT_EXECUTION_EXECUTOR_H
#define NUMSIM_FFT_EXECUTION_EXECUTOR_H

#include <concepts>
#include <cstddef>

namespace numsim::fft {

/**
 * @brief Runs independent work items, possibly in parallel.
 *
 * - concurrency(): number of workers the executor may use at once. The plans
 *   split each pass into about that many chunks, and every chunk owns its
 *   scratch buffer.
 * - bulk(n, f): calls f(i) once for every i in [0, n) and returns when all
 *   calls have finished. Calls may run concurrently and in any order.
 */
template <typename E>
concept executor = requires(E const &e, std::size_t n, void (*f)(std::size_t)) {
  { e.concurrency() } -> std::convertible_to<std::size_t>;
  e.bulk(n, f);
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_EXECUTION_EXECUTOR_H
