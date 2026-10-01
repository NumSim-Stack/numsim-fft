#ifndef NUMSIM_FFT_EXECUTION_SEQUENTIAL_H
#define NUMSIM_FFT_EXECUTION_SEQUENTIAL_H

#include <cstddef>

namespace numsim::fft {

/// Runs all work items in order on the calling thread; the reference executor.
struct sequential_executor {
  constexpr std::size_t concurrency() const noexcept { return 1; }

  template <typename F> void bulk(std::size_t n, F &&f) const {
    for (std::size_t i{0}; i < n; ++i)
      f(i);
  }
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_EXECUTION_SEQUENTIAL_H
