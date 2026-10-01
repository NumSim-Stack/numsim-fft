#ifndef NUMSIM_FFT_EXECUTION_OPENMP_H
#define NUMSIM_FFT_EXECUTION_OPENMP_H

#if !defined(NUMSIM_FFT_HAS_OPENMP)
#error "numsim-fft: OpenMP executor requested but NUMSIM_FFT_ENABLE_OPENMP is off"
#endif

#include <omp.h>

#include <cstddef>
#include <exception>
#include <mutex>

namespace numsim::fft {

/**
 * @brief Runs work items on an OpenMP thread team (dynamic schedule).
 *
 * num_threads == 0 uses omp_get_max_threads(). The first exception thrown by
 * a work item is rethrown on the calling thread after the loop has finished;
 * the remaining items still run.
 */
struct openmp_executor {
  int num_threads{0};

  std::size_t concurrency() const noexcept {
    return static_cast<std::size_t>(num_threads > 0 ? num_threads : omp_get_max_threads());
  }

  template <typename F> void bulk(std::size_t n, F &&f) const {
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto const count{static_cast<std::ptrdiff_t>(n)};
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(concurrency()))
    for (std::ptrdiff_t i = 0; i < count; ++i) {
      try {
        f(static_cast<std::size_t>(i));
      } catch (...) {
        std::lock_guard<std::mutex> lock{failure_mutex};
        if (!failure)
          failure = std::current_exception();
      }
    }
    if (failure)
      std::rethrow_exception(failure);
  }
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_EXECUTION_OPENMP_H
