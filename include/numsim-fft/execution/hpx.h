#ifndef NUMSIM_FFT_EXECUTION_HPX_H
#define NUMSIM_FFT_EXECUTION_HPX_H

#if !defined(NUMSIM_FFT_HAS_HPX)
#error "numsim-fft: HPX executor requested but NUMSIM_FFT_ENABLE_HPX is off"
#endif

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/runtime.hpp>

#include <atomic>
#include <cstddef>
#include <exception>

namespace numsim::fft {

/**
 * @brief Runs work items as HPX tasks (hpx::experimental::for_loop with the
 * parallel policy) on the running HPX runtime.
 *
 * The HPX runtime must be active: call it from hpx_main / inside
 * hpx::local::init (or after hpx::local::start). The first exception thrown
 * by a work item is rethrown as-is, not wrapped in hpx::exception_list.
 */
struct hpx_executor {
  std::size_t concurrency() const noexcept {
    return static_cast<std::size_t>(hpx::get_num_worker_threads());
  }

  template <typename F> void bulk(std::size_t n, F &&f) const {
    std::exception_ptr failure;
    std::atomic_flag failed;
    hpx::experimental::for_loop(hpx::execution::par, std::size_t{0}, n, [&](std::size_t i) {
      try {
        f(i);
      } catch (...) {
        if (!failed.test_and_set())
          failure = std::current_exception();
      }
    });
    if (failure)
      std::rethrow_exception(failure);
  }
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_EXECUTION_HPX_H
