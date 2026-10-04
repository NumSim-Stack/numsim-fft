#ifndef NUMSIM_FFT_EXECUTION_CHUNKING_H
#define NUMSIM_FFT_EXECUTION_CHUNKING_H

#include <algorithm>
#include <cstddef>

namespace numsim::fft::detail {

/// Calls f(first, last) for the chunks of [0, n): about 8 per worker (one
/// for a single worker). The field algebra and the first touch of new
/// fields share this split, so a chunk's pages are first written by the
/// same kind of work item that later streams through them.
template <typename Exec, typename F> void chunked_for(std::size_t n, Exec const &exec, F const &f) {
  if (n == 0)
    return;
  std::size_t const workers{std::max<std::size_t>(1, exec.concurrency())};
  std::size_t const chunks{std::min<std::size_t>(n, workers == 1 ? 1 : 8 * workers)};
  exec.bulk(chunks, [&](std::size_t c) { f(c * n / chunks, (c + 1) * n / chunks); });
}

} // namespace numsim::fft::detail

#endif // NUMSIM_FFT_EXECUTION_CHUNKING_H
