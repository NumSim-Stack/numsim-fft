#ifndef NUMSIM_FFT_DISTRIBUTED_COLLECTIVE_H
#define NUMSIM_FFT_DISTRIBUTED_COLLECTIVE_H

#include <mpl/mpl.hpp>

#include <cstddef>

namespace numsim::fft::detail {

/// Collective: true on every rank iff `value` is equal on all ranks.
inline bool same_on_all_ranks(mpl::communicator const &comm, std::size_t value) {
  unsigned long lo{value}, hi{value};
  comm.allreduce(mpl::min<unsigned long>(), lo);
  comm.allreduce(mpl::max<unsigned long>(), hi);
  return lo == hi;
}

} // namespace numsim::fft::detail

#endif // NUMSIM_FFT_DISTRIBUTED_COLLECTIVE_H
