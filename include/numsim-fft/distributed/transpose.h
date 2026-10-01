#ifndef NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_H
#define NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_H

#include "../core/error.h"
#include "../core/expected.h"
#include "slab_decomposition.h"

#include <mpl/mpl.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace numsim_fft::detail {

/**
 * @brief Global redistribution between the two slab layouts, for scalars S.
 *
 * Only the first two axes take part; everything behind them (the remaining
 * axes times the element components) moves as one block of `R` scalars.
 *
 *  - rows layout: [n0_local(rank)][M1][R], axis 0 split by `rows`.
 *  - cols layout: [N0][m1_local(rank)][R], axis 1 split by `cols`.
 *
 * Messages use MPI_Alltoallv with int counts; a message above INT_MAX
 * scalars is reported as error::message_too_large.
 */
template <typename S> class slab_transpose {
public:
  using size_type = std::size_t;

  slab_transpose(mpl::communicator const &comm, slab_decomposition rows,
                 slab_decomposition cols, size_type R)
      : _comm{comm}, _rows{rows}, _cols{cols}, _R{R},
        _rank{static_cast<size_type>(comm.rank())}, _size{static_cast<size_type>(comm.size())} {}

  /// Largest message of either direction fits into an int count.
  bool counts_fit_int() const noexcept {
    size_type largest{0};
    for (size_type p{0}; p < _size; ++p) {
      largest = std::max(largest, _rows.local_size(_rank) * _cols.local_size(p) * _R);
      largest = std::max(largest, _rows.local_size(p) * _cols.local_size(_rank) * _R);
    }
    return largest <= static_cast<size_type>(std::numeric_limits<int>::max());
  }

  /// rows layout -> cols layout.
  void rows_to_cols(S const *src, S *dst) const {
    size_type const n0{_rows.local_size(_rank)}, M1{_cols.global_size()};
    size_type const m1{_cols.local_size(_rank)};
    // Pack: the block for rank q is src[:, offset1(q) .. +m1(q), :].
    std::vector<S> send(n0 * M1 * _R);
    mpl::contiguous_layouts<S> send_l(_size), recv_l(_size);
    mpl::displacements send_d(_size), recv_d(_size);
    size_type pos{0};
    for (size_type q{0}; q < _size; ++q) {
      size_type const m1q{_cols.local_size(q)}, off1{_cols.offset(q)};
      send_l[q] = mpl::contiguous_layout<S>(n0 * m1q * _R);
      send_d[q] = static_cast<MPI_Aint>(pos * sizeof(S));
      for (size_type i0{0}; i0 < n0; ++i0) {
        std::copy_n(src + (i0 * M1 + off1) * _R, m1q * _R, send.data() + pos);
        pos += m1q * _R;
      }
      // From rank q: its rows, our columns; lands contiguously in dst.
      recv_l[q] = mpl::contiguous_layout<S>(_rows.local_size(q) * m1 * _R);
      recv_d[q] = static_cast<MPI_Aint>(_rows.offset(q) * m1 * _R * sizeof(S));
    }
    _comm.alltoallv(send.data(), send_l, send_d, dst, recv_l, recv_d);
  }

  /// cols layout -> rows layout.
  void cols_to_rows(S const *src, S *dst) const {
    size_type const n0{_rows.local_size(_rank)}, M1{_cols.global_size()};
    size_type const m1{_cols.local_size(_rank)};
    mpl::contiguous_layouts<S> send_l(_size), recv_l(_size);
    mpl::displacements send_d(_size), recv_d(_size);
    std::vector<S> recv(n0 * M1 * _R);
    size_type pos{0};
    for (size_type q{0}; q < _size; ++q) {
      // To rank q: its rows of our columns, contiguous in src.
      send_l[q] = mpl::contiguous_layout<S>(_rows.local_size(q) * m1 * _R);
      send_d[q] = static_cast<MPI_Aint>(_rows.offset(q) * m1 * _R * sizeof(S));
      size_type const count{n0 * _cols.local_size(q) * _R};
      recv_l[q] = mpl::contiguous_layout<S>(count);
      recv_d[q] = static_cast<MPI_Aint>(pos * sizeof(S));
      pos += count;
    }
    _comm.alltoallv(src, send_l, send_d, recv.data(), recv_l, recv_d);
    // Unpack: the block from rank q is [n0][m1(q)][R] -> dst[:, offset1(q).., :].
    pos = 0;
    for (size_type q{0}; q < _size; ++q) {
      size_type const m1q{_cols.local_size(q)}, off1{_cols.offset(q)};
      for (size_type i0{0}; i0 < n0; ++i0) {
        std::copy_n(recv.data() + pos, m1q * _R, dst + (i0 * M1 + off1) * _R);
        pos += m1q * _R;
      }
    }
  }

private:
  mpl::communicator const &_comm;
  slab_decomposition _rows;
  slab_decomposition _cols;
  size_type _R;
  size_type _rank;
  size_type _size;
};

} // namespace numsim_fft::detail

#endif // NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_H
