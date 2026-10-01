#ifndef NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_H
#define NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_H

#include "../core/error.h"
#include "../core/expected.h"
#include "../core/workspace.h"
#include "slab_decomposition.h"
#include "transpose_sizes.h"

#include <mpl/mpl.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace numsim::fft::detail {

/**
 * @brief Global redistribution between the two slab layouts, for scalars S.
 *
 * Only the first two axes take part; everything behind them (the remaining
 * axes times the element components) moves as one block of `R` scalars.
 *
 *  - rows layout: [n0_local(rank)][M1][R], axis 0 split by `rows`.
 *  - cols layout: [N0][m1_local(rank)][R], axis 1 split by `cols`.
 *
 * Messages use MPI_Alltoallv with int counts and displacements; see
 * transpose_fits_int(). Callers must only call the transposes collectively.
 * Pack buffers and the MPI layouts come from the workspace and are reused.
 */
template <typename S> class slab_transpose {
public:
  using size_type = std::size_t;

  slab_transpose(mpl::communicator const &comm, slab_decomposition rows,
                 slab_decomposition cols, size_type R)
      : _comm{comm}, _rows{rows}, _cols{cols}, _R{R},
        _rank{static_cast<size_type>(comm.rank())}, _size{static_cast<size_type>(comm.size())} {}

  /// Messages and displacements fit MPI's int counts. Identical on every
  /// rank (pure function of the decomposition), so it is safe to return
  /// early on it before a collective.
  bool fits_int() const noexcept { return transpose_fits_int(_rows, _cols, _R); }

  /// MPI layouts and displacements of both directions, built once per
  /// (workspace, scalar type, R) and reused.
  struct cache {
    size_type R{0};
    size_type size{0};
    mpl::contiguous_layouts<S> rc_send, rc_recv, cr_send, cr_recv;
    mpl::displacements rc_send_d, rc_recv_d, cr_send_d, cr_recv_d;
  };

  template <real_scalar T> cache &layouts(workspace<T> &ws) const {
    return ws.template object<cache>(
        workspace_slot::transpose_cache, [&] { return build_cache(); },
        [&](cache const &c) { return c.R == _R && c.size == _size; });
  }

  /// rows layout -> cols layout.
  template <real_scalar T> void rows_to_cols(S const *src, S *dst, workspace<T> &ws) const {
    size_type const n0{_rows.local_size(_rank)}, M1{_cols.global_size()};
    cache const &c{layouts(ws)};
    // Pack: the block for rank q is src[:, offset1(q) .. +m1(q), :].
    S *const send{ws.template buffer<S>(workspace_slot::send, n0 * M1 * _R)};
    size_type pos{0};
    for (size_type q{0}; q < _size; ++q) {
      size_type const m1q{_cols.local_size(q)}, off1{_cols.offset(q)};
      for (size_type i0{0}; i0 < n0; ++i0) {
        std::copy_n(src + (i0 * M1 + off1) * _R, m1q * _R, send + pos);
        pos += m1q * _R;
      }
    }
    // From rank q: its rows, our columns; lands contiguously in dst.
    _comm.alltoallv(send, c.rc_send, c.rc_send_d, dst, c.rc_recv, c.rc_recv_d);
  }

  /// cols layout -> rows layout.
  template <real_scalar T> void cols_to_rows(S const *src, S *dst, workspace<T> &ws) const {
    size_type const n0{_rows.local_size(_rank)}, M1{_cols.global_size()};
    cache const &c{layouts(ws)};
    S *const recv{ws.template buffer<S>(workspace_slot::receive, n0 * M1 * _R)};
    // To rank q: its rows of our columns, contiguous in src.
    _comm.alltoallv(src, c.cr_send, c.cr_send_d, recv, c.cr_recv, c.cr_recv_d);
    // Unpack: the block from rank q is [n0][m1(q)][R] -> dst[:, offset1(q).., :].
    size_type pos{0};
    for (size_type q{0}; q < _size; ++q) {
      size_type const m1q{_cols.local_size(q)}, off1{_cols.offset(q)};
      for (size_type i0{0}; i0 < n0; ++i0) {
        std::copy_n(recv + pos, m1q * _R, dst + (i0 * M1 + off1) * _R);
        pos += m1q * _R;
      }
    }
  }

private:
  cache build_cache() const {
    size_type const n0{_rows.local_size(_rank)}, m1{_cols.local_size(_rank)};
    cache c{_R, _size, mpl::contiguous_layouts<S>(_size), mpl::contiguous_layouts<S>(_size),
            mpl::contiguous_layouts<S>(_size), mpl::contiguous_layouts<S>(_size),
            mpl::displacements(_size), mpl::displacements(_size), mpl::displacements(_size),
            mpl::displacements(_size)};
    size_type rc_pos{0}, cr_pos{0};
    for (size_type q{0}; q < _size; ++q) {
      size_type const n0q{_rows.local_size(q)}, m1q{_cols.local_size(q)};
      // rows -> cols: send our rows x q's columns (packed), receive q's rows
      // x our columns at row offset(q) of the cols layout.
      c.rc_send[q] = mpl::contiguous_layout<S>(n0 * m1q * _R);
      c.rc_send_d[q] = static_cast<MPI_Aint>(rc_pos * sizeof(S));
      rc_pos += n0 * m1q * _R;
      c.rc_recv[q] = mpl::contiguous_layout<S>(n0q * m1 * _R);
      c.rc_recv_d[q] = static_cast<MPI_Aint>(_rows.offset(q) * m1 * _R * sizeof(S));
      // cols -> rows: the reverse.
      c.cr_send[q] = mpl::contiguous_layout<S>(n0q * m1 * _R);
      c.cr_send_d[q] = static_cast<MPI_Aint>(_rows.offset(q) * m1 * _R * sizeof(S));
      c.cr_recv[q] = mpl::contiguous_layout<S>(n0 * m1q * _R);
      c.cr_recv_d[q] = static_cast<MPI_Aint>(cr_pos * sizeof(S));
      cr_pos += n0 * m1q * _R;
    }
    return c;
  }

private:
  mpl::communicator const &_comm;
  slab_decomposition _rows;
  slab_decomposition _cols;
  size_type _R;
  size_type _rank;
  size_type _size;
};

} // namespace numsim::fft::detail

#endif // NUMSIM_FFT_DISTRIBUTED_TRANSPOSE_H
