#ifndef NUMSIM_FFT_DISTRIBUTED_PENCIL_TRANSPOSE_H
#define NUMSIM_FFT_DISTRIBUTED_PENCIL_TRANSPOSE_H

#include "../core/workspace.h"
#include "slab_decomposition.h"
#include "transpose_sizes.h"

#include <mpl/mpl.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace numsim::fft::detail {

/// Exchange of two adjacent axes over a (sub-)communicator: a layout [outer][a_local][B][R]
/// <-> b layout [outer][A][b_local][R]; MPI layouts cached in `cache_slot`.
template <typename S> class pencil_transpose {
public:
  using size_type = std::size_t;

  pencil_transpose(mpl::communicator const &comm, size_type outer, slab_decomposition a,
                   slab_decomposition b, size_type R, size_type cache_slot)
      : _comm{comm}, _outer{outer}, _a{a}, _b{b}, _R{R}, _slot{cache_slot},
        _rank{static_cast<size_type>(comm.rank())}, _size{static_cast<size_type>(comm.size())} {}

  /// Messages and buffers fit MPI int counts (same verdict on every rank).
  static bool fits_int(size_type outer_max, slab_decomposition a, slab_decomposition b, size_type R) noexcept {
    size_type const a_max{a.local_size(0)}, b_max{b.local_size(0)};
    constexpr auto limit{static_cast<size_type>(std::numeric_limits<int>::max())};
    size_type const block{saturating_mul(outer_max, R)};
    return saturating_mul(saturating_mul(block, a_max), b_max) <= limit &&
           saturating_mul(saturating_mul(block, a_max), b.global_size()) <= limit &&
           saturating_mul(saturating_mul(block, a.global_size()), b_max) <= limit;
  }

  /// a layout -> b layout.
  template <typename T> void a_to_b(S const *src, S *dst, workspace<T> &ws) const {
    size_type const al{_a.local_size(_rank)}, A{_a.global_size()}, B{_b.global_size()},
        bl{_b.local_size(_rank)};
    // own block: one direct copy
    for (size_type o{0}; o < _outer; ++o)
      for (size_type i{0}; i < al; ++i)
        std::copy_n(src + ((o * al + i) * B + _b.offset(_rank)) * _R, bl * _R,
                    dst + ((o * A + _a.offset(_rank) + i) * bl) * _R);
    if (_size == 1)
      return;
    S *const send{ws.template buffer<S>(workspace_slot::send, _outer * al * B * _R)};
    size_type pos{0};
    for (size_type q{0}; q < _size; ++q) {
      if (q == _rank)
        continue;
      size_type const bq{_b.local_size(q)}, off{_b.offset(q)};
      for (size_type o{0}; o < _outer; ++o)
        for (size_type i{0}; i < al; ++i, pos += bq * _R)
          std::copy_n(src + ((o * al + i) * B + off) * _R, bq * _R, send + pos);
    }
    cache const &c{layouts(ws)};
    if (_outer == 1) { // the blocks from q are contiguous in dst: receive in place
      _comm.alltoallv(send, c.ab_send, c.ab_send_d, dst, c.ab_recv, c.ab_recv_d);
      return;
    }
    S *const recv{ws.template buffer<S>(workspace_slot::receive, _outer * A * bl * _R)};
    _comm.alltoallv(send, c.ab_send, c.ab_send_d, recv, c.ab_recv, c.ab_recv_d);
    pos = 0;
    for (size_type q{0}; q < _size; ++q) {
      if (q == _rank)
        continue;
      size_type const aq{_a.local_size(q)}, off{_a.offset(q)};
      for (size_type o{0}; o < _outer; ++o)
        for (size_type i{0}; i < aq; ++i, pos += bl * _R)
          std::copy_n(recv + pos, bl * _R, dst + ((o * A + off + i) * bl) * _R);
    }
  }

  /// b layout -> a layout.
  template <typename T> void b_to_a(S const *src, S *dst, workspace<T> &ws) const {
    size_type const al{_a.local_size(_rank)}, A{_a.global_size()}, B{_b.global_size()},
        bl{_b.local_size(_rank)};
    for (size_type o{0}; o < _outer; ++o)
      for (size_type i{0}; i < al; ++i)
        std::copy_n(src + ((o * A + _a.offset(_rank) + i) * bl) * _R, bl * _R,
                    dst + ((o * al + i) * B + _b.offset(_rank)) * _R);
    if (_size == 1)
      return;
    cache const &c{layouts(ws)};
    S *const recv{ws.template buffer<S>(workspace_slot::receive, _outer * al * B * _R)};
    if (_outer == 1) { // the blocks for q are contiguous in src: send in place
      _comm.alltoallv(src, c.ba_send, c.ba_send_d, recv, c.ba_recv, c.ba_recv_d);
    } else {
      S *const send{ws.template buffer<S>(workspace_slot::send, _outer * A * bl * _R)};
      size_type pos{0};
      for (size_type q{0}; q < _size; ++q) {
        if (q == _rank)
          continue;
        size_type const aq{_a.local_size(q)}, off{_a.offset(q)};
        for (size_type o{0}; o < _outer; ++o)
          for (size_type i{0}; i < aq; ++i, pos += bl * _R)
            std::copy_n(src + ((o * A + off + i) * bl) * _R, bl * _R, send + pos);
      }
      _comm.alltoallv(send, c.ba_send, c.ba_send_d, recv, c.ba_recv, c.ba_recv_d);
    }
    size_type pos{0};
    for (size_type q{0}; q < _size; ++q) {
      if (q == _rank)
        continue;
      size_type const bq{_b.local_size(q)}, off{_b.offset(q)};
      for (size_type o{0}; o < _outer; ++o)
        for (size_type i{0}; i < al; ++i, pos += bq * _R)
          std::copy_n(recv + pos, bq * _R, dst + ((o * al + i) * B + off) * _R);
    }
  }

private:
  /// What the cached layouts depend on.
  struct cache_key {
    MPI_Comm comm{MPI_COMM_NULL};
    size_type rank{0}, size{0}, outer{0}, R{0}, A{0}, B{0};
    bool operator==(cache_key const &) const = default;
  };
  struct cache {
    cache_key key;
    mpl::contiguous_layouts<S> ab_send, ab_recv, ba_send, ba_recv;
    mpl::displacements ab_send_d, ab_recv_d, ba_send_d, ba_recv_d;
  };

  cache_key key() const noexcept {
    return {_comm.native_handle(), _rank, _size, _outer, _R, _a.global_size(), _b.global_size()};
  }

  template <typename T> cache const &layouts(workspace<T> &ws) const {
    return ws.template object<cache>(
        _slot, [&] { return build_cache(); }, [&](cache const &c) { return c.key == key(); });
  }

  /// Own block: count 0; packed blocks in q order; outer == 1 receives in place.
  cache build_cache() const {
    size_type const al{_a.local_size(_rank)}, bl{_b.local_size(_rank)};
    cache c{key(),
            mpl::contiguous_layouts<S>(_size),
            mpl::contiguous_layouts<S>(_size),
            mpl::contiguous_layouts<S>(_size),
            mpl::contiguous_layouts<S>(_size),
            mpl::displacements(_size),
            mpl::displacements(_size),
            mpl::displacements(_size),
            mpl::displacements(_size)};
    size_type to_pos{0}, from_pos{0};
    for (size_type q{0}; q < _size; ++q) {
      bool const self{q == _rank};
      // a -> b: to q our a rows x its b columns, from q its a rows x our b
      // columns; b -> a is the same exchange reversed.
      size_type const to_q{self ? 0 : _outer * al * _b.local_size(q) * _R};
      size_type const from_q{self ? 0 : _outer * _a.local_size(q) * bl * _R};
      auto const bytes = [](size_type n) { return static_cast<MPI_Aint>(n * sizeof(S)); };
      size_type const in_place{_a.offset(q) * bl * _R};
      c.ab_send[q] = mpl::contiguous_layout<S>(to_q);
      c.ab_send_d[q] = bytes(to_pos);
      c.ab_recv[q] = mpl::contiguous_layout<S>(from_q);
      c.ab_recv_d[q] = bytes(_outer == 1 ? in_place : from_pos);
      c.ba_send[q] = mpl::contiguous_layout<S>(from_q);
      c.ba_send_d[q] = bytes(_outer == 1 ? in_place : from_pos);
      c.ba_recv[q] = mpl::contiguous_layout<S>(to_q);
      c.ba_recv_d[q] = bytes(to_pos);
      to_pos += to_q;
      from_pos += from_q;
    }
    return c;
  }

  mpl::communicator const &_comm;
  size_type _outer;
  slab_decomposition _a;
  slab_decomposition _b;
  size_type _R;
  size_type _slot;
  size_type _rank;
  size_type _size;
};

} // namespace numsim::fft::detail

#endif // NUMSIM_FFT_DISTRIBUTED_PENCIL_TRANSPOSE_H
