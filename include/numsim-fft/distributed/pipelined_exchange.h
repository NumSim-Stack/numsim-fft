#ifndef NUMSIM_FFT_DISTRIBUTED_PIPELINED_EXCHANGE_H
#define NUMSIM_FFT_DISTRIBUTED_PIPELINED_EXCHANGE_H

#include "../core/workspace.h"
#include "slab_decomposition.h"

#include <mpi.h>
#include <mpl/mpl.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace numsim::fft::detail {

/**
 * @brief The two-axis exchange of pencil_transpose (a layout
 * [outer][a_local][B][R] <-> b layout [outer][A][b_local][R]), split into
 * `chunks` non-blocking all-to-alls that are pipelined with the caller's
 * local work:
 *
 *   produce(0); start(0);
 *   for j = 1..k:  produce(j); start(j);   (j < k)
 *                  wait(j-1); consume(j-1);
 *
 * so the transform of one chunk runs while the previous chunk is in flight.
 * produce(j) must have written the source rows of chunk j, consume(j) may
 * use the destination rows of chunk j.
 *
 * Chunks split either the outer rows (split::outer: chunk j is the outer
 * rows [j*outer/k, (j+1)*outer/k), the same on every rank, so all ranks of
 * the communicator must have the same `outer`) or the a rows of every rank
 * (split::a: chunk j of rank q is its a rows [j*a(q)/k, (j+1)*a(q)/k)).
 *
 * MPI_Ialltoallv is called directly: the count and displacement arrays
 * must stay valid until completion, they live in a workspace cache. Every
 * rank must use the same `chunks` (collective calls must match).
 */
template <typename S> class pipelined_exchange {
public:
  using size_type = std::size_t;
  enum class split { outer, a };

  pipelined_exchange(mpl::communicator const &comm, size_type outer, slab_decomposition a, slab_decomposition b,
                     size_type R, size_type chunks, split mode, size_type cache_slot)
      : _comm{comm}, _outer{outer}, _a{a}, _b{b}, _R{R}, _k{std::max<size_type>(chunks, 1)}, _mode{mode},
        _slot{cache_slot}, _rank{static_cast<size_type>(comm.rank())}, _size{static_cast<size_type>(comm.size())} {}

  /// [first, last) of chunk j of n rows.
  static constexpr std::pair<size_type, size_type> chunk(size_type j, size_type n, size_type k) noexcept {
    return {j * n / k, (j + 1) * n / k};
  }

  /// a layout -> b layout.
  template <typename T, typename Produce, typename Consume>
  void a_to_b(S const *src, S *dst, workspace<T> &ws, Produce &&produce, Consume &&consume) const {
    run<true>(src, dst, ws, produce, consume);
  }

  /// b layout -> a layout.
  template <typename T, typename Produce, typename Consume>
  void b_to_a(S const *src, S *dst, workspace<T> &ws, Produce &&produce, Consume &&consume) const {
    run<false>(src, dst, ws, produce, consume);
  }

private:
  struct direction {
    std::vector<std::vector<int>> send_counts, send_displs, recv_counts, recv_displs; // [chunk][rank]
  };
  struct cache {
    size_type outer{0}, R{0}, A{0}, B{0}, size{0}, k{0};
    split mode{split::outer};
    direction ab, ba;
  };

  /// Outer rows and a rows of rank q in chunk j.
  std::pair<size_type, size_type> outer_rows(size_type j) const {
    return _mode == split::outer ? chunk(j, _outer, _k) : std::pair<size_type, size_type>{0, _outer};
  }
  std::pair<size_type, size_type> a_rows(size_type j, size_type q) const {
    size_type const aq{_a.local_size(q)};
    return _mode == split::a ? chunk(j, aq, _k) : std::pair<size_type, size_type>{0, aq};
  }
  /// The b-layout blocks of chunk j are contiguous (receive/send in place).
  bool b_side_in_place() const noexcept { return _mode == split::a && _outer == 1; }

  /// Calls f(offset_in_a_layout, offset_in_b_layout, count) for every
  /// contiguous run that chunk j moves between `q`'s side (b columns of q
  /// for the a layout of this rank, a rows of q for the b layout).
  template <bool AtoB, typename F> void runs(size_type j, size_type q, F const &f) const {
    size_type const al{_a.local_size(_rank)}, A{_a.global_size()}, B{_b.global_size()}, bl{_b.local_size(_rank)};
    auto const [o0, o1]{outer_rows(j)};
    // a-layout side: this rank's a rows, q's b columns; b-layout side: q's a
    // rows, this rank's b columns. Which one this rank sends depends on AtoB.
    if constexpr (AtoB) {
      (void)A;
      auto const [i0, i1]{a_rows(j, _rank)};
      for (size_type o{o0}; o < o1; ++o)
        for (size_type i{i0}; i < i1; ++i)
          f(((o * al + i) * B + _b.offset(q)) * _R, _b.local_size(q) * _R);
    } else {
      (void)B;
      auto const [i0, i1]{a_rows(j, q)};
      for (size_type o{o0}; o < o1; ++o)
        for (size_type i{i0}; i < i1; ++i)
          f(((o * A + _a.offset(q) + i) * bl) * _R, bl * _R);
    }
  }
  /// Runs on the receiving side, mirrored.
  template <bool AtoB, typename F> void recv_runs(size_type j, size_type q, F const &f) const {
    runs<!AtoB>(j, q, f);
  }

  size_type send_total(bool atob, size_type j, size_type q) const {
    size_type n{0};
    if (atob)
      runs<true>(j, q, [&](size_type, size_type c) { n += c; });
    else
      runs<false>(j, q, [&](size_type, size_type c) { n += c; });
    return n;
  }

  /// Counts and displacements (elements) of both directions, every chunk.
  cache build_cache() const {
    cache c{_outer, _R, _a.global_size(), _b.global_size(), _size, _k, _mode, {}, {}};
    auto fill = [&](direction &d, bool atob) {
      d.send_counts.assign(_k, std::vector<int>(_size, 0));
      d.send_displs = d.recv_counts = d.recv_displs = d.send_counts;
      size_type send_pos{0}, recv_pos{0};
      for (size_type j{0}; j < _k; ++j)
        for (size_type q{0}; q < _size; ++q) {
          if (q == _rank)
            continue;
          size_type const to_q{send_total(atob, j, q)}, from_q{send_total(!atob, j, q)};
          d.send_counts[j][q] = static_cast<int>(to_q);
          d.recv_counts[j][q] = static_cast<int>(from_q);
          // packed by default; the b side in place where it is contiguous
          // (the first offset of q's single run)
          size_type in_place{0};
          if (b_side_in_place()) {
            bool first{true};
            runs<false>(j, q, [&](size_type off, size_type) {
              if (first)
                in_place = off;
              first = false;
            });
          }
          d.send_displs[j][q] = static_cast<int>(b_side_in_place() && !atob ? in_place : send_pos);
          d.recv_displs[j][q] = static_cast<int>(b_side_in_place() && atob ? in_place : recv_pos);
          send_pos += to_q;
          recv_pos += from_q;
        }
    };
    fill(c.ab, true);
    fill(c.ba, false);
    return c;
  }

  template <typename T> cache const &layouts(workspace<T> &ws) const {
    return ws.template object<cache>(
        _slot, [&] { return build_cache(); },
        [&](cache const &c) {
          return c.outer == _outer && c.R == _R && c.A == _a.global_size() && c.B == _b.global_size() &&
                 c.size == _size && c.k == _k && c.mode == _mode;
        });
  }

  template <bool AtoB, typename T, typename Produce, typename Consume>
  void run(S const *src, S *dst, workspace<T> &ws, Produce &produce, Consume &consume) const {
    cache const &c{layouts(ws)};
    direction const &d{AtoB ? c.ab : c.ba};
    // send = a-layout side for AtoB (packed), b side for b_to_a (packed or in place)
    bool const send_in_place{!AtoB && b_side_in_place()};
    bool const recv_in_place{AtoB && b_side_in_place()};
    size_type send_size{0}, recv_size{0};
    for (size_type j{0}; j < _k; ++j)
      for (size_type q{0}; q < _size; ++q) {
        send_size += static_cast<size_type>(d.send_counts[j][q]);
        recv_size += static_cast<size_type>(d.recv_counts[j][q]);
      }
    S *const send{send_in_place ? nullptr : ws.template buffer<S>(workspace_slot::send, send_size)};
    S *const recv{recv_in_place ? nullptr : ws.template buffer<S>(workspace_slot::receive, recv_size)};
    std::vector<MPI_Request> requests(_k, MPI_REQUEST_NULL);
    MPI_Datatype const type{mpl::detail::datatype_traits<S>::get_datatype()};

    auto start = [&](size_type j) {
      // own block: direct copy between the layouts
      if constexpr (AtoB) {
        std::vector<size_type> to;
        recv_runs<true>(j, _rank, [&](size_type off, size_type) { to.push_back(off); });
        size_type r{0};
        runs<true>(j, _rank, [&](size_type off, size_type n) { std::copy_n(src + off, n, dst + to[r++]); });
      } else {
        std::vector<size_type> to;
        recv_runs<false>(j, _rank, [&](size_type off, size_type) { to.push_back(off); });
        size_type r{0};
        runs<false>(j, _rank, [&](size_type off, size_type n) { std::copy_n(src + off, n, dst + to[r++]); });
      }
      if (_size == 1)
        return;
      if (!send_in_place)
        for (size_type q{0}; q < _size; ++q) {
          if (q == _rank)
            continue;
          S *p{send + d.send_displs[j][q]};
          runs<AtoB>(j, q, [&](size_type off, size_type n) {
            std::copy_n(src + off, n, p);
            p += n;
          });
        }
      MPI_Ialltoallv(send_in_place ? src : send, d.send_counts[j].data(), d.send_displs[j].data(), type,
                     recv_in_place ? dst : recv, d.recv_counts[j].data(), d.recv_displs[j].data(), type,
                     _comm.native_handle(), &requests[j]);
    };
    auto finish = [&](size_type j) {
      if (_size > 1) {
        MPI_Wait(&requests[j], MPI_STATUS_IGNORE);
        if (!recv_in_place)
          for (size_type q{0}; q < _size; ++q) {
            if (q == _rank)
              continue;
            S const *p{recv + d.recv_displs[j][q]};
            recv_runs<AtoB>(j, q, [&](size_type off, size_type n) {
              std::copy_n(p, n, dst + off);
              p += n;
            });
          }
      }
      consume(j);
    };
    auto progress = [&](size_type upto) {
      // let MPI move the chunks in flight while this rank computes
      for (size_type j{0}; j < upto; ++j)
        if (requests[j] != MPI_REQUEST_NULL) {
          int done{0};
          MPI_Test(&requests[j], &done, MPI_STATUS_IGNORE);
          if (done)
            requests[j] = MPI_REQUEST_NULL; // MPI_Wait on a null request returns at once
        }
    };

    produce(size_type{0});
    start(0);
    for (size_type j{1}; j <= _k; ++j) {
      if (j < _k) {
        progress(j);
        produce(j);
        start(j);
      }
      finish(j - 1);
    }
  }

  mpl::communicator const &_comm;
  size_type _outer;
  slab_decomposition _a;
  slab_decomposition _b;
  size_type _R;
  size_type _k;
  split _mode;
  size_type _slot;
  size_type _rank;
  size_type _size;
};

} // namespace numsim::fft::detail

#endif // NUMSIM_FFT_DISTRIBUTED_PIPELINED_EXCHANGE_H
