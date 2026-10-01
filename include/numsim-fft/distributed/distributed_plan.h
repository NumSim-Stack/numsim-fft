#ifndef NUMSIM_FFT_DISTRIBUTED_DISTRIBUTED_PLAN_H
#define NUMSIM_FFT_DISTRIBUTED_DISTRIBUTED_PLAN_H

#if !defined(NUMSIM_FFT_HAS_MPI)
#error "numsim-fft: distributed plans need NUMSIM_FFT_ENABLE_MPI"
#endif

#include "../core/error.h"
#include "../core/expected.h"
#include "../transform/plan.h"
#include "slab_decomposition.h"
#include "transpose.h"

#include <mpl/mpl.hpp>

#include <array>
#include <cstddef>

namespace numsim_fft {

/**
 * @brief n-D transform (Dim >= 2) of a field distributed over the ranks of
 * an MPI communicator, with a slab decomposition.
 *
 * Layouts (each rank holds an ordinary field with local extents):
 *  - physical space: split along axis 0,
 *    local extents [n0_local, N1, ..., N_{d-1}], starting at physical_offset();
 *  - spectral space: split along axis 1, kept in natural axis order,
 *    local extents [S0, s1_local, S2, ...] of the global spectral grid S,
 *    starting at spectral_offset() on axis 1.
 *
 * Forward: transforms along axes 1..d-1 locally, one global transpose
 * (MPI_Alltoallv), then the axis-0 transform. Backward mirrors it. The
 * kinds, domains and normalisation are those of plan (the global logical
 * size is used for normalisation). If the half-spectrum axis of an r2c plan
 * is axis 0, the local phase stays real and the r2c runs after the transpose.
 *
 * The local passes take any executor, so MPI combines with OpenMP or HPX
 * within a rank. Ranks may own no rows or columns (more ranks than points).
 *
 * Collective semantics: create(), forward(), backward() and the destructor
 * are collective over the communicator (the plan duplicates it). Errors are
 * agreed on by all ranks: if any rank's fields do not match, every rank
 * returns error::extents_mismatch and nobody enters the transpose. The plan
 * is move-only, so no rank can duplicate the communicator on its own.
 *
 * Pass a workspace in loops: the pre/post-transpose fields, the MPI pack
 * buffers and the MPI layouts then persist between calls.
 */
template <real_scalar T, std::size_t Dim> class distributed_plan {
  static_assert(Dim >= 2, "distributed_plan: slab decomposition needs Dim >= 2");

public:
  using size_type = std::size_t;
  using kinds_type = std::array<axis_kind, Dim>;
  using extents_type = extents<Dim>;

  [[nodiscard]] static expected<distributed_plan, error> create(mpl::communicator const &comm,
                                                  extents_type const &global,
                                                  transform_domain domain, kinds_type const &kinds,
                                                  plan_options const &options = {}) {
    // Validates the arguments exactly as a serial plan would.
    auto serial{plan<T, Dim>::create(global, domain, kinds, options)};
    if (!serial)
      return unexpected(serial.error());
    return distributed_plan{comm, *serial, domain, kinds, options};
  }

  extents_type const &global_physical_extents() const noexcept { return _global_physical; }
  extents_type const &global_spectral_extents() const noexcept { return _global_spectral; }
  extents_type const &local_physical_extents() const noexcept { return _local_physical; }
  extents_type const &local_spectral_extents() const noexcept { return _local_spectral; }

  /// First global index along axis 0 of the local physical slab.
  size_type physical_offset() const noexcept { return _rows.offset(_rank); }

  /// First global index along axis 1 of the local spectral slab.
  size_type spectral_offset() const noexcept { return _cols.offset(_rank); }

  mpl::communicator const &communicator() const noexcept { return _comm; }

  distributed_plan(distributed_plan const &) = delete;
  distributed_plan &operator=(distributed_plan const &) = delete;
  distributed_plan(distributed_plan &&) noexcept = default;
  distributed_plan &operator=(distributed_plan &&) noexcept = default;

  template <typename EIn, typename AIn, typename EOut, typename AOut,
            executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out,
                                              Exec const &exec = {}) const {
    workspace<T> ws;
    return forward(in, out, exec, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out, workspace<T> &ws) const {
    return forward(in, out, sequential_executor{}, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out, Exec const &exec,
                                              workspace<T> &ws) const {
    if (!all_ranks_match(in.extents() == _local_physical && out.extents() == _local_spectral))
      return unexpected(error::extents_mismatch);
    constexpr size_type C{field<EIn, Dim, AIn>::components};
    T const s{_scale_forward};
    if (_half_spectrum_on_axis0) {
      // real local phase, real transpose, then r2c along axis 0
      auto &a{temporary<field<EIn, Dim>>(ws, workspace_slot::transpose_a, _local.spectral_extents())};
      auto &b{temporary<field<EIn, Dim>>(ws, workspace_slot::transpose_b, _axis0.physical_extents())};
      return _local.template run<true>(in, a, exec, T(1), ws)
          .and_then([&] { return transpose<true, C>(a, b, ws); })
          .and_then([&] { return _axis0.template run<true>(b, out, exec, s, ws); });
    }
    auto &a{temporary<field<EOut, Dim>>(ws, workspace_slot::transpose_a, _local.spectral_extents())};
    return _local.template run<true>(in, a, exec, T(1), ws)
        .and_then([&] { return transpose<true, C>(a, out, ws); })
        .and_then([&] { return _axis0.template run<true>(out, out, exec, s, ws); });
  }

  template <typename EIn, typename AIn, typename EOut, typename AOut,
            executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> backward(field<EIn, Dim, AIn> const &in,
                                               field<EOut, Dim, AOut> &out,
                                               Exec const &exec = {}) const {
    workspace<T> ws;
    return backward(in, out, exec, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error> backward(field<EIn, Dim, AIn> const &in,
                                               field<EOut, Dim, AOut> &out, workspace<T> &ws) const {
    return backward(in, out, sequential_executor{}, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error> backward(field<EIn, Dim, AIn> const &in,
                                               field<EOut, Dim, AOut> &out, Exec const &exec,
                                               workspace<T> &ws) const {
    if (!all_ranks_match(in.extents() == _local_spectral && out.extents() == _local_physical))
      return unexpected(error::extents_mismatch);
    constexpr size_type C{field<EIn, Dim, AIn>::components};
    T const s{_scale_backward};
    if (_half_spectrum_on_axis0) {
      auto &b{temporary<field<EOut, Dim>>(ws, workspace_slot::transpose_b, _axis0.physical_extents())};
      auto &a{temporary<field<EOut, Dim>>(ws, workspace_slot::transpose_a, _local.spectral_extents())};
      return _axis0.template run<false>(in, b, exec, T(1), ws)
          .and_then([&] { return transpose<false, C>(b, a, ws); })
          .and_then([&] { return _local.template run<false>(a, out, exec, s, ws); });
    }
    auto &b{temporary<field<EIn, Dim>>(ws, workspace_slot::transpose_b, _axis0.spectral_extents())};
    auto &a{temporary<field<EIn, Dim>>(ws, workspace_slot::transpose_a, _local.spectral_extents())};
    return _axis0.template run<false>(in, b, exec, T(1), ws)
        .and_then([&] { return transpose<false, C>(b, a, ws); })
        .and_then([&] { return _local.template run<false>(a, out, exec, s, ws); });
  }

private:
  distributed_plan(mpl::communicator const &comm, plan<T, Dim> const &global,
                   transform_domain domain, kinds_type const &kinds,
                   plan_options const &options)
      : _comm{comm}, _rank{static_cast<size_type>(_comm.rank())},
        _size{static_cast<size_type>(_comm.size())}, _scale_forward{global.scale(true)},
        _scale_backward{global.scale(false)}, _global_physical{global.physical_extents()}, _global_spectral{global.spectral_extents()},
        _rows{_global_physical[0], _size},
        _cols{_global_spectral[1], _size},
        _half_spectrum_on_axis0{global.r2c_axis() == size_type{0}},
        _local_physical{_global_physical.with_extent(0, _rows.local_size(_rank))},
        _local_spectral{_global_spectral.with_extent(1, _cols.local_size(_rank))},
        _local{make_local(domain, kinds, options)}, _axis0{make_axis0(domain, kinds, options)} {}

  /// Transforms along axes 1..d-1 on [n0_local, N1, ...].
  plan<T, Dim> make_local(transform_domain domain, kinds_type kinds,
                          plan_options options) const {
    kinds[0] = axis_kind::identity;
    options.norm = normalization::none;
    if (_half_spectrum_on_axis0)
      domain = transform_domain::real_to_real; // axes 1.. are DCT/DST or identity
    return plan<T, Dim>::create(_local_physical, domain, kinds, options).value();
  }

  /// Transform along axis 0 on the transposed layout [N0, m1_local, ...].
  plan<T, Dim> make_axis0(transform_domain domain, kinds_type const &all,
                          plan_options options) const {
    kinds_type kinds;
    kinds.fill(axis_kind::identity);
    kinds[0] = all[0];
    options.norm = normalization::none;
    if (_half_spectrum_on_axis0)
      return plan<T, Dim>::create(_global_physical.with_extent(1, _cols.local_size(_rank)),
                                  transform_domain::real_to_complex, kinds, options)
          .value();
    if (domain == transform_domain::real_to_complex)
      domain = transform_domain::complex_to_complex; // data are complex after r2c
    return plan<T, Dim>::create(_local_spectral, domain, kinds, options).value();
  }

  /// Field of the given extents kept in the workspace (rebuilt on change).
  template <typename F>
  static F &temporary(workspace<T> &ws, size_type slot, extents_type const &e) {
    return ws.template object<F>(
        slot, [&] { return F{e}; }, [&](F const &f) { return f.extents() == e; });
  }

  /// Collective agreement on a local condition (logical and over ranks).
  bool all_ranks_match(bool local) const {
    int ok{local ? 1 : 0};
    _comm.allreduce(mpl::min<int>(), ok);
    return ok == 1;
  }

  /// rows <-> cols redistribution of the first two axes.
  template <bool RowsToCols, size_type C, typename E1, typename A1, typename E2, typename A2>
  expected<void, error> transpose(field<E1, Dim, A1> const &src, field<E2, Dim, A2> &dst,
                                  workspace<T> &ws) const {
    using S = typename field<E1, Dim, A1>::scalar_type;
    // src/dst of the transpose share axes 2.. with the global spectral grid,
    // except in the axis-0 half-spectrum case (then the physical grid).
    auto const &ref{_half_spectrum_on_axis0 ? _global_physical : _global_spectral};
    detail::slab_transpose<S> const t{_comm, _rows, _cols, ref.inner(1) * C};
    if (!t.fits_int()) // same verdict on every rank
      return unexpected(error::message_too_large);
    if constexpr (RowsToCols)
      t.rows_to_cols(src.data(), dst.data(), ws);
    else
      t.cols_to_rows(src.data(), dst.data(), ws);
    return {};
  }

  mpl::communicator _comm;
  size_type _rank;
  size_type _size;
  T _scale_forward;
  T _scale_backward;
  extents_type _global_physical;
  extents_type _global_spectral;
  slab_decomposition _rows;      // physical axis 0 (also axis 0 of the transpose)
  slab_decomposition _cols;      // spectral axis 1
  bool _half_spectrum_on_axis0;
  extents_type _local_physical;
  extents_type _local_spectral;
  plan<T, Dim> _local;
  plan<T, Dim> _axis0;
};

template <real_scalar T, std::size_t Dim>
[[nodiscard]] expected<distributed_plan<T, Dim>, error>
make_distributed_c2c_plan(mpl::communicator const &comm, extents<Dim> const &global,
                          std::array<axis_kind, Dim> const &kinds = {},
                          plan_options const &options = {}) {
  return distributed_plan<T, Dim>::create(comm, global, transform_domain::complex_to_complex,
                                          kinds, options);
}

template <real_scalar T, std::size_t Dim>
[[nodiscard]] expected<distributed_plan<T, Dim>, error>
make_distributed_r2c_plan(mpl::communicator const &comm, extents<Dim> const &global,
                          std::array<axis_kind, Dim> const &kinds = {},
                          plan_options const &options = {}) {
  return distributed_plan<T, Dim>::create(comm, global, transform_domain::real_to_complex, kinds,
                                          options);
}

template <real_scalar T, std::size_t Dim>
[[nodiscard]] expected<distributed_plan<T, Dim>, error>
make_distributed_r2r_plan(mpl::communicator const &comm, extents<Dim> const &global,
                          std::array<axis_kind, Dim> const &kinds,
                          plan_options const &options = {}) {
  return distributed_plan<T, Dim>::create(comm, global, transform_domain::real_to_real, kinds,
                                          options);
}

} // namespace numsim_fft

#endif // NUMSIM_FFT_DISTRIBUTED_DISTRIBUTED_PLAN_H
