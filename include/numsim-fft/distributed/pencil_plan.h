#ifndef NUMSIM_FFT_DISTRIBUTED_PENCIL_PLAN_H
#define NUMSIM_FFT_DISTRIBUTED_PENCIL_PLAN_H

#if !defined(NUMSIM_FFT_HAS_MPI)
#error "numsim-fft: distributed plans need NUMSIM_FFT_ENABLE_MPI"
#endif

#include "../core/error.h"
#include "../core/expected.h"
#include "../transform/plan.h"
#include "pencil_transpose.h"
#include "pipelined_exchange.h"
#include "slab_decomposition.h"

#include <mpl/mpl.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace numsim::fft {

/// 3D transform over a P0 x P1 process grid (pencils): physical [n0(r0), n1(r1), N2],
/// spectral [S0, s1(r0), s2(r1)]; collective and error semantics as distributed_plan.
/// exchange_chunks > 1 pipelines one exchange per direction (no hooks; identical results).
template <real_scalar T> class pencil_plan {
public:
  using size_type = std::size_t;
  using kinds_type = std::array<axis_kind, 3>;
  using extents_type = extents<3>;
  using grid_type = std::array<size_type, 2>;
  using offsets_type = std::array<size_type, 3>;

  /// `grid` {0, 0} picks default_grid(); otherwise P0 * P1 must equal the
  /// number of ranks (error::invalid_process_grid).
  [[nodiscard]] static expected<pencil_plan, error>
  create(mpl::communicator const &comm, extents_type const &global, transform_domain domain,
         kinds_type const &kinds, plan_options const &options = {}, grid_type grid = {0, 0}) {
    auto serial{plan<T, 3>::create(global, domain, kinds, options)};
    if (!serial)
      return unexpected(serial.error());
    auto const ranks{static_cast<size_type>(comm.size())};
    if (grid[0] == 0 && grid[1] == 0)
      grid = default_grid(ranks, global[0]);
    if (grid[0] * grid[1] != ranks)
      return unexpected(error::invalid_process_grid);
    return pencil_plan{comm, *serial, domain, kinds, options, grid};
  }

  /// P0 = largest divisor of the rank count not above N0 (one row while every rank gets a slab).
  static grid_type default_grid(size_type ranks, size_type n0) noexcept {
    size_type p0{std::min(ranks, std::max<size_type>(n0, 1))};
    while (ranks % p0 != 0)
      --p0;
    return {p0, ranks / p0};
  }

  extents_type const &global_physical_extents() const noexcept { return _global_physical; }
  extents_type const &global_spectral_extents() const noexcept { return _global_spectral; }
  extents_type const &local_physical_extents() const noexcept { return _local_physical; }
  extents_type const &local_spectral_extents() const noexcept { return _local_spectral; }
  grid_type const &process_grid() const noexcept { return _grid; }

  /// First global index of the local physical pencil on every axis.
  offsets_type physical_offsets() const noexcept { return {_d0.offset(_r0), _d1.offset(_r1), 0}; }
  /// First global index of the local spectral pencil on every axis.
  offsets_type spectral_offsets() const noexcept { return {0, _e1.offset(_r0), _e2.offset(_r1)}; }

  mpl::communicator const &communicator() const noexcept { return _comm; }

  pencil_plan(pencil_plan const &) = delete;
  pencil_plan &operator=(pencil_plan const &) = delete;
  pencil_plan(pencil_plan &&) noexcept = default;
  pencil_plan &operator=(pencil_plan &&) noexcept = default;

  // --- forward ---

  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> forward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                              Exec const &exec = {}) const {
    workspace<T> ws;
    return forward(in, out, exec, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error> forward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                              workspace<T> &ws) const {
    return forward(in, out, sequential_executor{}, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error> forward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                              Exec const &exec, workspace<T> &ws) const {
    return forward_dispatch(in, out, exec, ws, static_cast<no_hook const *>(nullptr));
  }
  /// With a load hook on the first (axis-2) pass; points are local indices.
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec, typename F>
  [[nodiscard]] expected<void, error> forward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                              Exec const &exec, workspace<T> &ws, load_hook<F> const &hook) const {
    return forward_dispatch(in, out, exec, ws, &hook.f);
  }

  // --- backward ---

  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> backward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                               Exec const &exec = {}) const {
    workspace<T> ws;
    return backward(in, out, exec, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error> backward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                               workspace<T> &ws) const {
    return backward(in, out, sequential_executor{}, ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error> backward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                               Exec const &exec, workspace<T> &ws) const {
    return backward_dispatch(in, out, exec, ws, static_cast<no_hook *>(nullptr));
  }
  /// With a store hook on the last (axis-2) pass; points are local indices.
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec, typename S>
  [[nodiscard]] expected<void, error> backward(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                               Exec const &exec, workspace<T> &ws, store_hook<S> hook) const {
    return backward_dispatch(in, out, exec, ws, &hook.s);
  }

private:
  // Ez / Ey: element types after the axis-2 / axis-1 pass (complex from the r2c pass on).
  template <typename EIn, typename AIn, typename EOut, typename AOut, typename Exec, typename Load>
  expected<void, error> forward_dispatch(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                         Exec const &exec, workspace<T> &ws, Load const *load) const {
    if (!all_ranks_match(in.extents() == _local_physical && out.extents() == _local_spectral))
      return unexpected(error::extents_mismatch);
    if (_r2c_axis == size_type{2})
      return forward_impl<EOut, EOut>(in, out, exec, ws, load);
    if (_r2c_axis == size_type{1})
      return forward_impl<EIn, EOut>(in, out, exec, ws, load);
    return forward_impl<EIn, EIn>(in, out, exec, ws, load);
  }

  template <typename Ez, typename Ey, typename EIn, typename AIn, typename EOut, typename AOut, typename Exec,
            typename Load>
  expected<void, error> forward_impl(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out, Exec const &exec,
                                     workspace<T> &ws, Load const *load) const {
    constexpr size_type C{field<EIn, 3, AIn>::components};
    if (!fits_int(C))
      return unexpected(error::message_too_large);
    // With one rank in a row (column) the two layouts of its exchange are
    // the same, and the exchange is skipped.
    bool const row_exchange{_grid[1] > 1}, col_exchange{_grid[0] > 1};
    auto *const z_out{&temporary<field<Ez, 3>>(ws, workspace_slot::transpose_a, _plan_z.spectral_extents())};
    auto *const y_in{row_exchange ? &temporary<field<Ez, 3>>(ws, workspace_slot::transpose_b,
                                                             _plan_y.physical_extents())
                                  : z_out};
    auto *const y_out{&y_field<Ez, Ey>(ws, *y_in)};
    auto *const x_in{col_exchange ? &x_field<Ey>(ws, out) : y_out};
    using Sz = typename field<Ez, 3>::scalar_type;
    using Sy = typename field<Ey, 3>::scalar_type;
    if (_chunks > 1 && load == nullptr && (row_exchange || col_exchange)) {
      using Sin = typename field<EIn, 3, AIn>::scalar_type;
      size_type const rows{_local_physical[0]};
      size_type const in_row{_local_physical.inner(0) * C}, z_row{_plan_z.spectral_extents().inner(0) * C},
          yi_row{_plan_y.physical_extents().inner(0) * C}, yo_row{_plan_y.spectral_extents().inner(0) * C};
      expected<void, error> status{};
      auto z_pass = [&](size_type r0, size_type r1) {
        if (r1 > r0 && status)
          status = chunk_plan(_z_chunks, r1 - r0)
                       .template run_ptr<true, C>(static_cast<Sin const *>(in.data()) + r0 * in_row,
                                                  z_out->data() + r0 * z_row, exec, T(1), ws);
      };
      auto y_pass = [&](size_type r0, size_type r1) {
        if (r1 > r0 && status)
          status = chunk_plan(_y_chunks, r1 - r0)
                       .template run_ptr<true, C>(static_cast<Sz const *>(y_in->data()) + r0 * yi_row,
                                                  y_out->data() + r0 * yo_row, exec, T(1), ws);
      };
      auto rows_of = [&](size_type j) { return detail::pipelined_exchange<Sz>::chunk(j, rows, _chunks); };
      if (row_exchange) {
        // z of chunk j+1 runs while chunk j crosses the row; y as chunks arrive
        rows_pipeline<Sz>(C).a_to_b(
            z_out->data(), y_in->data(), ws, [&](size_type j) { z_pass(rows_of(j).first, rows_of(j).second); },
            [&](size_type j) { y_pass(rows_of(j).first, rows_of(j).second); });
        if (status && col_exchange)
          cols_transpose<Sy>(C).a_to_b(y_out->data(), x_in->data(), ws);
      } else {
        // one row: z and y of chunk j+1 run while chunk j crosses the column
        cols_pipeline<Sy>(C).a_to_b(
            y_out->data(), x_in->data(), ws,
            [&](size_type j) {
              z_pass(rows_of(j).first, rows_of(j).second);
              y_pass(rows_of(j).first, rows_of(j).second);
            },
            [](size_type) {});
      }
      if (!status)
        return status;
      return _plan_x.template run<true>(*x_in, out, exec, _scale_forward, ws);
    }
    return _plan_z.template run<true>(in, *z_out, exec, T(1), ws, nullptr, load, static_cast<no_hook *>(nullptr))
        .and_then([&]() -> expected<void, error> {
          if (row_exchange)
            rows_transpose<Sz>(C).a_to_b(z_out->data(), y_in->data(), ws);
          return _plan_y.template run<true>(*y_in, *y_out, exec, T(1), ws);
        })
        .and_then([&]() -> expected<void, error> {
          if (col_exchange)
            cols_transpose<Sy>(C).a_to_b(y_out->data(), x_in->data(), ws);
          return _plan_x.template run<true>(*x_in, out, exec, _scale_forward, ws);
        });
  }

  template <typename EIn, typename AIn, typename EOut, typename AOut, typename Exec, typename Store>
  expected<void, error> backward_dispatch(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out,
                                          Exec const &exec, workspace<T> &ws, Store *store) const {
    if (!all_ranks_match(in.extents() == _local_spectral && out.extents() == _local_physical))
      return unexpected(error::extents_mismatch);
    if (_r2c_axis == size_type{2})
      return backward_impl<EIn, EIn>(in, out, exec, ws, store);
    if (_r2c_axis == size_type{1})
      return backward_impl<EOut, EIn>(in, out, exec, ws, store);
    return backward_impl<EOut, EOut>(in, out, exec, ws, store);
  }

  template <typename Ez, typename Ey, typename EIn, typename AIn, typename EOut, typename AOut, typename Exec,
            typename Store>
  expected<void, error> backward_impl(field<EIn, 3, AIn> const &in, field<EOut, 3, AOut> &out, Exec const &exec,
                                      workspace<T> &ws, Store *store) const {
    constexpr size_type C{field<EIn, 3, AIn>::components};
    if (!fits_int(C))
      return unexpected(error::message_too_large);
    bool const row_exchange{_grid[1] > 1}, col_exchange{_grid[0] > 1};
    auto *const x_out{&temporary<field<Ey, 3>>(ws, workspace_slot::pencil_x, _plan_x.physical_extents())};
    field<Ez, 3> *y_out{nullptr};
    field<Ey, 3> *y_in{nullptr};
    if (col_exchange) {
      y_out = &temporary<field<Ez, 3>>(ws, workspace_slot::transpose_b, _plan_y.physical_extents());
      y_in = &y_field<Ez, Ey>(ws, *y_out);
    } else {
      y_in = x_out;
      y_out = &y_side<Ey, Ez>(ws, *x_out);
    }
    auto *const z_in{row_exchange ? &temporary<field<Ez, 3>>(ws, workspace_slot::transpose_a,
                                                             _plan_z.spectral_extents())
                                  : y_out};
    using Sz = typename field<Ez, 3>::scalar_type;
    using Sy = typename field<Ey, 3>::scalar_type;
    if (_chunks > 1 && store == nullptr && (row_exchange || col_exchange)) {
      using Sout = typename field<EOut, 3, AOut>::scalar_type;
      size_type const rows{_local_physical[0]};
      size_type const out_row{_local_physical.inner(0) * C}, z_row{_plan_z.spectral_extents().inner(0) * C},
          yi_row{_plan_y.spectral_extents().inner(0) * C}, yo_row{_plan_y.physical_extents().inner(0) * C};
      expected<void, error> status{};
      auto y_pass = [&](size_type r0, size_type r1) {
        if (r1 > r0 && status)
          status = chunk_plan(_y_chunks, r1 - r0)
                       .template run_ptr<false, C>(static_cast<Sy const *>(y_in->data()) + r0 * yi_row,
                                                   y_out->data() + r0 * yo_row, exec, T(1), ws);
      };
      auto z_pass = [&](size_type r0, size_type r1) {
        if (r1 > r0 && status)
          status = chunk_plan(_z_chunks, r1 - r0)
                       .template run_ptr<false, C>(static_cast<Sz const *>(z_in->data()) + r0 * z_row,
                                                   static_cast<Sout *>(out.data()) + r0 * out_row, exec,
                                                   _scale_backward, ws);
      };
      auto rows_of = [&](size_type j) { return detail::pipelined_exchange<Sz>::chunk(j, rows, _chunks); };
      status = _plan_x.template run<false>(in, *x_out, exec, T(1), ws);
      if (!status)
        return status;
      if (row_exchange) {
        if (col_exchange)
          cols_transpose<Sy>(C).b_to_a(x_out->data(), y_in->data(), ws);
        rows_pipeline<Sz>(C).b_to_a(
            y_out->data(), z_in->data(), ws, [&](size_type j) { y_pass(rows_of(j).first, rows_of(j).second); },
            [&](size_type j) { z_pass(rows_of(j).first, rows_of(j).second); });
      } else {
        cols_pipeline<Sy>(C).b_to_a(x_out->data(), y_in->data(), ws, [](size_type) {}, [&](size_type j) {
          y_pass(rows_of(j).first, rows_of(j).second);
          z_pass(rows_of(j).first, rows_of(j).second);
        });
      }
      return status;
    }
    return _plan_x.template run<false>(in, *x_out, exec, T(1), ws)
        .and_then([&]() -> expected<void, error> {
          if (col_exchange)
            cols_transpose<Sy>(C).b_to_a(x_out->data(), y_in->data(), ws);
          return _plan_y.template run<false>(*y_in, *y_out, exec, T(1), ws);
        })
        .and_then([&]() -> expected<void, error> {
          if (row_exchange)
            rows_transpose<Sz>(C).b_to_a(y_out->data(), z_in->data(), ws);
          return _plan_z.template run<false>(*z_in, out, exec, _scale_backward, ws, nullptr,
                                             static_cast<no_hook const *>(nullptr), store);
        });
  }

  /// The physical side of the backward axis-1 pass when its spectral side is
  /// given: the same field for an in-place pass, else its own.
  template <typename Ey, typename Ez> field<Ez, 3> &y_side(workspace<T> &ws, field<Ey, 3> &spectral) const {
    if constexpr (std::is_same_v<Ez, Ey>) {
      if (_plan_y.physical_extents() == _plan_y.spectral_extents())
        return spectral;
    }
    return temporary<field<Ez, 3>>(ws, workspace_slot::transpose_b, _plan_y.physical_extents());
  }

  /// The spectral side of the axis-1 pass: the physical-side field itself
  /// when the pass keeps type and extents (in-place pass), else its own.
  template <typename Ez, typename Ey> field<Ey, 3> &y_field(workspace<T> &ws, field<Ez, 3> &same) const {
    if constexpr (std::is_same_v<Ez, Ey>) {
      if (_plan_y.physical_extents() == _plan_y.spectral_extents())
        return same;
    }
    return temporary<field<Ey, 3>>(ws, workspace_slot::pencil_y, _plan_y.spectral_extents());
  }

  /// The physical side of the axis-0 pass: the output itself when the pass
  /// keeps type and extents (in-place pass), else a temporary.
  template <typename Ey, typename EOut, typename AOut>
  field<Ey, 3> &x_field(workspace<T> &ws, field<EOut, 3, AOut> &out) const {
    if constexpr (std::is_same_v<field<Ey, 3>, field<EOut, 3, AOut>>) {
      if (_plan_x.physical_extents() == _plan_x.spectral_extents())
        return out;
    }
    return temporary<field<Ey, 3>>(ws, workspace_slot::pencil_x, _plan_x.physical_extents());
  }

  pencil_plan(mpl::communicator const &comm, plan<T, 3> const &global, transform_domain domain,
              kinds_type const &kinds, plan_options const &options, grid_type grid)
      : _comm{comm}, _grid{grid}, _r0{static_cast<size_type>(_comm.rank()) / grid[1]},
        _r1{static_cast<size_type>(_comm.rank()) % grid[1]},
        _row{mpl::communicator::split, _comm, static_cast<int>(_r0), static_cast<int>(_r1)},
        _col{mpl::communicator::split, _comm, static_cast<int>(_r1), static_cast<int>(_r0)},
        _scale_forward{global.scale(true)}, _scale_backward{global.scale(false)},
        _global_physical{global.physical_extents()}, _global_spectral{global.spectral_extents()},
        _r2c_axis{global.r2c_axis()}, _d0{_global_physical[0], grid[0]}, _d1{_global_physical[1], grid[1]},
        _e1{_global_spectral[1], grid[0]}, _e2{_global_spectral[2], grid[1]},
        _local_physical{_d0.local_size(_r0), _d1.local_size(_r1), _global_physical[2]},
        _local_spectral{_global_spectral[0], _e1.local_size(_r0), _e2.local_size(_r1)},
        _plan_z{axis_plan(domain, kinds, options, 2, {_local_physical[0], _local_physical[1], _global_physical[2]})},
        _plan_y{axis_plan(domain, kinds, options, 1, {_local_physical[0], _global_physical[1], _e2.local_size(_r1)})},
        _plan_x{axis_plan(domain, kinds, options, 0,
                          {_global_physical[0], _e1.local_size(_r0), _e2.local_size(_r1)})},
        _chunks{std::max<size_type>(options.exchange_chunks, 1)},
        _z_chunks{make_chunk_plans(domain, kinds, options, 2, _plan_z.physical_extents())},
        _y_chunks{make_chunk_plans(domain, kinds, options, 1, _plan_y.physical_extents())} {}

  /// One-axis plan on the local layout of its pass: r2c if the axis is the
  /// r2c axis, real before it, complex after it.
  plan<T, 3> axis_plan(transform_domain domain, kinds_type const &all, plan_options options, size_type axis,
                       extents_type const &shape) const {
    kinds_type kinds;
    kinds.fill(axis_kind::identity);
    kinds[axis] = all[axis];
    options.norm = normalization::none;
    if (_r2c_axis) {
      if (axis == *_r2c_axis)
        domain = transform_domain::real_to_complex;
      else if (axis > *_r2c_axis)
        domain = transform_domain::real_to_real; // passes before the r2c pass
      else
        domain = transform_domain::complex_to_complex;
    }
    return plan<T, 3>::create(shape, domain, kinds, options).value();
  }

  /// The row exchange in chunks of the outer (axis-0) rows.
  template <typename S> detail::pipelined_exchange<S> rows_pipeline(size_type C) const {
    return {_row, _local_physical[0], _d1, _e2, C, _chunks, detail::pipelined_exchange<S>::split::outer,
            workspace_slot::exchange_cache};
  }
  /// The column exchange in chunks of every rank's axis-0 rows.
  template <typename S> detail::pipelined_exchange<S> cols_pipeline(size_type C) const {
    return {_col, 1, _d0, _e1, _e2.local_size(_r1) * C, _chunks, detail::pipelined_exchange<S>::split::a,
            workspace_slot::exchange_cache};
  }

  using chunk_plans = std::vector<std::pair<size_type, plan<T, 3>>>;

  static plan<T, 3> const &chunk_plan(chunk_plans const &plans, size_type rows) {
    for (auto const &[n, p] : plans)
      if (n == rows)
        return p;
    return plans.front().second; // not reached: every chunk size has a plan
  }

  /// Plans of one pass for every distinct chunk row count (exchange_chunks > 1).
  chunk_plans make_chunk_plans(transform_domain domain, kinds_type const &kinds, plan_options const &options,
                               size_type axis, extents_type const &shape) const {
    chunk_plans plans;
    if (_chunks < 2)
      return plans;
    for (size_type j{0}; j < _chunks; ++j) {
      auto const [r0, r1]{detail::pipelined_exchange<T>::chunk(j, shape[0], _chunks)};
      size_type const n{r1 - r0};
      if (n == 0 || std::ranges::any_of(plans, [n](auto const &p) { return p.first == n; }))
        continue;
      plans.emplace_back(n, axis_plan(domain, kinds, options, axis, shape.with_extent(0, n)));
    }
    return plans;
  }

  /// Axes 1 <-> 2 within the row: [n0][n1(r1)][S2] <-> [n0][N1][s2(r1)].
  template <typename S> detail::pencil_transpose<S> rows_transpose(size_type C) const {
    return {_row, _local_physical[0], _d1, _e2, C, workspace_slot::pencil_rows_cache};
  }
  /// Axes 0 <-> 1 within the column: [n0(r0)][S1][s2] <-> [N0][s1(r0)][s2].
  template <typename S> detail::pencil_transpose<S> cols_transpose(size_type C) const {
    return {_col, 1, _d0, _e1, _e2.local_size(_r1) * C, workspace_slot::pencil_cols_cache};
  }

  /// Both exchanges fit MPI's int counts on every rank (same verdict everywhere).
  bool fits_int(size_type C) const noexcept {
    using pt = detail::pencil_transpose<T>;
    return pt::fits_int(_d0.local_size(0), _d1, _e2, C) && pt::fits_int(1, _d0, _e1, _e2.local_size(0) * C);
  }

  template <typename F> static F &temporary(workspace<T> &ws, size_type slot, extents_type const &e) {
    return ws.template object<F>(slot, [&] { return F{e}; }, [&](F const &f) { return f.extents() == e; });
  }

  bool all_ranks_match(bool local) const {
    int ok{local ? 1 : 0};
    _comm.allreduce(mpl::min<int>(), ok);
    return ok == 1;
  }

  mpl::communicator _comm;
  grid_type _grid;
  size_type _r0;
  size_type _r1;
  mpl::communicator _row; // same r0: exchanges axes 1 and 2
  mpl::communicator _col; // same r1: exchanges axes 0 and 1
  T _scale_forward;
  T _scale_backward;
  extents_type _global_physical;
  extents_type _global_spectral;
  std::optional<size_type> _r2c_axis;
  slab_decomposition _d0; // physical axis 0 over P0
  slab_decomposition _d1; // physical axis 1 over P1
  slab_decomposition _e1; // spectral axis 1 over P0
  slab_decomposition _e2; // spectral axis 2 over P1
  extents_type _local_physical;
  extents_type _local_spectral;
  plan<T, 3> _plan_z;
  plan<T, 3> _plan_y;
  plan<T, 3> _plan_x;
  size_type _chunks;     // exchange_chunks
  chunk_plans _z_chunks; // axis-2 plans per chunk row count
  chunk_plans _y_chunks; // axis-1 plans per chunk row count
};

template <real_scalar T>
[[nodiscard]] expected<pencil_plan<T>, error>
make_pencil_c2c_plan(mpl::communicator const &comm, extents<3> const &global,
                     std::array<axis_kind, 3> const &kinds = {}, plan_options const &options = {},
                     std::array<std::size_t, 2> grid = {0, 0}) {
  return pencil_plan<T>::create(comm, global, transform_domain::complex_to_complex, kinds, options, grid);
}

template <real_scalar T>
[[nodiscard]] expected<pencil_plan<T>, error>
make_pencil_r2c_plan(mpl::communicator const &comm, extents<3> const &global,
                     std::array<axis_kind, 3> const &kinds = {}, plan_options const &options = {},
                     std::array<std::size_t, 2> grid = {0, 0}) {
  return pencil_plan<T>::create(comm, global, transform_domain::real_to_complex, kinds, options, grid);
}

template <real_scalar T>
[[nodiscard]] expected<pencil_plan<T>, error>
make_pencil_r2r_plan(mpl::communicator const &comm, extents<3> const &global, std::array<axis_kind, 3> const &kinds,
                     plan_options const &options = {}, std::array<std::size_t, 2> grid = {0, 0}) {
  return pencil_plan<T>::create(comm, global, transform_domain::real_to_real, kinds, options, grid);
}

} // namespace numsim::fft

#endif // NUMSIM_FFT_DISTRIBUTED_PENCIL_PLAN_H
