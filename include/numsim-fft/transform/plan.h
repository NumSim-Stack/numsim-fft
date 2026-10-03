#ifndef NUMSIM_FFT_TRANSFORM_PLAN_H
#define NUMSIM_FFT_TRANSFORM_PLAN_H

#include "../core/aligned_allocator.h"
#include "../core/error.h"
#include "../core/expected.h"
#include "../core/scalar_traits.h"
#include "../core/workspace.h"
#include "../execution/executor.h"
#include "../execution/sequential.h"
#include "../field/field.h"
#include "../kernel/c2c_plan_1d.h"
#include "../kernel/r2c_plan_1d.h"
#include "../kernel/r2r_plan_1d.h"
#include "../layout/extents.h"
#include "axis_kind.h"
#include "normalization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace numsim::fft {

template <real_scalar T, std::size_t Dim> class distributed_plan;

/// Scalar types of the physical-space and spectral fields.
enum class transform_domain {
  complex_to_complex, ///< complex <-> complex
  real_to_complex,    ///< real <-> Hermitian half spectrum
  real_to_real        ///< real <-> real, DCT/DST on every axis
};

struct plan_options {
  normalization norm{normalization::backward};
  /// Lines transformed together (vector batch); 0 picks a cache-sized block.
  std::size_t max_batch{0};
};

/// No hook (the default of the hook parameters).
struct no_hook {};

/**
 * @brief Point hook applied while the first pass of a forward transform
 * reads the input: `f(point, src, dst)` writes the C scalars of `point`
 * that the transform sees, given the C scalars stored in the input field.
 * Saves a separate pass over memory for a point-wise map before the
 * transform (e.g. a material tangent times a strain field).
 */
template <typename F> struct load_hook {
  F f;
};
template <typename F> load_hook(F) -> load_hook<F>;

/**
 * @brief Point hook called right after the last pass of a backward
 * transform wrote a point: `s.begin(items)` once before the pass, then
 * `s(item, point, values)` for every point, with the C scalars written and
 * the work item (a block of lines) that wrote them. Items are fixed by the
 * grid, not by the executor, and one item is written by one thread: a
 * reduction accumulated per item and summed over the items in order gives
 * the same result for every executor and thread count.
 */
template <typename S> struct store_hook {
  S &s;
};
template <typename S> store_hook(S &) -> store_hook<S>;

/**
 * @brief Reusable n-D transform of fields on a fixed grid.
 *
 * Every axis gets a transform kind: periodic (DFT) or a DCT/DST. Axes are
 * processed one after another (row-column). Each 1D pass transforms blocks
 * of adjacent lines as vector batches, so the components of tensor elements
 * and the faster grid axes are in the vectorised inner loop.
 *
 * Domains:
 *  - complex_to_complex: DFT on periodic axes. The r2r kinds are applied to
 *    the real and imaginary parts.
 *  - real_to_complex: the *last periodic* axis p becomes a real-input
 *    transform, and its spectral extent is n_p/2+1. The forward transform
 *    does r2c along p first, then the other axes on the complex data. The
 *    backward transform does the other axes first, then c2r along p.
 *  - real_to_real: DCT/DST on all axes.
 *
 * One plan serves every element type with scalar T or std::complex<T>
 * (scalars, rank-2 and rank-4 tmech tensors); the component count comes
 * from the field type.
 *
 * Execution: forward(in, out, exec, ws) and backward(in, out, exec, ws) read
 * `in` and never modify it. Plans are immutable, so they may be called
 * concurrently, each caller with its own workspace. For c2c and r2r plans,
 * `in` and `out` may be the same field. In loops pass a workspace: it keeps
 * the scratch buffers and the intermediate field of a multi-axis r2c
 * backward transform, so no allocation happens after the first call.
 *
 * backward_destructive(in, out, exec, ws) is the same transform but may
 * overwrite `in` with intermediate data; a multi-axis r2c backward transform
 * then needs no field-sized temporary and one pass less over memory.
 */
template <real_scalar T, std::size_t Dim> class plan {
public:
  using size_type = std::size_t;
  using real_type = T;
  using complex_type = std::complex<T>;
  using kinds_type = std::array<axis_kind, Dim>;
  using extents_type = extents<Dim>;

  [[nodiscard]] static expected<plan, error> create(extents_type const &shape,
                                                    transform_domain domain,
                                                    kinds_type const &kinds,
                                                    plan_options const &options = {}) {
    std::optional<size_type> r2c_axis;
    for (size_type d{0}; d < Dim; ++d) {
      if (shape[d] == 0 && is_transformed(kinds[d]))
        return unexpected(error::zero_extent);
      if (shape[d] < minimum_size(kinds[d]))
        return unexpected(error::size_below_minimum);
      if (kinds[d] == axis_kind::periodic) {
        if (domain == transform_domain::real_to_real)
          return unexpected(error::periodic_axis_in_r2r);
        r2c_axis = d;
      }
    }
    if (domain == transform_domain::real_to_complex && !r2c_axis)
      return unexpected(error::no_periodic_axis);
    if (domain != transform_domain::real_to_complex)
      r2c_axis.reset();
    return plan{shape, domain, kinds, options, r2c_axis};
  }

  transform_domain domain() const noexcept { return _domain; }
  kinds_type const &kinds() const noexcept { return _kinds; }
  plan_options const &options() const noexcept { return _options; }

  /// Grid of the physical-space field.
  extents_type const &physical_extents() const noexcept { return _physical; }

  /// Grid of the spectral field (the r2c axis holds n/2+1 values).
  extents_type const &spectral_extents() const noexcept { return _spectral; }

  /// Axis with the real-to-complex transform, if the domain is r2c.
  std::optional<size_type> r2c_axis() const noexcept { return _r2c_axis; }

  /// Product of the logical sizes over the axes (normalisation factor N).
  long double logical_size() const noexcept {
    long double n{1};
    for (size_type d{0}; d < Dim; ++d)
      n *= static_cast<long double>(numsim::fft::logical_size(_kinds[d], _physical[d]));
    return n;
  }

  /// Physical space -> spectral space. Without a workspace a temporary one
  /// is used (allocates per call; pass a workspace in loops).
  template <typename EIn, typename AIn, typename EOut, typename AOut,
            executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out,
                                              Exec const &exec = {}) const {
    workspace<T> ws;
    return run<true>(in, out, exec, scale(true), ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out, Exec const &exec,
                                              workspace<T> &ws) const {
    return run<true>(in, out, exec, scale(true), ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out, workspace<T> &ws) const {
    return run<true>(in, out, sequential_executor{}, scale(true), ws);
  }

  /// Forward transform with a load hook on the first pass (see load_hook).
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec, typename F>
  [[nodiscard]] expected<void, error> forward(field<EIn, Dim, AIn> const &in,
                                              field<EOut, Dim, AOut> &out, Exec const &exec,
                                              workspace<T> &ws, load_hook<F> const &hook) const {
    return run<true>(in, out, exec, scale(true), ws, nullptr, &hook.f,
                     static_cast<no_hook *>(nullptr));
  }

  /// Spectral space -> physical space.
  template <typename EIn, typename AIn, typename EOut, typename AOut,
            executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> backward(field<EIn, Dim, AIn> const &in,
                                               field<EOut, Dim, AOut> &out,
                                               Exec const &exec = {}) const {
    workspace<T> ws;
    return run<false>(in, out, exec, scale(false), ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error> backward(field<EIn, Dim, AIn> const &in,
                                               field<EOut, Dim, AOut> &out, Exec const &exec,
                                               workspace<T> &ws) const {
    return run<false>(in, out, exec, scale(false), ws);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error>
  backward(field<EIn, Dim, AIn> const &in, field<EOut, Dim, AOut> &out, workspace<T> &ws) const {
    return run<false>(in, out, sequential_executor{}, scale(false), ws);
  }

  /// Backward transform with a store hook on the last pass (see store_hook).
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec, typename S>
  [[nodiscard]] expected<void, error> backward(field<EIn, Dim, AIn> const &in,
                                               field<EOut, Dim, AOut> &out, Exec const &exec,
                                               workspace<T> &ws, store_hook<S> hook) const {
    return run<false>(in, out, exec, scale(false), ws, nullptr,
                      static_cast<no_hook const *>(nullptr), &hook.s);
  }

  /// Spectral space -> physical space; `in` may be overwritten.
  template <typename EIn, typename AIn, typename EOut, typename AOut,
            executor Exec = sequential_executor>
  [[nodiscard]] expected<void, error> backward_destructive(field<EIn, Dim, AIn> &in,
                                                           field<EOut, Dim, AOut> &out,
                                                           Exec const &exec = {}) const {
    workspace<T> ws;
    return run<false>(in, out, exec, scale(false), ws, in.data());
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec>
  [[nodiscard]] expected<void, error>
  backward_destructive(field<EIn, Dim, AIn> &in, field<EOut, Dim, AOut> &out, Exec const &exec,
                       workspace<T> &ws) const {
    return run<false>(in, out, exec, scale(false), ws, in.data());
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut, executor Exec, typename S>
  [[nodiscard]] expected<void, error>
  backward_destructive(field<EIn, Dim, AIn> &in, field<EOut, Dim, AOut> &out, Exec const &exec,
                       workspace<T> &ws, store_hook<S> hook) const {
    return run<false>(in, out, exec, scale(false), ws, in.data(),
                      static_cast<no_hook const *>(nullptr), &hook.s);
  }
  template <typename EIn, typename AIn, typename EOut, typename AOut>
  [[nodiscard]] expected<void, error> backward_destructive(field<EIn, Dim, AIn> &in,
                                                           field<EOut, Dim, AOut> &out,
                                                           workspace<T> &ws) const {
    return run<false>(in, out, sequential_executor{}, scale(false), ws, in.data());
  }

private:
  template <real_scalar, std::size_t> friend class distributed_plan;

  struct axis_plans {
    std::optional<kernel::c2c_plan_1d<T>> c2c;
    std::optional<kernel::r2c_plan_1d<T>> r2c;
    std::optional<kernel::r2r_plan_1d<T>> r2r_forward;
    std::optional<kernel::r2r_plan_1d<T>> r2r_backward;
  };

  plan(extents_type const &shape, transform_domain domain, kinds_type const &kinds,
       plan_options const &options, std::optional<size_type> r2c_axis)
      : _physical{shape}, _spectral{shape}, _domain{domain}, _kinds{kinds}, _options{options},
        _r2c_axis{r2c_axis} {
    if (_r2c_axis)
      _spectral = shape.with_extent(*_r2c_axis, shape[*_r2c_axis] / 2 + 1);
    for (size_type d{0}; d < Dim; ++d) {
      size_type const n{shape[d]};
      if (!is_transformed(kinds[d]))
        continue;
      if (_r2c_axis && d == *_r2c_axis)
        _axes[d].r2c.emplace(n);
      else if (kinds[d] == axis_kind::periodic)
        _axes[d].c2c.emplace(n);
      else {
        _axes[d].r2r_forward.emplace(n, kinds[d]);
        _axes[d].r2r_backward.emplace(n, inverse_kind(kinds[d]));
      }
    }
  }

  T scale(bool forward) const noexcept {
    switch (_options.norm) {
    case normalization::none:
      return T(1);
    case normalization::backward:
      return forward ? T(1) : static_cast<T>(1.0L / logical_size());
    case normalization::ortho:
      return static_cast<T>(1.0L / std::sqrt(logical_size()));
    }
    return T(1);
  }

  /// Lines per block for a kernel needing `scratch_per_line` complex values
  /// of scratch per line: keeps the working set around 64 KiB (measured
  /// best compromise between L1 residency and per-block overhead).
  size_type block_size(size_type scratch_per_line, size_type inner) const noexcept {
    if (inner == 0)
      return 1;
    if (_options.max_batch > 0)
      return std::min(_options.max_batch, inner);
    constexpr size_type budget{64 * 1024};
    size_type const bytes{std::max<size_type>(1, scratch_per_line * sizeof(complex_type))};
    return std::clamp<size_type>(budget / bytes, 1, inner);
  }

  /// With point hooks a block of lines must hold whole points (multiples of
  /// the component count C; lines are always aligned to C).
  static size_type hook_block(size_type block, size_type C, size_type inner, bool hooks) noexcept {
    if (!hooks || C <= 1)
      return block;
    return std::min(inner, std::max(C, block / C * C));
  }

  /**
   * @brief One axis pass: for each of `outer` slabs, lines along the axis
   * with element stride `inner` (in units of the data type). `op(in, out, B,
   * scratch)` transforms B adjacent lines. Work items (slab, block of lines)
   * are split into chunks. Each chunk owns its scratch.
   */
  template <typename In, typename Out, typename Op, typename Exec, typename Load = no_hook,
            typename Store = no_hook>
  static void for_each_block(Exec const &exec, workspace<T> &ws, size_type outer, size_type inner,
                             size_type in_slab, size_type out_slab, size_type block,
                             size_type scratch_len, In *in, Out *out, Op const &op, size_type C = 1,
                             Load const *load = nullptr, Store *store = nullptr) {
    size_type const blocks_per_slab{inner == 0 ? 0 : (inner + block - 1) / block};
    size_type const items{outer * blocks_per_slab};
    // A few chunks per worker for load balance; one for a single worker.
    size_type const workers{std::max<size_type>(1, exec.concurrency())};
    size_type const chunks{std::min<size_type>(items, workers == 1 ? 1 : 4 * workers)};
    if constexpr (!std::is_same_v<Store, no_hook>)
      if (store)
        store->begin(items);
    if (items == 0)
      return; // empty grid, e.g. a rank without rows
    using staged_type = std::remove_const_t<In>;
    // staging buffer of the load hook: the mapped input lines of one block
    size_type const n_in{in_slab / inner}, n_out{out_slab / inner};
    size_type const staging{load ? (n_in * block * sizeof(staged_type) + sizeof(complex_type) - 1) /
                                       sizeof(complex_type)
                                 : 0};
    ws.prepare_chunks(chunks);
    exec.bulk(chunks, [&](size_type chunk) {
      complex_type *const scratch_data{ws.chunk_scratch(chunk, scratch_len + staging)};
      staged_type *const staged{reinterpret_cast<staged_type *>(scratch_data + scratch_len)};
      size_type const first{chunk * items / chunks};
      size_type const last{(chunk + 1) * items / chunks};
      for (size_type item{first}; item < last; ++item) {
        size_type const o{item / blocks_per_slab};
        size_type const b0{(item % blocks_per_slab) * block};
        size_type const B{std::min(block, inner - b0)};
        if constexpr (!std::is_same_v<Load, no_hook>) {
          if (load) {
            for (size_type j{0}; j < n_in; ++j) {
              size_type const base{o * in_slab + j * inner + b0};
              for (size_type q{0}; q < B; q += C)
                (*load)((base + q) / C, in + base + q, staged + j * B + q);
            }
            op(static_cast<In *>(staged), B, out + o * out_slab + b0, B, scratch_data);
          } else {
            op(in + o * in_slab + b0, inner, out + o * out_slab + b0, B, scratch_data);
          }
        } else {
          op(in + o * in_slab + b0, inner, out + o * out_slab + b0, B, scratch_data);
        }
        if constexpr (!std::is_same_v<Store, no_hook>)
          if (store)
            for (size_type j{0}; j < n_out; ++j) {
              size_type const base{o * out_slab + j * inner + b0};
              for (size_type q{0}; q < B; q += C)
                (*store)(item, (base + q) / C, static_cast<Out const *>(out + base + q));
            }
      }
    });
  }

  /// Complex pass along `axis` (c2c, or r2r on real and imaginary parts).
  /// Returns false when hooks are requested on an r2r axis (its lines are
  /// the real and imaginary parts, not whole points).
  template <bool Forward, typename Exec, typename Load = no_hook, typename Store = no_hook>
  bool complex_pass(Exec const &exec, workspace<T> &ws, size_type axis, extents_type const &ext,
                    size_type components, complex_type const *in, complex_type *out, T scale_factor,
                    Load const *load = nullptr, Store *store = nullptr) const {
    size_type const outer{ext.outer(axis)}, n{ext[axis]};
    size_type const inner{ext.inner(axis) * components};
    axis_plans const &ap{_axes[axis]};
    if (ap.c2c) {
      auto const &k{*ap.c2c};
      size_type const block{
          hook_block(block_size(k.scratch_size(1), inner), components, inner, load || store)};
      for_each_block(
          exec, ws, outer, inner, n * inner, n * inner, block, k.scratch_size(block), in, out,
          [&](complex_type const *i, size_type is, complex_type *o, size_type B, complex_type *s) {
            if constexpr (Forward)
              k.forward(i, is, o, inner, B, s, scale_factor);
            else
              k.backward(i, is, o, inner, B, s, scale_factor);
          },
          components, load, store);
      return true;
    }
    if (load || store)
      return false;
    // A complex line batch is a real batch of twice the width.
    auto const &k{Forward ? *ap.r2r_forward : *ap.r2r_backward};
    real_pass_impl(exec, ws, k, outer, n, 2 * inner, reinterpret_cast<T const *>(in),
                   reinterpret_cast<T *>(out), scale_factor);
    return true;
  }

  /// Real r2r pass along `axis`.
  template <bool Forward, typename Exec, typename Load = no_hook, typename Store = no_hook>
  void real_pass(Exec const &exec, workspace<T> &ws, size_type axis, extents_type const &ext,
                 size_type components, T const *in, T *out, T scale_factor,
                 Load const *load = nullptr, Store *store = nullptr) const {
    auto const &k{Forward ? *_axes[axis].r2r_forward : *_axes[axis].r2r_backward};
    real_pass_impl(exec, ws, k, ext.outer(axis), ext[axis], ext.inner(axis) * components, in, out,
                   scale_factor, components, load, store);
  }

  template <typename Exec, typename Load = no_hook, typename Store = no_hook>
  void real_pass_impl(Exec const &exec, workspace<T> &ws, kernel::r2r_plan_1d<T> const &k,
                      size_type outer, size_type n, size_type inner, T const *in, T *out,
                      T scale_factor, size_type components = 1, Load const *load = nullptr,
                      Store *store = nullptr) const {
    size_type const block{
        hook_block(block_size(k.scratch_size(1), inner), components, inner, load || store)};
    for_each_block(
        exec, ws, outer, inner, n * inner, n * inner, block, k.scratch_size(block), in, out,
        [&](T const *i, size_type is, T *o, size_type B, complex_type *s) {
          k.execute(i, is, o, inner, B, s, scale_factor);
        },
        components, load, store);
  }

  /// r2c (Forward) or c2r pass along the r2c axis.
  template <bool Forward, typename Exec, typename In, typename Out, typename Load = no_hook,
            typename Store = no_hook>
  void half_spectrum_pass(Exec const &exec, workspace<T> &ws, size_type components, In const *in,
                          Out *out, T scale_factor, Load const *load = nullptr,
                          Store *store = nullptr) const {
    size_type const axis{*_r2c_axis};
    auto const &k{*_axes[axis].r2c};
    size_type const outer{_physical.outer(axis)};
    size_type const inner{_physical.inner(axis) * components};
    size_type const n_real{_physical[axis]}, n_spec{_spectral[axis]};
    size_type const in_slab{(Forward ? n_real : n_spec) * inner};
    size_type const out_slab{(Forward ? n_spec : n_real) * inner};
    size_type const block{
        hook_block(block_size(k.scratch_size(1), inner), components, inner, load || store)};
    for_each_block(
        exec, ws, outer, inner, in_slab, out_slab, block, k.scratch_size(block), in, out,
        [&](In const *i, size_type is, Out *o, size_type B, complex_type *s) {
          if constexpr (Forward)
            k.forward(i, is, o, inner, B, s, scale_factor);
          else
            k.backward(i, is, o, inner, B, s, scale_factor);
        },
        components, load, store);
  }

  /// Hooks on a plan without transformed axes: map, copy and scale per point.
  template <typename S, typename Load, typename Store>
  static void copy_scaled_hooked(S const *in, S *out, size_type points, size_type C, T scale_factor,
                                 Load const *load, Store *store) {
    std::vector<S> tmp(C);
    if constexpr (!std::is_same_v<Store, no_hook>)
      if (store)
        store->begin(1);
    for (size_type p{0}; p < points; ++p) {
      S const *src{in + p * C};
      if constexpr (!std::is_same_v<Load, no_hook>)
        if (load) {
          (*load)(p, src, tmp.data());
          src = tmp.data();
        }
      for (size_type c{0}; c < C; ++c)
        out[p * C + c] = src[c] * scale_factor;
      if constexpr (!std::is_same_v<Store, no_hook>)
        if (store)
          (*store)(size_type{0}, p, static_cast<S const *>(out + p * C));
    }
  }

  template <typename S>
  static void copy_scaled(S const *in, S *out, size_type count, T scale_factor) noexcept {
    for (size_type i{0}; i < count; ++i)
      out[i] = in[i] * scale_factor;
  }

  /// `scratch_in` is the input's own storage when it may be overwritten
  /// (backward_destructive), otherwise null.
  /// `load` is applied by the first pass of a forward transform, `store` by
  /// the last pass of a backward transform (null: no hook).
  template <bool Forward, typename EIn, typename AIn, typename EOut, typename AOut, typename Exec,
            typename Load = no_hook, typename Store = no_hook>
  expected<void, error> run(field<EIn, Dim, AIn> const &in, field<EOut, Dim, AOut> &out,
                            Exec const &exec, T const s, workspace<T> &ws,
                            typename field<EIn, Dim, AIn>::scalar_type *scratch_in = nullptr,
                            Load const *load = nullptr, Store *store = nullptr) const {
    using in_scalar = typename field<EIn, Dim, AIn>::scalar_type;
    using out_scalar = typename field<EOut, Dim, AOut>::scalar_type;
    static_assert(field<EIn, Dim, AIn>::components == field<EOut, Dim, AOut>::components,
                  "plan: input and output elements must have the same number "
                  "of components");
    static_assert(std::is_same_v<real_type_t<in_scalar>, T> &&
                      std::is_same_v<real_type_t<out_scalar>, T>,
                  "plan: field scalar precision differs from the plan's");
    constexpr size_type C{field<EIn, Dim, AIn>::components};
    constexpr bool in_complex{is_complex_v<in_scalar>};
    constexpr bool out_complex{is_complex_v<out_scalar>};
    // Valid combinations: real->real (r2r), complex->complex (c2c),
    // real->complex forward / complex->real backward (r2c). The others can
    // never match a domain and are rejected at compile time; whether the
    // fields fit *this* plan's domain is checked at run time below.
    static_assert(Forward ? !(in_complex && !out_complex) : !(!in_complex && out_complex),
                  "plan: complex input with real output is only valid for backward "
                  "(and real input with complex output only for forward)");

    extents_type const &expected_in{Forward ? _physical : _spectral};
    extents_type const &expected_out{Forward ? _spectral : _physical};
    if (in.extents() != expected_in || out.extents() != expected_out)
      return unexpected(error::extents_mismatch);

    // Transformed axes in order; for r2c the half-spectrum axis is excluded.
    std::array<size_type, Dim> active{};
    size_type n_active{0};
    for (size_type d{0}; d < Dim; ++d)
      if (is_transformed(_kinds[d]) && !(_r2c_axis && d == *_r2c_axis))
        active[n_active++] = d;

    // hooks only where they apply: load on the first forward pass, store on
    // the last backward pass
    Load const *const first_load{Forward ? load : nullptr};
    Store *const last_store{Forward ? nullptr : store};
    bool const any_hook{first_load || last_store};

    switch (_domain) {
    case transform_domain::complex_to_complex:
      if constexpr (in_complex && out_complex) {
        if (n_active == 0) {
          if (any_hook)
            copy_scaled_hooked(in.data(), out.data(), in.size(), C, s, first_load, last_store);
          else
            copy_scaled(in.data(), out.data(), in.scalar_size(), s);
        }
        for (size_type a{0}; a < n_active; ++a)
          if (!complex_pass<Forward>(exec, ws, active[a], _physical, C,
                                     a == 0 ? in.data() : out.data(), out.data(),
                                     a + 1 == n_active ? s : T(1), a == 0 ? first_load : nullptr,
                                     a + 1 == n_active ? last_store : nullptr))
            return unexpected(error::hooks_unsupported);
        return {};
      }
      break;
    case transform_domain::real_to_real:
      if constexpr (!in_complex && !out_complex) {
        if (n_active == 0) {
          if (any_hook)
            copy_scaled_hooked(in.data(), out.data(), in.size(), C, s, first_load, last_store);
          else
            copy_scaled(in.data(), out.data(), in.scalar_size(), s);
        }
        for (size_type a{0}; a < n_active; ++a)
          real_pass<Forward>(exec, ws, active[a], _physical, C, a == 0 ? in.data() : out.data(),
                             out.data(), a + 1 == n_active ? s : T(1),
                             a == 0 ? first_load : nullptr,
                             a + 1 == n_active ? last_store : nullptr);
        return {};
      }
      break;
    case transform_domain::real_to_complex:
      if constexpr (Forward && !in_complex && out_complex) {
        half_spectrum_pass<true>(exec, ws, C, in.data(), out.data(), n_active == 0 ? s : T(1),
                                 first_load, static_cast<no_hook *>(nullptr));
        for (size_type a{0}; a < n_active; ++a)
          if (!complex_pass<true>(exec, ws, active[a], _spectral, C, out.data(), out.data(),
                                  a + 1 == n_active ? s : T(1)))
            return unexpected(error::hooks_unsupported);
        return {};
      } else if constexpr (!Forward && in_complex && !out_complex) {
        if (n_active == 0) {
          half_spectrum_pass<false>(exec, ws, C, in.data(), out.data(), s,
                                    static_cast<no_hook const *>(nullptr), last_store);
        } else {
          // The other axes go first; the first of them writes into the
          // intermediate field (the input itself when it may be destroyed),
          // the rest work in place.
          complex_type *const work{scratch_in
                                       ? scratch_in
                                       : ws.template buffer<complex_type>(
                                             workspace_slot::backward_copy, in.scalar_size())};
          for (size_type a{0}; a < n_active; ++a)
            complex_pass<false>(exec, ws, active[a], _spectral, C, a == 0 ? in.data() : work, work,
                                T(1));
          half_spectrum_pass<false>(exec, ws, C, work, out.data(), s,
                                    static_cast<no_hook const *>(nullptr), last_store);
        }
        return {};
      }
      break;
    }
    return unexpected(error::domain_mismatch);
  }

  extents_type _physical;
  extents_type _spectral;
  transform_domain _domain;
  kinds_type _kinds;
  plan_options _options;
  std::optional<size_type> _r2c_axis;
  std::array<axis_plans, Dim> _axes{};
};

/// Complex-to-complex plan. Kinds default to periodic on every axis.
template <real_scalar T, std::size_t Dim>
[[nodiscard]] expected<plan<T, Dim>, error>
make_c2c_plan(extents<Dim> const &shape, std::array<axis_kind, Dim> const &kinds = {},
              plan_options const &options = {}) {
  return plan<T, Dim>::create(shape, transform_domain::complex_to_complex, kinds, options);
}

/// Real-to-complex plan; the last periodic axis carries the half spectrum.
template <real_scalar T, std::size_t Dim>
[[nodiscard]] expected<plan<T, Dim>, error>
make_r2c_plan(extents<Dim> const &shape, std::array<axis_kind, Dim> const &kinds = {},
              plan_options const &options = {}) {
  return plan<T, Dim>::create(shape, transform_domain::real_to_complex, kinds, options);
}

/// Real-to-real plan; every axis needs a DCT/DST kind.
template <real_scalar T, std::size_t Dim>
[[nodiscard]] expected<plan<T, Dim>, error> make_r2r_plan(extents<Dim> const &shape,
                                                          std::array<axis_kind, Dim> const &kinds,
                                                          plan_options const &options = {}) {
  return plan<T, Dim>::create(shape, transform_domain::real_to_real, kinds, options);
}

} // namespace numsim::fft

#endif // NUMSIM_FFT_TRANSFORM_PLAN_H
