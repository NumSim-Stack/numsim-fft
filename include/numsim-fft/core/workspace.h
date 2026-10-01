#ifndef NUMSIM_FFT_CORE_WORKSPACE_H
#define NUMSIM_FFT_CORE_WORKSPACE_H

#include "aligned_allocator.h"
#include "scalar_traits.h"

#include <any>
#include <complex>
#include <cstddef>
#include <vector>

namespace numsim::fft {

/**
 * @brief Reusable temporaries for plan execution.
 *
 * Plans are immutable and shared; everything they need per call (kernel
 * scratch per work chunk, the input copy of an r2c backward transform, the
 * temporaries and MPI buffers of distributed plans) is borrowed from a
 * workspace. Buffers only grow, so after the first call with a given plan
 * no further allocation happens. Without an explicit workspace a plan uses
 * a temporary one.
 *
 * A workspace is not thread-safe: use one per thread. Several plans may
 * share one workspace in sequence.
 */
template <real_scalar T> class workspace {
public:
  using size_type = std::size_t;
  using complex_type = std::complex<T>;
  using buffer_type = std::vector<T, aligned_allocator<T>>;

  /// Buffer `slot` with room for `count` values of S (T or std::complex<T>).
  template <typename S> S *buffer(size_type slot, size_type count) {
    static_assert(std::is_same_v<S, T> || std::is_same_v<S, complex_type>);
    if (_buffers.size() <= slot)
      _buffers.resize(slot + 1);
    return grow<S>(_buffers[slot], count);
  }

  /// Makes room for `chunks` concurrent chunk buffers (call before bulk()).
  void prepare_chunks(size_type chunks) {
    if (_chunks.size() < chunks)
      _chunks.resize(chunks);
  }

  /// Scratch of chunk `chunk` (distinct chunks may be used concurrently).
  complex_type *chunk_scratch(size_type chunk, size_type count) {
    return grow<complex_type>(_chunks[chunk], count);
  }

  /**
   * @brief Typed object in `slot` (e.g. a temporary field). The object is
   * built with `make()` on first use or when `valid(object)` is false.
   */
  template <typename Obj, typename Make, typename Valid>
  Obj &object(size_type slot, Make &&make, Valid &&valid) {
    if (_objects.size() <= slot)
      _objects.resize(slot + 1);
    if (Obj *o{std::any_cast<Obj>(&_objects[slot])}; o && valid(*o))
      return *o;
    _objects[slot] = make();
    return *std::any_cast<Obj>(&_objects[slot]);
  }

  /// Releases all memory.
  void clear() noexcept {
    _buffers.clear();
    _chunks.clear();
    _objects.clear();
  }

private:
  template <typename S> static S *grow(buffer_type &buf, size_type count) {
    size_type const scalars{count * (sizeof(S) / sizeof(T))};
    if (buf.size() < scalars)
      buf.resize(scalars);
    return reinterpret_cast<S *>(buf.data());
  }

  std::vector<buffer_type> _buffers;
  std::vector<buffer_type> _chunks;
  std::vector<std::any> _objects;
};

/// Slots used by the library's plans (users may use any slot >= user_base).
namespace workspace_slot {
inline constexpr std::size_t backward_copy{0};  ///< plan: r2c backward input copy
inline constexpr std::size_t transpose_a{1};    ///< distributed: pre/post-transpose field
inline constexpr std::size_t transpose_b{2};    ///< distributed: transposed field
inline constexpr std::size_t send{3};           ///< distributed: packed send buffer
inline constexpr std::size_t receive{4};        ///< distributed: packed receive buffer
inline constexpr std::size_t transpose_cache{5}; ///< distributed: MPI layouts
inline constexpr std::size_t user_base{16};
} // namespace workspace_slot

} // namespace numsim::fft

#endif // NUMSIM_FFT_CORE_WORKSPACE_H
