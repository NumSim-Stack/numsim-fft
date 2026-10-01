#ifndef NUMSIM_FFT_CORE_ALIGNED_ALLOCATOR_H
#define NUMSIM_FFT_CORE_ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <new>

namespace numsim::fft {

/// Alignment of field storage and scratch buffers: one cache line, which
/// also covers every SIMD width up to AVX-512.
inline constexpr std::size_t default_alignment{64};

/// Minimal standard allocator returning Alignment-aligned memory.
template <typename T, std::size_t Alignment = default_alignment>
class aligned_allocator {
  static_assert(Alignment >= alignof(T), "alignment weaker than alignof(T)");
  static_assert((Alignment & (Alignment - 1)) == 0, "alignment must be a power of two");

public:
  using value_type = T;

  template <typename U> struct rebind {
    using other = aligned_allocator<U, Alignment>;
  };

  constexpr aligned_allocator() noexcept = default;

  template <typename U>
  constexpr aligned_allocator(aligned_allocator<U, Alignment> const &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    if (n > static_cast<std::size_t>(-1) / sizeof(T))
      throw std::bad_array_new_length();
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T *p, std::size_t) noexcept {
    // Unsized form: Clang < 19 disables sized deallocation by default.
    ::operator delete(p, std::align_val_t{Alignment});
  }

  template <typename U>
  constexpr bool operator==(aligned_allocator<U, Alignment> const &) const noexcept {
    return true;
  }
};

} // namespace numsim::fft

#endif // NUMSIM_FFT_CORE_ALIGNED_ALLOCATOR_H
