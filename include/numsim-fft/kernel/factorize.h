#ifndef NUMSIM_FFT_KERNEL_FACTORIZE_H
#define NUMSIM_FFT_KERNEL_FACTORIZE_H

#include <cstddef>
#include <vector>

namespace numsim_fft::kernel {

/// Largest prime handled by a direct (generic O(R^2)) butterfly; lengths
/// with a larger prime factor go through Bluestein's algorithm.
inline constexpr std::size_t max_direct_radix{31};

/**
 * @brief Radix sequence for the mixed-radix Stockham kernel.
 *
 * Factors of 4 first (cheapest butterfly per point), then a remaining 2,
 * then the odd primes in ascending order. factorize(1) is empty.
 */
inline std::vector<std::size_t> factorize(std::size_t n) {
  std::vector<std::size_t> radices;
  while (n % 4 == 0) {
    radices.push_back(4);
    n /= 4;
  }
  if (n % 2 == 0) {
    radices.push_back(2);
    n /= 2;
  }
  for (std::size_t p{3}; p * p <= n; p += 2) {
    while (n % p == 0) {
      radices.push_back(p);
      n /= p;
    }
  }
  if (n > 1)
    radices.push_back(n);
  return radices;
}

inline std::size_t largest_prime_factor(std::size_t n) {
  std::size_t largest{1};
  for (std::size_t p{2}; p * p <= n; ++p) {
    while (n % p == 0) {
      largest = p;
      n /= p;
    }
  }
  return n > 1 ? n : largest;
}

inline bool needs_bluestein(std::size_t n) {
  return largest_prime_factor(n) > max_direct_radix;
}

/// Smallest power of two >= n.
constexpr std::size_t next_power_of_two(std::size_t n) noexcept {
  std::size_t p{1};
  while (p < n)
    p <<= 1;
  return p;
}

} // namespace numsim_fft::kernel

#endif // NUMSIM_FFT_KERNEL_FACTORIZE_H
