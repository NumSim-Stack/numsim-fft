#include <numsim-fft/core/aligned_allocator.h>

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <vector>

using namespace numsim_fft;

TEST(aligned_allocator, vector_data_is_aligned) {
  for (std::size_t n : {1u, 3u, 17u, 1000u}) {
    std::vector<double, aligned_allocator<double>> v(n);
    auto const addr{reinterpret_cast<std::uintptr_t>(v.data())};
    EXPECT_EQ(addr % default_alignment, 0u) << "n = " << n;
  }
}

TEST(aligned_allocator, rebind_and_equality) {
  aligned_allocator<double> a;
  aligned_allocator<std::complex<double>> b{a};
  EXPECT_TRUE(a == aligned_allocator<double>{b});
  std::vector<std::complex<double>, decltype(b)> v(5, {1, 2}, b);
  EXPECT_EQ(v[4], (std::complex<double>{1, 2}));
}
