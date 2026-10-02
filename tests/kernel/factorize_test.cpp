#include <numsim-fft/kernel/factorize.h>

#include <gtest/gtest.h>

#include <functional>
#include <numeric>

using namespace numsim::fft::kernel;

namespace {
std::size_t product(std::vector<std::size_t> const &f) {
  return std::accumulate(f.begin(), f.end(), std::size_t{1}, std::multiplies<>{});
}
} // namespace

TEST(factorize, product_is_n_for_all_small_n) {
  for (std::size_t n{1}; n <= 2000; ++n)
    EXPECT_EQ(product(factorize(n)), n) << n;
}

TEST(factorize, prefers_radix_8) {
  EXPECT_EQ(factorize(16), (std::vector<std::size_t>{8, 2}));
  EXPECT_EQ(factorize(32), (std::vector<std::size_t>{8, 4}));
  EXPECT_EQ(factorize(64), (std::vector<std::size_t>{8, 8}));
  EXPECT_EQ(factorize(1), (std::vector<std::size_t>{}));
  EXPECT_EQ(factorize(60), (std::vector<std::size_t>{4, 3, 5}));
  EXPECT_EQ(factorize(120), (std::vector<std::size_t>{8, 3, 5}));
}

TEST(factorize, largest_prime_factor) {
  EXPECT_EQ(largest_prime_factor(1), 1u);
  EXPECT_EQ(largest_prime_factor(1024), 2u);
  EXPECT_EQ(largest_prime_factor(15015), 13u);
  EXPECT_EQ(largest_prime_factor(2 * 101), 101u);
}

TEST(factorize, bluestein_only_for_large_primes) {
  EXPECT_FALSE(needs_bluestein(1024));
  EXPECT_FALSE(needs_bluestein(31 * 4));
  EXPECT_TRUE(needs_bluestein(37));
  EXPECT_TRUE(needs_bluestein(2 * 101));
}
