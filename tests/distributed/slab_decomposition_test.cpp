#include <numsim-fft/distributed/slab_decomposition.h>

#include <gtest/gtest.h>

using numsim::fft::slab_decomposition;

TEST(slab_decomposition, even_split) {
  slab_decomposition const d{12, 4};
  for (std::size_t p{0}; p < 4; ++p) {
    EXPECT_EQ(d.local_size(p), 3u);
    EXPECT_EQ(d.offset(p), 3 * p);
  }
}

TEST(slab_decomposition, uneven_split_gives_the_first_parts_one_more) {
  slab_decomposition const d{10, 4};
  EXPECT_EQ(d.local_size(0), 3u);
  EXPECT_EQ(d.local_size(1), 3u);
  EXPECT_EQ(d.local_size(2), 2u);
  EXPECT_EQ(d.local_size(3), 2u);
  EXPECT_EQ(d.offset(3), 8u);
}

TEST(slab_decomposition, more_parts_than_points) {
  slab_decomposition const d{2, 3};
  EXPECT_EQ(d.local_size(0), 1u);
  EXPECT_EQ(d.local_size(1), 1u);
  EXPECT_EQ(d.local_size(2), 0u);
  EXPECT_EQ(d.offset(2), 2u);
}

TEST(slab_decomposition, parts_cover_the_range_exactly) {
  for (std::size_t n{0}; n < 20; ++n)
    for (std::size_t parts{1}; parts < 7; ++parts) {
      slab_decomposition const d{n, parts};
      std::size_t sum{0};
      for (std::size_t p{0}; p < parts; ++p) {
        EXPECT_EQ(d.offset(p), sum);
        sum += d.local_size(p);
      }
      EXPECT_EQ(sum, n);
    }
}
