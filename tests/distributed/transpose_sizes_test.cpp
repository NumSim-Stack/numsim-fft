// The MPI size check must be a pure function of the decomposition, so that
// every rank reaches the same verdict (otherwise some ranks skip the
// collective and the others hang), and it must bound the displacements
// (which MPI_Alltoallv takes as int element counts), not only the messages.

#include <numsim-fft/distributed/transpose_sizes.h>

#include <gtest/gtest.h>

#include <limits>

using namespace numsim_fft;

TEST(transpose_sizes, verdict_does_not_depend_on_the_rank) {
  slab_decomposition const rows{3, 2}, cols{3, 2};
  // rank 0 owns 2 rows and 2 cols, rank 1 owns 1 and 1: any per-rank
  // computation would differ; the pure function must not.
  auto const r{detail::transpose_sizes(rows, cols, 1)};
  EXPECT_EQ(r.largest_message, 2u * 2u);
  EXPECT_EQ(r.largest_local_total, 2u * 3u);
}

TEST(transpose_sizes, small_problem_fits) {
  EXPECT_TRUE(detail::transpose_fits_int(slab_decomposition{100, 4}, slab_decomposition{90, 4}, 81));
}

TEST(transpose_sizes, rejects_when_a_message_overflows_int) {
  std::size_t const big{static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1};
  EXPECT_FALSE(detail::transpose_fits_int(slab_decomposition{1, 1}, slab_decomposition{1, 1}, big));
}

TEST(transpose_sizes, rejects_when_only_the_displacements_overflow_int) {
  // 8 ranks: each message is n0 * m1 * R < INT_MAX but a local slab
  // n0 * M1 * R (the largest displacement) is not.
  std::size_t const R{1u << 20};
  slab_decomposition const rows{8 * 200, 8}, cols{8 * 200, 8}; // n0 = m1 = 200
  // message: 200*200*2^20 = 4.19e10 > INT_MAX already; scale down
  slab_decomposition const rows2{8 * 40, 8}, cols2{8 * 40, 8}; // n0 = m1 = 40
  auto const s{detail::transpose_sizes(rows2, cols2, R)};
  EXPECT_LT(s.largest_message, static_cast<std::size_t>(std::numeric_limits<int>::max()));
  EXPECT_GT(s.largest_local_total, static_cast<std::size_t>(std::numeric_limits<int>::max()));
  EXPECT_FALSE(detail::transpose_fits_int(rows2, cols2, R));
  (void)rows;
  (void)cols;
}
