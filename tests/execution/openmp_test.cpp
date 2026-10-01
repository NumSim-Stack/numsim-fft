#include <numsim-fft/execution/openmp.h>

#include <gtest/gtest.h>

using executor_types = testing::Types<numsim_fft::openmp_executor>;

#include "executor_tests.h"

TEST(openmp_executor, thread_count_can_be_fixed) {
  numsim_fft::openmp_executor const exec{.num_threads = 3};
  EXPECT_EQ(exec.concurrency(), 3u);
}
