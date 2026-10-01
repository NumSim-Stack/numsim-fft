#include <numsim-fft/execution/hpx.h>

#include <gtest/gtest.h>

#include <hpx/init.hpp>

using executor_types = testing::Types<numsim::fft::hpx_executor>;

#include "executor_tests.h"

TEST(hpx_executor, reports_hpx_worker_threads) {
  EXPECT_EQ(numsim::fft::hpx_executor{}.concurrency(),
            static_cast<std::size_t>(hpx::get_num_worker_threads()));
}

// The HPX runtime must be running while the tests execute.
namespace {
int hpx_main(int, char **) {
  int const result{RUN_ALL_TESTS()};
  hpx::finalize();
  return result;
}
} // namespace

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return hpx::init(hpx_main, argc, argv);
}
