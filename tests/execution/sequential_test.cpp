#include <numsim-fft/execution/sequential.h>

#include <gtest/gtest.h>

using executor_types = testing::Types<numsim::fft::sequential_executor>;

#include "executor_tests.h"
