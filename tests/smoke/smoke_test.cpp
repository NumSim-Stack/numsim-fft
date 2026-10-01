#include <numsim-fft/numsim_fft.h>

#include <gtest/gtest.h>

TEST(smoke, version_is_defined) {
  EXPECT_EQ(NUMSIM_FFT_VERSION_MAJOR, 0);
  EXPECT_GE(NUMSIM_FFT_VERSION_MINOR, 1);
}

TEST(smoke, tmech_is_usable) {
  tmech::tensor<double, 3, 2> const I{tmech::eye<double, 3, 2>()};
  EXPECT_DOUBLE_EQ(tmech::trace(I), 3.0);
}
