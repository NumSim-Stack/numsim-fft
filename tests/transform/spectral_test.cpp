#include <numsim-fft/transform/api.h>
#include <numsim-fft/transform/spectral.h>

#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace numsim::fft;
using ak = axis_kind;

TEST(spectral, periodic_wave_numbers_are_signed) {
  // n = 8, spacing 1: 2 pi / 8 * {0, 1, 2, 3, 4, -3, -2, -1}
  double const w{2 * std::numbers::pi / 8};
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::periodic, 0, 8), 0.0);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::periodic, 3, 8), 3 * w);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::periodic, 4, 8), 4 * w);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::periodic, 5, 8), -3 * w);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::periodic, 7, 8), -1 * w);
  // spacing scales the wave number
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::periodic, 1, 8, 0.5), 2 * w);
}

TEST(spectral, nyquist_detection) {
  EXPECT_TRUE(is_nyquist(4, 8));
  EXPECT_FALSE(is_nyquist(3, 8));
  EXPECT_FALSE(is_nyquist(3, 7)); // odd n has no Nyquist mode
}

TEST(spectral, r2r_wave_numbers_follow_the_kind) {
  double const pi{std::numbers::pi};
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dct1, 2, 5), pi * 2 / 4);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dct2, 2, 5), pi * 2 / 5);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dct3, 2, 5), pi * 2.5 / 5);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dct4, 2, 5), pi * 2.5 / 5);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dst1, 2, 5), pi * 3 / 6);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dst2, 2, 5), pi * 3 / 5);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dst3, 2, 5), pi * 2.5 / 5);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::dst4, 2, 5), pi * 2.5 / 5);
  EXPECT_DOUBLE_EQ(wave_number<double>(ak::identity, 2, 5), 0.0);
}

TEST(spectral, wave_vector_of_a_spectral_grid_point) {
  // r2c plan on 8 x 6: spectral grid 8 x 4
  auto const p{make_r2c_plan<double>(extents{8, 6}).value()};
  std::array<double, 2> const spacing{0.25, 0.5};
  auto const k{wave_vector(p, {5, 3}, spacing)};
  EXPECT_DOUBLE_EQ(k[0], wave_number<double>(ak::periodic, 5, 8, 0.25));
  EXPECT_DOUBLE_EQ(k[1], wave_number<double>(ak::periodic, 3, 6, 0.5));
}

TEST(spectral, spectral_derivative_of_a_sine) {
  // d/dx sin(2 pi m x / L) = (2 pi m / L) cos(2 pi m x / L)
  std::size_t const n{32};
  double const L{3.0}, h{L / n};
  int const m{3};
  field<double, 1> u{extents<1>{n}};
  for (std::size_t j{0}; j < n; ++j)
    u[j] = std::sin(2 * std::numbers::pi * m * static_cast<double>(j) * h / L);
  auto const p{make_r2c_plan<double>(extents<1>{n}).value()};
  field<std::complex<double>, 1> U{p.spectral_extents()};
  ASSERT_TRUE(p.forward(u, U).has_value());
  for (std::size_t k{0}; k < U.size(); ++k)
    U[k] *= std::complex<double>{0, is_nyquist(k, n) ? 0.0 : wave_number<double>(ak::periodic, k, n, h)};
  field<double, 1> du{extents<1>{n}};
  ASSERT_TRUE(p.backward(U, du).has_value());
  for (std::size_t j{0}; j < n; ++j)
    EXPECT_NEAR(du[j], 2 * std::numbers::pi * m / L *
                           std::cos(2 * std::numbers::pi * m * static_cast<double>(j) * h / L),
                1e-12);
}
