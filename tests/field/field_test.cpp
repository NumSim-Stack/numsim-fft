#include <numsim-fft/field/field.h>

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <vector>

using namespace numsim_fft;

TEST(field, scalar_field_is_zero_initialised_and_indexable) {
  field<double, 2> f{extents{3, 4}};
  EXPECT_EQ(f.size(), 12u);
  EXPECT_EQ(f.scalar_size(), 12u);
  for (auto v : f.scalars())
    EXPECT_EQ(v, 0.0);
  f[1, 2] = 5.0;
  EXPECT_EQ(f.data()[1 * 4 + 2], 5.0);
  EXPECT_EQ((f[std::array<std::size_t, 2>{1, 2}]), 5.0);
}

TEST(field, storage_is_aligned) {
  field<tmech::tensor<double, 3, 2>, 3> f{extents{3, 3, 3}};
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(f.data()) % default_alignment, 0u);
}

TEST(field, rank2_tensor_points_are_contiguous_aos) {
  field<tmech::tensor<double, 3, 2>, 2> f{extents{2, 3}};
  EXPECT_EQ(f.components, 9u);
  EXPECT_EQ(f.scalar_size(), 54u);
  f[1, 2](0, 1) = 7.0; // mutable component access
  // point (1,2) -> linear 5, component (0,1) -> 1
  EXPECT_EQ(f.data()[5 * 9 + 1], 7.0);
  EXPECT_EQ((f[1, 2](0, 1)), 7.0);
}

TEST(field, tmech_expression_assigns_into_view) {
  using tensor2 = tmech::tensor<double, 3, 2>;
  field<tensor2, 3> f{extents{2, 2, 2}};
  tensor2 const I{tmech::eye<double, 3, 2>()};
  tensor2 A;
  A(0, 1) = 2.0;
  f[1, 0, 1] = I + tmech::trans(A);
  tensor2 const result{f[1, 0, 1]};
  EXPECT_EQ(result(0, 0), 1.0);
  EXPECT_EQ(result(1, 0), 2.0);
  EXPECT_EQ(result(0, 1), 0.0);
  // the views take part in tmech expressions
  EXPECT_DOUBLE_EQ(tmech::trace(f[1, 0, 1]), 3.0);
}

TEST(field, view_copy_assignment_copies_values_not_pointers) {
  using tensor2 = tmech::tensor<double, 2, 2>;
  field<tensor2, 1> f{extents<1>{2}};
  f[0](1, 1) = 3.0;
  f[1] = f[0];
  f[0](1, 1) = 4.0;
  EXPECT_EQ((f[1](1, 1)), 3.0);
}

TEST(field, rank4_tensor_views) {
  using tensor4 = tmech::tensor<double, 3, 4>;
  field<tensor4, 1> f{extents<1>{3}};
  tensor4 const IIsym{0.5 * (tmech::otimesu(tmech::eye<double, 3, 2>(), tmech::eye<double, 3, 2>()) +
                             tmech::otimesl(tmech::eye<double, 3, 2>(), tmech::eye<double, 3, 2>()))};
  f[2] = IIsym;
  tmech::tensor<double, 3, 2> eps;
  eps(0, 1) = 1.0;
  eps(1, 0) = 3.0;
  tmech::tensor<double, 3, 2> const sym_eps{tmech::dcontract(f[2], eps)};
  EXPECT_DOUBLE_EQ(sym_eps(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(sym_eps(1, 0), 2.0);
  EXPECT_EQ(f.data()[2 * 81 + 0], 1.0); // (0,0,0,0) of IIsym
}

TEST(field, complex_tensor_field) {
  using ctensor = tmech::tensor<std::complex<double>, 3, 2>;
  field<ctensor, 2> f{extents{2, 2}};
  f[0, 1](2, 2) = {1.0, -1.0};
  EXPECT_EQ(f.data()[1 * 9 + 8], (std::complex<double>{1.0, -1.0}));
}

TEST(field, fill_constructor_and_const_access) {
  using tensor2 = tmech::tensor<double, 2, 2>;
  tensor2 const I{tmech::eye<double, 2, 2>()};
  field<tensor2, 2> const f{extents{2, 2}, I};
  for (std::size_t i{0}; i < 2; ++i)
    for (std::size_t j{0}; j < 2; ++j)
      EXPECT_DOUBLE_EQ(tmech::trace(f[i, j]), 2.0);
}

TEST(field, copy_in_and_out) {
  using tensor2 = tmech::tensor<double, 2, 2>;
  std::vector<tensor2> values(6);
  for (std::size_t p{0}; p < 6; ++p)
    values[p](0, 1) = static_cast<double>(p);
  field<tensor2, 2> f{extents{2, 3}};
  f.assign(values);
  EXPECT_EQ((f[1, 1](0, 1)), 4.0);
  auto const out{f.to_vector()};
  ASSERT_EQ(out.size(), 6u);
  EXPECT_EQ(out[5](0, 1), 5.0);
}

TEST(field, linear_point_access) {
  field<tmech::tensor<float, 2, 2>, 2> f{extents{2, 2}};
  f.point(3)(1, 0) = 2.0f;
  EXPECT_EQ((f[1, 1](1, 0)), 2.0f);
}

TEST(field, rebind_to_complex) {
  using real_field = field<tmech::tensor<double, 3, 2>, 3>;
  using complex_field = real_field::rebind_scalar<std::complex<double>>;
  static_assert(std::is_same_v<complex_field::element_type,
                               tmech::tensor<std::complex<double>, 3, 2>>);
  static_assert(complex_field::spatial_dimension == 3);
  SUCCEED();
}

TEST(field, mdspan_view_has_trailing_component_extent) {
  field<tmech::tensor<double, 2, 2>, 2> f{extents{3, 5}};
  f[2, 4](1, 0) = 9.0;
  auto const m{f.mdspan()};
  static_assert(decltype(m)::rank() == 3);
  EXPECT_EQ(m.extent(0), 3u);
  EXPECT_EQ(m.extent(1), 5u);
  EXPECT_EQ(m.extent(2), 4u);
  EXPECT_EQ((m[2, 4, 2]), 9.0);
}
