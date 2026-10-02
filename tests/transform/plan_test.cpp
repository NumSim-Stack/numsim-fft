#include <numsim-fft/transform/plan.h>

#include "common/random_fields.h"
#include "common/reference_nd.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

using namespace numsim::fft;
using ak = axis_kind;

namespace {

template <typename S, std::size_t D>
field<S, D> random_field(extents<D> const &e, unsigned seed = 5) {
  field<S, D> f{e};
  auto const v{test::random_vector<S>(f.scalar_size(), seed)};
  std::ranges::copy(v, f.data());
  return f;
}

template <typename S> std::vector<ref::complex> to_ref(std::span<S const> s) {
  std::vector<ref::complex> out;
  out.reserve(s.size());
  for (auto const &v : s) {
    if constexpr (test::is_complex<S>::value)
      out.emplace_back(v.real(), v.imag());
    else
      out.emplace_back(v, 0);
  }
  return out;
}

template <std::size_t D> std::array<ak, D> all_periodic() {
  std::array<ak, D> k;
  k.fill(ak::periodic);
  return k;
}

} // namespace

template <typename T> class nd_plan : public testing::Test {};
using real_types = testing::Types<float, double>;
TYPED_TEST_SUITE(nd_plan, real_types);

TYPED_TEST(nd_plan, c2c_matches_reference_1d_2d_3d) {
  using T = TypeParam;
  using C = std::complex<T>;
  {
    extents<1> const e{30};
    auto const p{make_c2c_plan<T>(e).value()};
    auto const in{random_field<C>(e)};
    field<C, 1> out{e};
    ASSERT_TRUE(p.forward(in, out).has_value());
    auto const expected{ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1,
                                        all_periodic<1>())};
    EXPECT_LT(test::relative_error(out.scalars(), expected),
              test::tolerance<T>(30));
  }
  {
    extents<2> const e{6, 10};
    auto const p{make_c2c_plan<T>(e).value()};
    auto const in{random_field<C>(e)};
    field<C, 2> out{e};
    ASSERT_TRUE(p.forward(in, out).has_value());
    auto const expected{ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1,
                                        all_periodic<2>())};
    EXPECT_LT(test::relative_error(out.scalars(), expected),
              test::tolerance<T>(60));
  }
  {
    extents<3> const e{4, 5, 6};
    auto const p{make_c2c_plan<T>(e).value()};
    auto const in{random_field<C>(e)};
    field<C, 3> out{e};
    ASSERT_TRUE(p.forward(in, out).has_value());
    auto const expected{ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1,
                                        all_periodic<3>())};
    EXPECT_LT(test::relative_error(out.scalars(), expected),
              test::tolerance<T>(120));
  }
}

TYPED_TEST(nd_plan, c2c_with_mixed_axis_kinds) {
  using T = TypeParam;
  using C = std::complex<T>;
  extents<3> const e{5, 6, 7};
  std::array<ak, 3> const kinds{ak::dct2, ak::periodic, ak::dst3};
  auto const p{make_c2c_plan<T>(e, kinds).value()};
  auto const in{random_field<C>(e)};
  field<C, 3> out{e};
  ASSERT_TRUE(p.forward(in, out).has_value());
  auto const expected{
      ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1, kinds)};
  EXPECT_LT(test::relative_error(out.scalars(), expected),
            test::tolerance<T>(210, 16));
}

TYPED_TEST(nd_plan, r2c_halves_last_periodic_axis_and_matches_reference) {
  using T = TypeParam;
  for (auto const &kinds :
       {std::array<ak, 3>{ak::periodic, ak::periodic, ak::periodic},
        std::array<ak, 3>{ak::periodic, ak::periodic, ak::dct1},
        std::array<ak, 3>{ak::dst2, ak::periodic, ak::dct4}}) {
    extents<3> const e{4, 7, 6};
    auto const p{make_r2c_plan<T>(e, kinds).value()};
    std::size_t axis{2};
    while (kinds[axis] != ak::periodic)
      --axis;
    ASSERT_EQ(p.r2c_axis(), axis);
    EXPECT_EQ(p.spectral_extents(), e.with_extent(axis, e[axis] / 2 + 1));
    auto const in{random_field<T>(e)};
    field<std::complex<T>, 3> out{p.spectral_extents()};
    ASSERT_TRUE(p.forward(in, out).has_value());
    auto const full{
        ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1, kinds)};
    auto const expected{
        ref::crop(full, e.as_array(), 1, axis, e[axis] / 2 + 1)};
    EXPECT_LT(test::relative_error(out.scalars(), expected),
              test::tolerance<T>(168, 16));
  }
}

TYPED_TEST(nd_plan, r2r_all_axes_matches_reference) {
  using T = TypeParam;
  extents<3> const e{3, 5, 4};
  std::array<ak, 3> const kinds{ak::dct1, ak::dst1, ak::dct4};
  auto const p{make_r2r_plan<T>(e, kinds).value()};
  auto const in{random_field<T>(e)};
  field<T, 3> out{e};
  ASSERT_TRUE(p.forward(in, out).has_value());
  auto const expected{
      ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1, kinds)};
  std::vector<ref::real> expected_re(expected.size());
  for (std::size_t i{0}; i < expected.size(); ++i)
    expected_re[i] = expected[i].real();
  EXPECT_LT(test::relative_error(out.scalars(), expected_re),
            test::tolerance<T>(60, 16));
}

TYPED_TEST(nd_plan, roundtrips_with_default_normalization) {
  using T = TypeParam;
  using C = std::complex<T>;
  extents<3> const e{5, 4, 9};
  std::array<ak, 3> const mixed{ak::dst4, ak::periodic, ak::periodic};
  {
    auto const p{make_c2c_plan<T>(e, mixed).value()};
    auto const in{random_field<C>(e)};
    field<C, 3> spec{e}, back{e};
    ASSERT_TRUE(p.forward(in, spec).has_value());
    ASSERT_TRUE(p.backward(spec, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(180, 16));
  }
  {
    auto const p{make_r2c_plan<T>(e, mixed).value()};
    auto const in{random_field<T>(e)};
    field<C, 3> spec{p.spectral_extents()};
    field<T, 3> back{e};
    ASSERT_TRUE(p.forward(in, spec).has_value());
    auto const spec_copy{spec};
    ASSERT_TRUE(p.backward(spec, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(180, 16));
    // backward leaves its input untouched
    EXPECT_EQ(test::relative_error(spec.scalars(), spec_copy.scalars()), 0.0L);
  }
  {
    std::array<ak, 3> const r2r_kinds{ak::dct2, ak::dst3, ak::dct1};
    auto const p{make_r2r_plan<T>(e, r2r_kinds).value()};
    auto const in{random_field<T>(e)};
    field<T, 3> spec{e}, back{e};
    ASSERT_TRUE(p.forward(in, spec).has_value());
    ASSERT_TRUE(p.backward(spec, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(180, 16));
  }
}

TYPED_TEST(
    nd_plan,
    destructive_backward_gives_the_same_result_and_may_overwrite_its_input) {
  using T = TypeParam;
  using C = std::complex<T>;
  extents<3> const e{5, 4, 9};
  std::array<ak, 3> const mixed{ak::dst4, ak::periodic, ak::periodic};
  auto const p{make_r2c_plan<T>(e, mixed).value()};
  auto const in{random_field<T>(e)};
  field<C, 3> spec{p.spectral_extents()};
  field<T, 3> back{e}, back_destructive{e};
  ASSERT_TRUE(p.forward(in, spec).has_value());
  ASSERT_TRUE(p.backward(spec, back).has_value());
  workspace<T> ws;
  ASSERT_TRUE(
      p.backward_destructive(spec, back_destructive, sequential_executor{}, ws)
          .has_value());
  EXPECT_EQ(test::relative_error(back_destructive.scalars(), back.scalars()),
            0.0L);
  // no field-sized temporary is needed when the input may be overwritten
  EXPECT_EQ(ws.buffer_size(workspace_slot::backward_copy), 0u);

  // c2c and r2r plans: identical to backward (nothing to destroy)
  auto const pc{make_c2c_plan<T>(e, mixed).value()};
  auto const cin{random_field<C>(e)};
  field<C, 3> cspec{e}, cback{e}, cback_destructive{e};
  ASSERT_TRUE(pc.forward(cin, cspec).has_value());
  ASSERT_TRUE(pc.backward(cspec, cback).has_value());
  ASSERT_TRUE(pc.backward_destructive(cspec, cback_destructive).has_value());
  EXPECT_EQ(test::relative_error(cback_destructive.scalars(), cback.scalars()),
            0.0L);
}

TYPED_TEST(nd_plan, r2c_roundtrip_1d_needs_no_temporary_and_2d_odd_sizes) {
  using T = TypeParam;
  for (std::size_t n : {1u, 2u, 9u}) {
    extents<1> const e{n};
    auto const p{make_r2c_plan<T>(e).value()};
    auto const in{random_field<T>(e)};
    field<std::complex<T>, 1> spec{p.spectral_extents()};
    field<T, 1> back{e};
    ASSERT_TRUE(p.forward(in, spec).has_value());
    ASSERT_TRUE(p.backward(spec, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(n));
  }
  extents<2> const e{7, 5};
  auto const p{make_r2c_plan<T>(e).value()};
  auto const in{random_field<T>(e)};
  field<std::complex<T>, 2> spec{p.spectral_extents()};
  field<T, 2> back{e};
  ASSERT_TRUE(p.forward(in, spec).has_value());
  ASSERT_TRUE(p.backward(spec, back).has_value());
  EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
            test::tolerance<T>(35));
}

TEST(nd_plan_options, normalization_modes) {
  using C = std::complex<double>;
  extents<2> const e{8, 6};
  auto const in{random_field<C>(e)};
  long double energy_in{0};
  for (auto v : in.scalars())
    energy_in += test::abs2(v);

  // ortho: unitary, energy preserving, roundtrip identity
  auto const ortho{
      make_c2c_plan<double>(e, {}, {.norm = normalization::ortho}).value()};
  field<C, 2> spec{e}, back{e};
  ASSERT_TRUE(ortho.forward(in, spec).has_value());
  long double energy_spec{0};
  for (auto v : spec.scalars())
    energy_spec += test::abs2(v);
  EXPECT_NEAR(static_cast<double>(energy_spec / energy_in), 1.0, 1e-13);
  ASSERT_TRUE(ortho.backward(spec, back).has_value());
  EXPECT_LT(test::relative_error(back.scalars(), in.scalars()), 1e-14L);

  // none: roundtrip gives N * x
  auto const none{
      make_c2c_plan<double>(e, {}, {.norm = normalization::none}).value()};
  ASSERT_TRUE(none.forward(in, spec).has_value());
  ASSERT_TRUE(none.backward(spec, back).has_value());
  std::vector<C> scaled(in.scalars().begin(), in.scalars().end());
  for (auto &v : scaled)
    v *= 48.0;
  EXPECT_LT(test::relative_error(back.scalars(), scaled), 1e-14L);
}

TEST(nd_plan_options, batch_blocking_does_not_change_the_result) {
  using C = std::complex<double>;
  extents<3> const e{6, 5, 4};
  std::array<ak, 3> const kinds{ak::periodic, ak::dct3, ak::periodic};
  auto const in{random_field<C>(e)};
  field<C, 3> reference{e}, blocked{e};
  ASSERT_TRUE(make_c2c_plan<double>(e, kinds)
                  .value()
                  .forward(in, reference)
                  .has_value());
  for (std::size_t max_batch : {1u, 3u, 7u}) {
    auto const p{
        make_c2c_plan<double>(e, kinds, {.max_batch = max_batch}).value()};
    ASSERT_TRUE(p.forward(in, blocked).has_value());
    EXPECT_LT(test::relative_error(blocked.scalars(), reference.scalars()),
              1e-15L)
        << max_batch;
  }
}

TEST(nd_plan_options, in_place_c2c) {
  using C = std::complex<double>;
  extents<2> const e{6, 6};
  auto const p{make_c2c_plan<double>(e).value()};
  auto f{random_field<C>(e)};
  auto const original{f};
  field<C, 2> out{e};
  ASSERT_TRUE(p.forward(original, out).has_value());
  ASSERT_TRUE(p.forward(f, f).has_value());
  EXPECT_LT(test::relative_error(f.scalars(), out.scalars()), 1e-15L);
}

TEST(nd_plan_errors, invalid_plans) {
  EXPECT_EQ(make_c2c_plan<double>(extents{4, 0}).error(), error::zero_extent);
  EXPECT_EQ(make_r2r_plan<double>(extents{4, 1}, {ak::dct2, ak::dct1}).error(),
            error::size_below_minimum);
  EXPECT_EQ(
      make_r2r_plan<double>(extents{4, 4}, {ak::dct2, ak::periodic}).error(),
      error::periodic_axis_in_r2r);
  EXPECT_EQ(make_r2c_plan<double>(extents{4, 4}, {ak::dct2, ak::dst2}).error(),
            error::no_periodic_axis);
}

TEST(nd_plan_errors, extents_mismatch_is_reported) {
  using C = std::complex<double>;
  auto const p{make_c2c_plan<double>(extents{4, 4}).value()};
  field<C, 2> const in{extents{4, 5}};
  field<C, 2> out{extents{4, 4}};
  EXPECT_EQ(p.forward(in, out).error(), error::extents_mismatch);
  auto const r{make_r2c_plan<double>(extents{4, 4}).value()};
  field<double, 2> const rin{extents{4, 4}};
  field<C, 2> wrong{extents{4, 4}}; // should be 4 x 3
  EXPECT_EQ(r.forward(rin, wrong).error(), error::extents_mismatch);
}

TEST(nd_plan_errors, error_messages) {
  EXPECT_FALSE(to_string(error::extents_mismatch).empty());
  EXPECT_FALSE(to_string(error::zero_extent).empty());
}

TYPED_TEST(nd_plan, identity_axes_are_not_transformed) {
  using T = TypeParam;
  using C = std::complex<T>;
  extents<3> const e{4, 5, 6};
  {
    std::array<ak, 3> const kinds{ak::identity, ak::periodic, ak::dct2};
    auto const p{make_c2c_plan<T>(e, kinds).value()};
    auto const in{random_field<C>(e)};
    field<C, 3> out{e}, back{e};
    ASSERT_TRUE(p.forward(in, out).has_value());
    auto const expected{
        ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1, kinds)};
    EXPECT_LT(test::relative_error(out.scalars(), expected),
              test::tolerance<T>(30, 16));
    ASSERT_TRUE(p.backward(out, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(30, 16));
  }
  {
    // r2c along axis 0: the later axes are identity
    std::array<ak, 3> const kinds{ak::periodic, ak::identity, ak::identity};
    auto const p{make_r2c_plan<T>(e, kinds).value()};
    EXPECT_EQ(p.r2c_axis(), 0u);
    auto const in{random_field<T>(e)};
    field<C, 3> out{p.spectral_extents()};
    field<T, 3> back{e};
    ASSERT_TRUE(p.forward(in, out).has_value());
    auto const full{
        ref::forward_nd(to_ref(in.scalars()), e.as_array(), 1, kinds)};
    EXPECT_LT(test::relative_error(out.scalars(),
                                   ref::crop(full, e.as_array(), 1, 0, 3)),
              test::tolerance<T>(4, 16));
    ASSERT_TRUE(p.backward(out, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(4, 16));
  }
  {
    std::array<ak, 3> const kinds{ak::dst1, ak::identity, ak::dct3};
    auto const p{make_r2r_plan<T>(e, kinds).value()};
    auto const in{random_field<T>(e)};
    field<T, 3> out{e}, back{e};
    ASSERT_TRUE(p.forward(in, out).has_value());
    ASSERT_TRUE(p.backward(out, back).has_value());
    EXPECT_LT(test::relative_error(back.scalars(), in.scalars()),
              test::tolerance<T>(30, 16));
  }
}

TEST(nd_plan_identity, all_identity_plan_copies_and_scales) {
  using C = std::complex<double>;
  extents<2> const e{3, 4};
  auto const in{random_field<C>(e)};
  field<C, 2> out{e};
  auto const p{make_c2c_plan<double>(e, {ak::identity, ak::identity}).value()};
  ASSERT_TRUE(p.forward(in, out).has_value());
  EXPECT_TRUE(std::ranges::equal(in.scalars(), out.scalars()));
  auto const ortho{make_c2c_plan<double>(e, {ak::identity, ak::identity},
                                         {.norm = normalization::ortho})
                       .value()};
  ASSERT_TRUE(
      ortho.backward(in, out).has_value()); // logical size 1: still a copy
  EXPECT_TRUE(std::ranges::equal(in.scalars(), out.scalars()));
}

TEST(nd_plan_identity, zero_extent_is_allowed_on_identity_axes_only) {
  using C = std::complex<double>;
  extents<3> const e{0, 4, 4};
  auto const p{
      make_r2c_plan<double>(e, {ak::identity, ak::periodic, ak::periodic})};
  ASSERT_TRUE(p.has_value());
  field<double, 3> const in{e};
  field<C, 3> out{p->spectral_extents()};
  EXPECT_TRUE(p->forward(in, out).has_value());
  EXPECT_EQ(make_c2c_plan<double>(e, {ak::periodic, ak::periodic, ak::periodic})
                .error(),
            error::zero_extent);
}

TEST(nd_plan_identity, empty_identity_axis_behind_a_transformed_axis) {
  // axis 0 is transformed but every line batch is empty (inner extent 0)
  using C = std::complex<double>;
  extents<2> const e{4, 0};
  auto const p{make_c2c_plan<double>(e, {ak::periodic, ak::identity}).value()};
  field<C, 2> const in{e};
  field<C, 2> out{e};
  EXPECT_TRUE(p.forward(in, out).has_value());
  EXPECT_TRUE(p.backward(in, out).has_value());
}
