// Transforms of tmech tensor fields: every component is transformed on its
// own, and pointwise tensor algebra commutes with the transform.

#include <numsim-fft/transform/plan.h>

#include "common/random_fields.h"
#include "common/tolerances.h"

#include <gtest/gtest.h>

#include <complex>

using namespace numsim::fft;
using ak = axis_kind;

namespace {

template <typename E, std::size_t D> field<E, D> random_field(extents<D> const &e) {
  field<E, D> f{e};
  auto const v{test::random_vector<typename field<E, D>::scalar_type>(f.scalar_size(), 11)};
  std::ranges::copy(v, f.data());
  return f;
}

/// Component c of every point, as a scalar field.
template <typename E, std::size_t D>
auto component(field<E, D> const &f, std::size_t c) {
  field<typename field<E, D>::scalar_type, D> s{f.extents()};
  for (std::size_t p{0}; p < f.size(); ++p)
    s.data()[p] = f.data()[p * f.components + c];
  return s;
}

} // namespace

template <typename E> class tensor_field_transform : public testing::Test {};
using tensor_types =
    testing::Types<tmech::tensor<double, 2, 2>, tmech::tensor<double, 3, 2>,
                   tmech::tensor<double, 2, 4>, tmech::tensor<double, 3, 4>>;
TYPED_TEST_SUITE(tensor_field_transform, tensor_types);

TYPED_TEST(tensor_field_transform, r2c_equals_componentwise_scalar_transform) {
  using E = TypeParam;
  using CE = element_traits<E>::template rebind<std::complex<double>>;
  extents<3> const e{4, 3, 6};
  std::array<ak, 3> const kinds{ak::periodic, ak::dct2, ak::periodic};
  auto const p{make_r2c_plan<double>(e, kinds).value()};
  auto const in{random_field<E>(e)};
  field<CE, 3> out{p.spectral_extents()};
  ASSERT_TRUE(p.forward(in, out).has_value());
  for (std::size_t c{0}; c < in.components; ++c) {
    auto const scalar_in{component(in, c)};
    field<std::complex<double>, 3> scalar_out{p.spectral_extents()};
    ASSERT_TRUE(p.forward(scalar_in, scalar_out).has_value());
    EXPECT_LT(test::relative_error(component(out, c).scalars(), scalar_out.scalars()), 1e-15L)
        << "component " << c;
  }
  field<E, 3> back{e};
  ASSERT_TRUE(p.backward(out, back).has_value());
  EXPECT_LT(test::relative_error(back.scalars(), in.scalars()), 1e-14L);
}

TEST(tensor_field_transform, constant_stiffness_commutes_with_transform) {
  // FFT(C : eps) == C : FFT(eps) for a constant rank-4 tensor C.
  using tensor2 = tmech::tensor<double, 3, 2>;
  using tensor4 = tmech::tensor<double, 3, 4>;
  using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;
  using ctensor4 = tmech::tensor<std::complex<double>, 3, 4>;

  tmech::tensor<double, 3, 2> const I{tmech::eye<double, 3, 2>()};
  double const lambda{1.5}, mu{0.7};
  tensor4 const C{lambda * tmech::otimes(I, I) +
                  mu * (tmech::otimesu(I, I) + tmech::otimesl(I, I))};
  ctensor4 Cc;
  for (std::size_t i{0}; i < 81; ++i)
    Cc.raw_data()[i] = C.raw_data()[i];

  extents<3> const e{4, 4, 4};
  auto const eps{random_field<tensor2>(e)};
  field<tensor2, 3> sig{e};
  for (std::size_t p{0}; p < eps.size(); ++p)
    sig.point(p) = tmech::dcontract(C, eps.point(p));

  auto const plan{make_r2c_plan<double>(e).value()};
  field<ctensor2, 3> eps_hat{plan.spectral_extents()}, sig_hat{plan.spectral_extents()};
  ASSERT_TRUE(plan.forward(eps, eps_hat).has_value());
  ASSERT_TRUE(plan.forward(sig, sig_hat).has_value());

  field<ctensor2, 3> expected{plan.spectral_extents()};
  for (std::size_t p{0}; p < expected.size(); ++p)
    expected.point(p) = tmech::dcontract(Cc, eps_hat.point(p));
  EXPECT_LT(test::relative_error(sig_hat.scalars(), expected.scalars()), 1e-14L);
}
