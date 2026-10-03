// Point hooks of a plan: a load hook maps every point while the first pass
// of a forward transform reads it, a store hook sees every point right
// after the last pass of a backward transform wrote it. Results must equal
// mapping before / reducing after the plain transform, bit for bit.

#include <numsim-fft/numsim_fft.h>

#include "common/random_fields.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

using namespace numsim::fft;
using ak = axis_kind;
using tensor2 = tmech::tensor<double, 3, 2>;
using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;

namespace {

template <typename E, std::size_t D> field<E, D> random_field(extents<D> const &e, unsigned seed) {
  field<E, D> f{e};
  auto const v{test::random_vector<typename field<E, D>::scalar_type>(f.scalar_size(), seed)};
  std::ranges::copy(v, f.data());
  return f;
}

/// A point-dependent affine map of the C components: dst_c = (p % 7 + 1) src_c + c.
template <typename S> struct affine_map {
  std::size_t C;
  void operator()(std::size_t point, S const *src, S *dst) const {
    for (std::size_t c{0}; c < C; ++c)
      dst[c] = static_cast<double>(point % 7 + 1) * src[c] + static_cast<double>(c);
  }
};

template <typename E, std::size_t D, typename S>
field<E, D> mapped(field<E, D> const &f, affine_map<S> const &m) {
  field<E, D> g{f.extents()};
  for (std::size_t p{0}; p < f.size(); ++p)
    m(p, f.data() + p * f.components, g.data() + p * f.components);
  return g;
}

/// Store hook: weighted sum of the components per work item (summed in
/// item order afterwards) and a visit count per point.
template <typename S> struct weighted_sum {
  std::size_t C;
  std::vector<double> partial;
  std::vector<int> visits;
  void begin(std::size_t items) { partial.assign(items, 0.0); }
  void operator()(std::size_t item, std::size_t point, S const *v) {
    ++visits[point];
    for (std::size_t c{0}; c < C; ++c)
      partial[item] += static_cast<double>(point % 5 + 1) * std::real(v[c]);
  }
  double total() const {
    double s{0};
    for (double p : partial)
      s += p;
    return s;
  }
};

template <typename E, std::size_t D> double weighted_reference(field<E, D> const &f) {
  double s{0};
  for (std::size_t p{0}; p < f.size(); ++p)
    for (std::size_t c{0}; c < f.components; ++c)
      s += static_cast<double>(p % 5 + 1) * std::real(f.data()[p * f.components + c]);
  return s;
}

template <typename Exec> double check_r2c(Exec const &exec) {
  extents<3> const e{6, 5, 8};
  auto const p{make_r2c_plan<double>(e).value()};
  auto const in{random_field<tensor2>(e, 3)};
  affine_map<double> const map{9};
  field<ctensor2, 3> plain{p.spectral_extents()}, hooked{p.spectral_extents()};
  workspace<double> ws;
  EXPECT_TRUE(p.forward(mapped(in, map), plain, exec, ws).has_value());
  EXPECT_TRUE(p.forward(in, hooked, exec, ws, load_hook{map}).has_value());
  for (std::size_t i{0}; i < plain.scalar_size(); ++i)
    EXPECT_EQ(hooked.data()[i], plain.data()[i]) << i;

  field<tensor2, 3> back_plain{e}, back_hooked{e};
  auto spec{plain};
  EXPECT_TRUE(p.backward(plain, back_plain, exec, ws).has_value());
  weighted_sum<double> sum{9, {}, std::vector<int>(e.size(), 0)};
  EXPECT_TRUE(p.backward_destructive(spec, back_hooked, exec, ws, store_hook{sum}).has_value());
  for (std::size_t i{0}; i < back_plain.scalar_size(); ++i)
    EXPECT_EQ(back_hooked.data()[i], back_plain.data()[i]) << i;
  for (int v : sum.visits)
    EXPECT_EQ(v, 1);
  EXPECT_NEAR(sum.total(), weighted_reference(back_plain),
              1e-10 * std::abs(weighted_reference(back_plain)));
  return sum.total();
}

} // namespace

TEST(plan_hooks, r2c_tensor_field_sequential) { check_r2c(sequential_executor{}); }

#if defined(NUMSIM_FFT_HAS_OPENMP)
TEST(plan_hooks, r2c_tensor_field_openmp_and_executor_independent_reductions) {
  double const sequential{check_r2c(sequential_executor{})};
  EXPECT_EQ(check_r2c(openmp_executor{.num_threads = 3}), sequential);
  EXPECT_EQ(check_r2c(openmp_executor{.num_threads = 4}), sequential);
}
#endif

TEST(plan_hooks, c2c_and_r2r_with_identity_and_mixed_axes) {
  extents<3> const e{4, 7, 6};
  {
    std::array<ak, 3> const kinds{ak::identity, ak::periodic, ak::periodic};
    auto const p{make_c2c_plan<double>(e, kinds).value()};
    auto const in{random_field<ctensor2>(e, 5)};
    affine_map<std::complex<double>> const map{9};
    field<ctensor2, 3> plain{e}, hooked{e};
    ASSERT_TRUE(p.forward(mapped(in, map), plain).has_value());
    workspace<double> ws;
    ASSERT_TRUE(p.forward(in, hooked, sequential_executor{}, ws, load_hook{map}).has_value());
    for (std::size_t i{0}; i < plain.scalar_size(); ++i)
      ASSERT_EQ(hooked.data()[i], plain.data()[i]);
    field<ctensor2, 3> back{e}, back_hooked{e};
    ASSERT_TRUE(p.backward(plain, back).has_value());
    weighted_sum<std::complex<double>> sum{9, {}, std::vector<int>(e.size(), 0)};
    ASSERT_TRUE(
        p.backward(plain, back_hooked, sequential_executor{}, ws, store_hook{sum}).has_value());
    for (int v : sum.visits)
      ASSERT_EQ(v, 1);
    EXPECT_NEAR(sum.total(), weighted_reference(back), 1e-10 * std::abs(weighted_reference(back)));
  }
  {
    std::array<ak, 3> const kinds{ak::dct2, ak::dst1, ak::dct4};
    auto const p{make_r2r_plan<double>(e, kinds).value()};
    auto const in{random_field<tensor2>(e, 7)};
    affine_map<double> const map{9};
    field<tensor2, 3> plain{e}, hooked{e};
    ASSERT_TRUE(p.forward(mapped(in, map), plain).has_value());
    workspace<double> ws;
    ASSERT_TRUE(p.forward(in, hooked, sequential_executor{}, ws, load_hook{map}).has_value());
    for (std::size_t i{0}; i < plain.scalar_size(); ++i)
      ASSERT_EQ(hooked.data()[i], plain.data()[i]);
  }
  {
    // identity only: the hooks still see every point
    std::array<ak, 3> const kinds{ak::identity, ak::identity, ak::identity};
    auto const p{make_c2c_plan<double>(e, kinds).value()};
    auto const in{random_field<ctensor2>(e, 9)};
    affine_map<std::complex<double>> const map{9};
    field<ctensor2, 3> plain{e}, hooked{e};
    ASSERT_TRUE(p.forward(mapped(in, map), plain).has_value());
    workspace<double> ws;
    ASSERT_TRUE(p.forward(in, hooked, sequential_executor{}, ws, load_hook{map}).has_value());
    for (std::size_t i{0}; i < plain.scalar_size(); ++i)
      ASSERT_EQ(hooked.data()[i], plain.data()[i]);
  }
}

TEST(plan_hooks, are_rejected_where_a_point_is_not_contiguous) {
  // r2r axes of a complex transform run on the real and imaginary parts
  extents<2> const e{5, 6};
  std::array<ak, 2> const kinds{ak::dct2, ak::periodic};
  auto const p{make_c2c_plan<double>(e, kinds).value()};
  auto const in{random_field<std::complex<double>>(e, 1)};
  field<std::complex<double>, 2> out{e};
  workspace<double> ws;
  affine_map<std::complex<double>> const map{1};
  auto const r{p.forward(in, out, sequential_executor{}, ws, load_hook{map})};
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error(), error::hooks_unsupported);
}
