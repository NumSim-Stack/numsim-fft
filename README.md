# numsim-fft

Generic, header-only C++23 FFT library for fields of
[tmech](https://github.com/petlenz/tmech) tensors on regular 1D, 2D and 3D
grids. It targets spectral solvers in mechanics, such as FFT-based
homogenisation.

- **Own FFT kernels:**
  - mixed-radix Stockham autosort (specialised radix 2/3/4/5, generic odd radices ≤ 31);
  - Bluestein for larger prime factors, so every length N ≥ 1 works;
  - `float`, `double` and `long double`.
- **Transforms per axis:**
  - periodic complex (c2c);
  - real ↔ Hermitian half spectrum (r2c / c2r);
  - DCT/DST I–IV, in FFTW's REDFT/RODFT conventions;
  - `identity` (leave an axis untransformed).
- **Grid values:** scalars and tmech rank-1, rank-2 and rank-4 tensors. At each
  point the tensor is a tmech view, so `sigma[i,j,k] = tmech::dcontract(C, eps[i,j,k])`
  works in place.
- **Shared memory:** a sequential, an OpenMP or an HPX executor.
- **Distributed memory:** MPI slab decomposition through [MPL](https://github.com/rabauke/mpl).
  It combines with any executor inside a rank.

## Requirements

| | |
|---|---|
| Compiler | GCC ≥ 13 or Clang ≥ 18 (C++23) |
| CMake | ≥ 3.25 |
| Required | tmech (sibling checkout `../tmech`, installed, or fetched) |
| Fallbacks (always available; the headers pick `std::` when the consuming compiler has it) | kokkos/mdspan for `std::mdspan`, tl::expected for `std::expected` (Clang 18 with libstdc++ 13 hides it) |
| Optional | OpenMP, HPX ≥ 1.11, MPI (+ MPL, fetched if not installed), GoogleTest (tests) |

Each dependency is found with `find_package` when installed. If it isn't,
FetchContent downloads and builds it.

## Building

```sh
cmake --preset gcc-release          # or gcc-debug, clang-debug, clang-release
cmake --build --preset gcc-release
ctest --preset gcc-release

cmake --preset gcc-mpi              # + MPI; tests run on 1..4 ranks
cmake --preset gcc-hpx              # + HPX executor (first build compiles HPX)
cmake --preset gcc-all              # MPI + HPX
```

| Option | Default | |
|---|---|---|
| `NUMSIM_FFT_BUILD_TESTS` / `_EXAMPLES` | ON if top level | |
| `NUMSIM_FFT_SANITIZER_TESTS` | ON if top level | ASan+UBSan (and TSan) test variants |
| `NUMSIM_FFT_ENABLE_OPENMP` | ON | `openmp_executor` |
| `NUMSIM_FFT_ENABLE_HPX` | OFF | `hpx_executor` |
| `NUMSIM_FFT_ENABLE_MPI` | OFF | `distributed_plan` |
| `NUMSIM_FFT_TMECH_DIR` | `../tmech` | tmech checkout used via `add_subdirectory` |

**HPX build notes:**
- When HPX is not installed, it is built from source (v1.11.0) as a
  single-locality build: networking off, system allocator.
- The distributed runtime stays enabled, because HPX 1.11.0's `hpx_init`
  does not compile without it.
- Expect a long first build.

**Using numsim-fft from another project:**

```cmake
find_package(numsim-fft CONFIG REQUIRED)   # or add_subdirectory / FetchContent
target_link_libraries(app PRIVATE numsim-fft::numsim-fft)
```

## Usage

```cpp
#include <numsim-fft/numsim_fft.h>
using namespace numsim_fft;

using tensor2  = tmech::tensor<double, 3, 2>;
using ctensor2 = tmech::tensor<std::complex<double>, 3, 2>;

field<tensor2, 3> eps{extents{64, 64, 64}};          // zero-initialised
eps[1, 2, 3] = tmech::sym(grad);                     // point views are tmech tensors
eps[1, 2, 3](0, 1) = 0.5;                            // mutable components

// reusable plan: real -> half spectrum, periodic on every axis
auto plan = make_r2c_plan<double>(eps.extents()).value();
field<ctensor2, 3> eps_hat{plan.spectral_extents()}; // 64 x 64 x 33
plan.forward(eps, eps_hat);                          // returns expected<void, error>
for (std::size_t p = 0; p < eps_hat.size(); ++p)
  eps_hat.point(p) = tmech::eval(tmech::dcontract(C, eps_hat.point(p)));
plan.backward(eps_hat, eps, openmp_executor{});      // any executor

// in loops: a workspace keeps all temporaries, no allocation after the first call
workspace<double> ws;
for (int it = 0; it < 1000; ++it) {
  plan.forward(eps, eps_hat, openmp_executor{}, ws).value();
  plan.backward(eps_hat, eps, openmp_executor{}, ws).value();
}

// mixed axis kinds: DCT-II along x, periodic along y, DST-I along z
auto p2 = make_r2c_plan<double>(extents{32, 32, 16},
                                {axis_kind::dct2, axis_kind::periodic, axis_kind::dst1});

// one-shot helpers
auto X = rfft(eps).value();
auto x = irfft(X, eps.extents()).value();
```

For spectral derivatives, `wave_number(kind, k, n, h)`, `wave_vector(plan, index, spacing)`
and `is_nyquist(k, n)` give the wave numbers of every axis kind; see
[`examples/spectral_strain.cpp`](examples/spectral_strain.cpp).

### Distributed (MPI)

```cpp
auto const& comm = mpl::environment::comm_world();
auto plan = make_distributed_r2c_plan<double>(comm, extents{256, 256, 256}).value();
field<tensor2, 3>  eps{plan.local_physical_extents()};   // rows physical_offset() ..
field<ctensor2, 3> eps_hat{plan.local_spectral_extents()}; // cols spectral_offset() ..
plan.forward(eps, eps_hat, openmp_executor{});           // hybrid MPI + OpenMP
```

- **Physical space:** split along axis 0.
- **Spectral space:** split along axis 1, kept in natural axis order, so no
  transpose back is needed. Wrap an existing `MPI_Comm` with `mpl::mpi_communicator`.
- **Collectives:** `create`, `forward`, `backward` and the destructor are
  collective (the plan duplicates the communicator and is move-only). Errors
  are agreed on by all ranks, so a mismatch on one rank is returned on every rank.

## Conventions

| | |
|---|---|
| Memory layout | row-major points, the components of a point adjacent (AoS): `[i0][i1][i2][component]` |
| Periodic | forward `X_k = Σ x_j e^{-2πi jk/N}`, backward with `+` |
| r2c | the **last periodic axis** holds `n/2+1` values. c2r ignores the imaginary parts of the DC and Nyquist entries |
| r2r | FFTW REDFT00/10/01/11 and RODFT00/10/01/11, unnormalised. Inverse pairs are I↔I, II↔III and IV↔IV (`inverse_kind`); the plan's `backward` applies them |
| Normalisation | `plan_options{.norm = …}`. `backward` (default) scales the backward transform by 1/N, `ortho` scales both by 1/√N, `none` scales neither. N = ∏ `logical_size(kind, n)`: n for periodic, 2(n−1) for DCT-I, 2(n+1) for DST-I, 2n otherwise |
| Errors | plan creation and execution return `expected<…, error>`; `to_string(error)` |

## Design

- **Kernels** (`kernel/`) transform *vector batches*: B adjacent lines at
  once, element j of line b at `ptr[j*stride + b]`. For a tensor field, the
  batch covers the components and every faster grid axis. Inside the kernel
  the data are split-complex (separate real and imaginary arrays), so the
  innermost butterfly loop is plain arithmetic on independent arrays and
  vectorises. `benchmark/` (Google Benchmark, `NUMSIM_FFT_BUILD_BENCHMARK=ON`)
  measures kernel GFlop/s and whole-field throughput.
- **Plans** (`transform/plan.h`) run one pass per axis.
  - Each pass is split into (slab, block) work items for the executor, and
    every chunk owns its scratch.
  - The partition doesn't depend on the executor, so all executors give
    bitwise identical results.
  - Block size: about 64 KiB of scratch per block, or `plan_options::max_batch`.
- **r2c:** r2c runs first along the half-spectrum axis. The c2c and r2r
  passes then run on the complex data; a complex batch is a real batch of
  twice the width. `backward` works on a copy of its input and never modifies it.
- **Workspaces** (`workspace<T>`) hold kernel scratch per chunk, the r2c
  backward copy, and the distributed temporaries, pack buffers and MPI
  layouts. One workspace per thread; plans stay immutable and shareable.
  Without one, a call uses a temporary workspace (allocates).
- **Executors** model `executor`: `concurrency()` and `bulk(n, f)`.
  Exceptions from work items are rethrown on the caller.
- **Distributed plans** are a local plan (`identity` on axis 0), one
  `MPI_Alltoallv` transpose, and an axis-0 plan (`identity` elsewhere).

## Tests

- GoogleTest, written test-first.
- Every test is also built and run under AddressSanitizer + UndefinedBehaviorSanitizer
  (`<name>_asan`), and the threaded tests under ThreadSanitizer (`<name>_tsan`);
  `ctest` runs all variants. `NUMSIM_FFT_SANITIZER_TESTS=OFF` disables them.
  The HPX test is excluded (HPX needs its own sanitizer build).
- References are naive O(N²) transforms in `long double`
  (`tests/common/reference_dft.h`), which are themselves checked against
  closed forms.
- Kernels are tested for N = 1..64, primes, 1000, 1024 and 15015 in
  float/double/long double.
- n-D plans are tested against separable references, including tensor
  fields and FFT(C:ε) = C:FFT(ε).
- The distributed plan is tested against the serial plan on 1–4 ranks,
  including uneven splits and ranks without data.

## Roadmap

- Pencil decomposition (2D process grids via MPL cartesian communicators),
  distributed 1D transforms, and HPX distributed (parcelport) backend.
- Radix-8/16 butterflies and explicit SIMD; a destructive c2r variant that
  avoids the input copy.
- Lippmann–Schwinger / Moulinec–Suquet solver on top of the field and plan API.
