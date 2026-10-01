# Dependencies of numsim-fft, resolved by numsim_dependency (numsim-cmake):
# an existing target, FETCHCONTENT_SOURCE_DIR_<NAME>, a sibling checkout
# under NUMSIM_DEVEL_DIR, an installed package, or a fetch of the pinned
# version, in that order. Fetched/added projects are SYSTEM.

include(cmake/numsim_bootstrap.cmake)
include(NumSimDependency)
include(NumSimWarnings)

# ---------------------------------------------------------------------------
# tmech (required)
# ---------------------------------------------------------------------------
if(DEFINED NUMSIM_FFT_TMECH_DIR)
    message(DEPRECATION "numsim-fft: NUMSIM_FFT_TMECH_DIR is obsolete; sibling checkouts are found via NUMSIM_DEVEL_DIR, or set FETCHCONTENT_SOURCE_DIR_TMECH")
    if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_TMECH AND EXISTS "${NUMSIM_FFT_TMECH_DIR}/CMakeLists.txt")
        set(FETCHCONTENT_SOURCE_DIR_TMECH "${NUMSIM_FFT_TMECH_DIR}")
    endif()
endif()
set(NUMSIM_FFT_TMECH_TAG "7267e01897158076effe006ba6877d11b2a3172b" CACHE STRING
    "tmech version (tag or commit) to fetch")   # TODO: v1.2.0 once tagged
numsim_dependency(tmech
    TARGET tmech::tmech
    GIT_REPOSITORY https://github.com/petlenz/tmech.git
    GIT_TAG        ${NUMSIM_FFT_TMECH_TAG}
    SIBLING        tmech
    FIND_PACKAGE_ARGS CONFIG
    # our exported target links tmech, so a fetched tmech is exported along
    OPTIONS TMECH_BUILD_TESTS=OFF TMECH_BUILD_EXAMPLES=OFF TMECH_BUILD_BENCHMARK=OFF
            TMECH_INSTALL_LIBRARY=${NUMSIM_FFT_INSTALL_LIBRARY})
target_link_libraries(${PROJECT_NAME} INTERFACE tmech::tmech)

# ---------------------------------------------------------------------------
# mdspan fallback. core/mdspan.h picks std::mdspan when the *consuming*
# compiler's standard library has it (libstdc++ >= 14) and the Kokkos
# reference implementation otherwise, so the fallback is always available.
# (std::expected has no fallback: GCC >= 13 / Clang >= 19 is the baseline.)
# ---------------------------------------------------------------------------
numsim_dependency(mdspan
    TARGET mdspan::mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    # 'stable' branch: <mdspan/mdspan.hpp>, namespace Kokkos. The last
    # release tag (mdspan-0.6.0) still injects into std::experimental.
    GIT_TAG        8989f70749e28f337e6f7aa210db88659dba6f2f
    OPTIONS MDSPAN_CXX_STANDARD=23)
target_link_libraries(${PROJECT_NAME} INTERFACE mdspan::mdspan)

# ---------------------------------------------------------------------------
# OpenMP executor (optional)
# ---------------------------------------------------------------------------
if(NUMSIM_FFT_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(${PROJECT_NAME} INTERFACE NUMSIM_FFT_HAS_OPENMP)
    else()
        # The option defaults to ON; do not fail toolchains without OpenMP.
        message(WARNING "numsim-fft: OpenMP not found, openmp_executor disabled")
        set(NUMSIM_FFT_ENABLE_OPENMP OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# HPX executor (optional, shared memory / single locality for now)
# ---------------------------------------------------------------------------
if(NUMSIM_FFT_ENABLE_HPX)
    # HPX rejects a preset CMAKE_CXX_STANDARD (it wants HPX_WITH_CXX_STANDARD),
    # so hide ours while it configures. Single-locality build: networking off.
    # The distributed runtime has to stay on: HPX 1.11.0's hpx_init (linked
    # into every executable) does not compile without it.
    set(_numsim_fft_saved_std ${CMAKE_CXX_STANDARD})
    unset(CMAKE_CXX_STANDARD)
    numsim_dependency(HPX
        TARGET HPX::hpx
        GIT_REPOSITORY https://github.com/TheHPXProject/hpx.git
        GIT_TAG        v1.11.0
        OPTIONS HPX_WITH_FETCH_ASIO=ON HPX_WITH_MALLOC=system
                HPX_WITH_DISTRIBUTED_RUNTIME=ON HPX_WITH_NETWORKING=OFF
                HPX_WITH_EXAMPLES=OFF HPX_WITH_TESTS=OFF HPX_WITH_CXX_STANDARD=23)
    set(CMAKE_CXX_STANDARD ${_numsim_fft_saved_std})
    target_link_libraries(${PROJECT_NAME} INTERFACE HPX::hpx)
    target_compile_definitions(${PROJECT_NAME} INTERFACE NUMSIM_FFT_HAS_HPX)
endif()

# ---------------------------------------------------------------------------
# MPI + MPL (optional, distributed transforms)
# ---------------------------------------------------------------------------
if(NUMSIM_FFT_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    numsim_dependency(mpl
        TARGET mpl::mpl
        GIT_REPOSITORY https://github.com/rabauke/mpl.git
        GIT_TAG        v0.4.0
        FIND_PACKAGE_ARGS CONFIG
        OPTIONS BUILD_TESTING=OFF MPL_BUILD_EXAMPLES=OFF)
    target_link_libraries(${PROJECT_NAME} INTERFACE mpl::mpl MPI::MPI_CXX)
    target_compile_definitions(${PROJECT_NAME} INTERFACE NUMSIM_FFT_HAS_MPI)
endif()

# ---------------------------------------------------------------------------
# GoogleTest (tests only)
# ---------------------------------------------------------------------------
if(NUMSIM_FFT_BUILD_TESTS)
    include(NumSimGoogleTest)
    numsim_googletest()
endif()
