# Third-party dependencies of numsim-fft.
#
# Every dependency follows the same rule: use an installed package when one
# is found, otherwise fetch it (FetchContent_Declare(... FIND_PACKAGE_ARGS),
# CMake >= 3.24). tmech additionally prefers a sibling checkout. Fetched and
# added projects are SYSTEM (CMake >= 3.25): their headers do not trip the
# warnings of our -Werror builds.

# ---------------------------------------------------------------------------
# tmech (required)
# ---------------------------------------------------------------------------
if(NOT TARGET tmech::tmech)
    # tmech's options are plain option() calls under an old policy level
    # (CMP0077 OLD), so normal variables would be ignored: use cache entries.
    set(TMECH_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
    set(TMECH_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(TMECH_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    # Our exported target links tmech::tmech, so tmech has to be in an
    # export set whenever we install.
    set(TMECH_INSTALL_LIBRARY ${NUMSIM_FFT_INSTALL_LIBRARY} CACHE BOOL "" FORCE)

    if(EXISTS "${NUMSIM_FFT_TMECH_DIR}/CMakeLists.txt")
        message(STATUS "numsim-fft: using tmech from ${NUMSIM_FFT_TMECH_DIR}")
        add_subdirectory(${NUMSIM_FFT_TMECH_DIR} ${CMAKE_BINARY_DIR}/_deps/tmech-build SYSTEM)
    else()
        # No version argument: tmech's generated version file is broken
        # (missing '$' in write_basic_package_version_file).
        FetchContent_Declare(tmech
            GIT_REPOSITORY https://github.com/petlenz/tmech.git
            GIT_TAG        v1.1.1
            GIT_SHALLOW    TRUE
            SYSTEM
            FIND_PACKAGE_ARGS CONFIG)
        FetchContent_MakeAvailable(tmech)
    endif()
endif()
target_link_libraries(${PROJECT_NAME} INTERFACE tmech::tmech)

# ---------------------------------------------------------------------------
# mdspan fallback. core/mdspan.h picks std::mdspan when the *consuming*
# compiler's standard library has it (libstdc++ >= 14) and the Kokkos
# reference implementation otherwise, so the fallback is always made
# available: a header-only library must not bake its own compiler's feature
# set into the installed package. (std::expected has no fallback: it is part
# of the toolchain baseline, GCC >= 13 / Clang >= 19.)
# ---------------------------------------------------------------------------
set(MDSPAN_CXX_STANDARD 23 CACHE STRING "" FORCE)
FetchContent_Declare(mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    # 'stable' branch: <mdspan/mdspan.hpp>, namespace Kokkos. The last
    # release tag (mdspan-0.6.0) still injects into std::experimental.
    GIT_TAG        8989f70749e28f337e6f7aa210db88659dba6f2f
    SYSTEM
    FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(mdspan)
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
    if(NOT TARGET HPX::hpx)
        # Only relevant when HPX has to be built from source. The first
        # build takes a long time.
        set(HPX_WITH_FETCH_ASIO          ON     CACHE BOOL   "" FORCE)
        set(HPX_WITH_MALLOC              system CACHE STRING "" FORCE)
        # Single-locality build: networking off. The distributed runtime has
        # to stay on: HPX 1.11.0's hpx_init (linked into every executable)
        # does not compile without it.
        set(HPX_WITH_DISTRIBUTED_RUNTIME ON     CACHE BOOL   "" FORCE)
        set(HPX_WITH_NETWORKING          OFF    CACHE BOOL   "" FORCE)
        set(HPX_WITH_EXAMPLES            OFF    CACHE BOOL   "" FORCE)
        set(HPX_WITH_TESTS               OFF    CACHE BOOL   "" FORCE)
        set(HPX_WITH_CXX_STANDARD        23     CACHE STRING "" FORCE)
        FetchContent_Declare(HPX
            GIT_REPOSITORY https://github.com/TheHPXProject/hpx.git
            GIT_TAG        v1.11.0
            GIT_SHALLOW    TRUE
            SYSTEM
            FIND_PACKAGE_ARGS)
        # HPX rejects a preset CMAKE_CXX_STANDARD (it wants
        # HPX_WITH_CXX_STANDARD), so hide ours while it configures.
        set(_numsim_fft_saved_std ${CMAKE_CXX_STANDARD})
        unset(CMAKE_CXX_STANDARD)
        FetchContent_MakeAvailable(HPX)
        set(CMAKE_CXX_STANDARD ${_numsim_fft_saved_std})
    endif()
    target_link_libraries(${PROJECT_NAME} INTERFACE HPX::hpx)
    target_compile_definitions(${PROJECT_NAME} INTERFACE NUMSIM_FFT_HAS_HPX)
endif()

# ---------------------------------------------------------------------------
# MPI + MPL (optional, distributed transforms)
# ---------------------------------------------------------------------------
if(NUMSIM_FFT_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    if(NOT TARGET mpl::mpl)
        FetchContent_Declare(mpl
            GIT_REPOSITORY https://github.com/rabauke/mpl.git
            GIT_TAG        v0.4.0
            GIT_SHALLOW    TRUE
            SYSTEM
            FIND_PACKAGE_ARGS)
        FetchContent_MakeAvailable(mpl)
    endif()
    target_link_libraries(${PROJECT_NAME} INTERFACE mpl::mpl MPI::MPI_CXX)
    target_compile_definitions(${PROJECT_NAME} INTERFACE NUMSIM_FFT_HAS_MPI)
endif()

# ---------------------------------------------------------------------------
# GoogleTest (tests only)
# ---------------------------------------------------------------------------
if(NUMSIM_FFT_BUILD_TESTS)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
        FIND_PACKAGE_ARGS NAMES GTest)
    FetchContent_MakeAvailable(googletest)
endif()
