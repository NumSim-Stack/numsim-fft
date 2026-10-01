cmake_minimum_required(VERSION 3.25)

# The version lives in include/numsim-fft/config.h.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/numsim-fft/config.h" _numsim_fft_version_lines
     REGEX "#define NUMSIM_FFT_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(_line IN LISTS _numsim_fft_version_lines)
    if(_line MATCHES "#define NUMSIM_FFT_VERSION_(MAJOR|MINOR|PATCH) +([0-9]+)")
        set(_numsim_fft_version_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
    endif()
endforeach()

project(numsim-fft
    VERSION ${_numsim_fft_version_MAJOR}.${_numsim_fft_version_MINOR}.${_numsim_fft_version_PATCH}
    DESCRIPTION "Generic, header-only C++23 FFT library for tmech tensor fields"
    LANGUAGES CXX)

include(FetchContent)

# Default C++ standard for tests / examples in this tree. The library's own
# C++23 requirement is an INTERFACE compile feature on the target, so
# consumers inherit it through the linked target.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NUMSIM_FFT_IS_TOP_LEVEL ON)
else()
    set(NUMSIM_FFT_IS_TOP_LEVEL OFF)
endif()

# Build options
option(NUMSIM_FFT_BUILD_TESTS      "Build the numsim-fft test suite"        ${NUMSIM_FFT_IS_TOP_LEVEL})
option(NUMSIM_FFT_BUILD_EXAMPLES   "Build the numsim-fft examples"          ${NUMSIM_FFT_IS_TOP_LEVEL})
option(NUMSIM_FFT_BUILD_BENCHMARK  "Build the numsim-fft benchmarks"        OFF)
option(NUMSIM_FFT_SANITIZER_TESTS  "Also build and run the tests under ASan+UBSan (and TSan where threads are involved)" ${NUMSIM_FFT_IS_TOP_LEVEL})
option(NUMSIM_FFT_INSTALL_LIBRARY  "Install the numsim-fft library"         ${NUMSIM_FFT_IS_TOP_LEVEL})
option(NUMSIM_FFT_ENABLE_OPENMP    "Enable the OpenMP executor"             ON)
option(NUMSIM_FFT_ENABLE_HPX       "Enable the HPX (local) executor"        OFF)
option(NUMSIM_FFT_ENABLE_MPI       "Enable distributed transforms via MPI"  OFF)
set(NUMSIM_FFT_TMECH_DIR "${PROJECT_SOURCE_DIR}/../tmech" CACHE PATH
    "Local tmech checkout; used via add_subdirectory when it exists")

# numsim-fft is header-only: an INTERFACE target carrying include paths,
# the C++23 requirement and the optional backend dependencies.
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# GNUInstallDirs BEFORE target_include_directories(): CMAKE_INSTALL_INCLUDEDIR
# is expanded when the line is read (see numsim-core/CMakeLists.txt).
include(GNUInstallDirs)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

include(cmake/dependencies.cmake)

# Installation
if(NUMSIM_FFT_INSTALL_LIBRARY)
    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_Targets)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )
    install(
        EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )
endif()

if(NUMSIM_FFT_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()

if(NUMSIM_FFT_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
endif()

if(NUMSIM_FFT_BUILD_BENCHMARK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/CMakeLists.txt)
    add_subdirectory(benchmark)
endif()
