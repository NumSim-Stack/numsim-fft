# Test helper:
#   add_numsim_fft_test(<name> SOURCES <src...> [MPI_PROCS <n...>] [NO_MAIN]
#                       [NO_SANITIZERS] [TSAN])
#
# Every test directory holds one executable. Tests link gtest_main unless
# NO_MAIN is given (MPI / HPX tests provide their own main).
#
# With NUMSIM_FFT_SANITIZER_TESTS the same sources are built again with
# AddressSanitizer + UndefinedBehaviorSanitizer (<name>_asan) and, if TSAN
# is given, with ThreadSanitizer (<name>_tsan). All variants are registered
# with ctest. NO_SANITIZERS skips this (HPX: its coroutines need HPX itself
# built with sanitizer support). TSan is opt-in because it reports false
# positives through uninstrumented runtimes such as libgomp and Open MPI.

set(NUMSIM_FFT_ASAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer
                          -fno-sanitize-recover=undefined)
set(NUMSIM_FFT_TSAN_FLAGS -fsanitize=thread -fno-omit-frame-pointer)

function(_numsim_fft_test_executable TARGET_NAME)
    cmake_parse_arguments(ARG "NO_MAIN" "" "SOURCES;FLAGS" ${ARGN})
    add_executable(${TARGET_NAME} ${ARG_SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_link_libraries(${TARGET_NAME} PRIVATE numsim-fft::numsim-fft GTest::gtest)
    if(NOT ARG_NO_MAIN)
        target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Werror>
        ${ARG_FLAGS})
    target_link_options(${TARGET_NAME} PRIVATE ${ARG_FLAGS})
endfunction()

function(_numsim_fft_register_test TARGET_NAME SUFFIX)
    cmake_parse_arguments(ARG "" "" "MPI_PROCS;ENVIRONMENT" ${ARGN})
    if(ARG_MPI_PROCS)
        foreach(np IN LISTS ARG_MPI_PROCS)
            add_test(NAME ${TARGET_NAME}_np${np}${SUFFIX}
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                             ${NUMSIM_FFT_MPIEXEC_PREFLAGS} ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
            set_tests_properties(${TARGET_NAME}_np${np}${SUFFIX} PROPERTIES
                                 PROCESSORS ${np} ENVIRONMENT "${ARG_ENVIRONMENT}")
        endforeach()
    else()
        gtest_discover_tests(${TARGET_NAME} DISCOVERY_TIMEOUT 120 TEST_SUFFIX "${SUFFIX}"
                             PROPERTIES ENVIRONMENT "${ARG_ENVIRONMENT}")
    endif()
endfunction()

function(add_numsim_fft_test TARGET_NAME)
    cmake_parse_arguments(ARG "NO_MAIN;NO_SANITIZERS;TSAN" "" "SOURCES;MPI_PROCS" ${ARGN})
    set(_main_arg)
    if(ARG_NO_MAIN)
        set(_main_arg NO_MAIN)
    endif()

    _numsim_fft_test_executable(${TARGET_NAME} ${_main_arg} SOURCES ${ARG_SOURCES})
    _numsim_fft_register_test(${TARGET_NAME} "" MPI_PROCS ${ARG_MPI_PROCS})

    if(NUMSIM_FFT_SANITIZER_TESTS AND NOT ARG_NO_SANITIZERS)
        # Open MPI leaks a little at shutdown; leak checks stay on elsewhere.
        set(_asan_env "UBSAN_OPTIONS=print_stacktrace=1")
        if(ARG_MPI_PROCS)
            list(APPEND _asan_env "ASAN_OPTIONS=detect_leaks=0")
        endif()
        _numsim_fft_test_executable(${TARGET_NAME}_asan ${_main_arg} SOURCES ${ARG_SOURCES}
                                    FLAGS ${NUMSIM_FFT_ASAN_FLAGS})
        _numsim_fft_register_test(${TARGET_NAME}_asan "_asan" MPI_PROCS ${ARG_MPI_PROCS}
                                  ENVIRONMENT ${_asan_env})
        if(ARG_TSAN)
            _numsim_fft_test_executable(${TARGET_NAME}_tsan ${_main_arg} SOURCES ${ARG_SOURCES}
                                        FLAGS ${NUMSIM_FFT_TSAN_FLAGS})
            if(NUMSIM_FFT_SETARCH)
                # libtsan (GCC 13) aborts with "unexpected memory mapping" under
                # the ASLR entropy of recent kernels; run without ASLR.
                set_target_properties(${TARGET_NAME}_tsan PROPERTIES
                                      CROSSCOMPILING_EMULATOR "${NUMSIM_FFT_SETARCH};${CMAKE_HOST_SYSTEM_PROCESSOR};-R")
            endif()
            _numsim_fft_register_test(${TARGET_NAME}_tsan "_tsan" MPI_PROCS ${ARG_MPI_PROCS}
                                      ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1")
        endif()
    endif()
endfunction()

# Open MPI needs --oversubscribe to run more ranks than cores; other MPIs
# reject the flag.
set(NUMSIM_FFT_MPIEXEC_PREFLAGS)
if(NUMSIM_FFT_ENABLE_MPI)
    execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version OUTPUT_VARIABLE _mpiexec_version
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_mpiexec_version MATCHES "Open MPI|OpenRTE|open-mpi|PRTE")
        set(NUMSIM_FFT_MPIEXEC_PREFLAGS --oversubscribe)
    endif()
endif()

find_program(NUMSIM_FFT_SETARCH setarch)
mark_as_advanced(NUMSIM_FFT_SETARCH)

if(NUMSIM_FFT_SANITIZER_TESTS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "numsim-fft: sanitizer tests need GCC or Clang; disabled")
    set(NUMSIM_FFT_SANITIZER_TESTS OFF)
endif()

file(GLOB _test_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_dir IN LISTS _test_dirs)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/CMakeLists.txt)
        add_subdirectory(${_dir})
    endif()
endforeach()
