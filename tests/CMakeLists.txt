# Tests use numsim_add_test (numsim-cmake): every test is also built and run
# under ASan+UBSan (<name>_asan) and, where marked TSAN, under ThreadSanitizer
# (<name>_tsan); NUMSIM_SANITIZER_TESTS (default ON) controls it. Arguments
# after SOURCES are passed through (MPI_PROCS, NO_MAIN, NO_SANITIZERS, TSAN).
macro(add_numsim_fft_test TARGET_NAME)
    numsim_add_test(${TARGET_NAME} ${ARGN}
        LINK numsim-fft::numsim-fft
        INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/tests)
endmacro()

file(GLOB _test_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_dir IN LISTS _test_dirs)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/CMakeLists.txt)
        add_subdirectory(${_dir})
    endif()
endforeach()
