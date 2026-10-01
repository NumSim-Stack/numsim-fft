# Test helper: add_numsim_fft_test(<name> SOURCES <src...> [MPI_PROCS <n...>] [NO_MAIN])
#
# Every test directory holds one executable. Tests link gtest_main unless
# NO_MAIN is given (MPI / HPX tests provide their own main).
function(add_numsim_fft_test TARGET_NAME)
    cmake_parse_arguments(ARG "NO_MAIN" "" "SOURCES;MPI_PROCS" ${ARGN})
    add_executable(${TARGET_NAME} ${ARG_SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_link_libraries(${TARGET_NAME} PRIVATE numsim-fft::numsim-fft GTest::gtest)
    if(NOT ARG_NO_MAIN)
        target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main)
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Werror>)
    if(ARG_MPI_PROCS)
        foreach(np IN LISTS ARG_MPI_PROCS)
            add_test(NAME ${TARGET_NAME}_np${np}
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                             --oversubscribe ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:${TARGET_NAME}> ${MPIEXEC_POSTFLAGS})
            set_tests_properties(${TARGET_NAME}_np${np} PROPERTIES PROCESSORS ${np})
        endforeach()
    else()
        gtest_discover_tests(${TARGET_NAME} DISCOVERY_TIMEOUT 60)
    endif()
endfunction()


file(GLOB _test_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_dir IN LISTS _test_dirs)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/CMakeLists.txt)
        add_subdirectory(${_dir})
    endif()
endforeach()
