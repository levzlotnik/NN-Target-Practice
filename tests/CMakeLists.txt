set(SOURCE_FILES_PROJECT_GLOBALS
        ../common.h
        ../common.cpp ../common_math.h)

add_executable(test_BLAS_basic test_BLAS_basic.cpp)
target_link_libraries(test_BLAS_basic BLAS)

add_executable(test_BLAS_matmul test_BLAS_matmul.cpp)
target_link_libraries(test_BLAS_matmul BLAS)