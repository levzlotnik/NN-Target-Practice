set(SOURCE_FILES_PROJECT_GLOBALS
        ../common.h
        ../common.cpp ../common_math.h)

add_executable(test_blas_basic test_blas_basic.cpp)
target_link_libraries(test_blas_basic blas)

add_executable(test_blas_matmul test_blas_matmul.cpp)
target_link_libraries(test_blas_matmul blas)

add_executable(test_autograd test_autograd.cpp)
target_link_libraries(test_autograd autograd)