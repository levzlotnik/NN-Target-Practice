
set(SOURCE_FILES_PROJECT_GLOBALS
    ../common.h
    ../common.cpp ../common_math.h)

add_library(BLAS SHARED
            ${SOURCE_FILES_PROJECT_GLOBALS}
            BLAS.h
            common_blas.h
            Tensor.h Tensor.cpp TensorMath.h TensorMath.cpp)
