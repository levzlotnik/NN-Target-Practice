
set(SOURCE_FILES_PROJECT_GLOBALS
        ../common.h
        ../common.cpp ../common_math.h)

add_library(blas SHARED
            ${SOURCE_FILES_PROJECT_GLOBALS}
            blas.h
            common_blas.h
            Tensor.h Tensor.cpp TensorMath.h TensorMath.cpp TensorCreation.h TensorCreation.cpp)
