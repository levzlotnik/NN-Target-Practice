set(SOURCE_FILES_PROJECT_GLOBALS
        ../common.h
        ../common.cpp ../common_math.h)

add_library(cuda_kernels STATIC
            cuda/cu_common.h
            cuda/cuda_launchers.h cuda/cuda_launchers.cpp
            cuda/cuda_kernels.h cuda/cuda_kernels.cu) 
set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES "50")


add_library(blas SHARED
            ${SOURCE_FILES_PROJECT_GLOBALS}
            blas.h
            common_blas.h
            Slice.h
            Tensor.h TensorView.h TensorSliced.h TensorTransposed.h
            all_tensors.h
            implementation.cpp
            TensorMath.h TensorMath.cpp 
            TensorCreation.h TensorCreation.cpp)
