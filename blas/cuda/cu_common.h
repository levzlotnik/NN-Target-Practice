#ifndef BLAS_CUDA_CU_COMMON_H_
#define BLAS_CUDA_CU_COMMON_H_

#include <cuda_runtime.h>

#include <string>

#include "../common_blas.h"

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define LINE_STRING STRINGIZE(__LINE__)

#define CU_CHECK_ERR(expr)                                                  \
    do {                                                                    \
        cudaError_t status = expr;                                          \
        if (status != cudaSuccess) {                                        \
            std::string message =                                           \
                __FILE__ ":" LINE_STRING ": Error running " #expr;          \
            throw std::runtime_error(message + cudaGetErrorString(status)); \
        }                                                                   \
    } while (0);

#define H_D __host__ __device__
#define I_H_D __inline__ H_D

namespace blas {
namespace cuda {

enum blasStatus {
    BLAS_CUDA_OK,   // no errors
    BLAS_CUDA_OOM,  // out of memory
    BLAS_CUDA_OOR   // out of range
};

class cuShape_t {
   public:
    cuShape_t(const shape_t& shape);
    ~cuShape_t();

    I_H_D size_t size() const { return ndims; }
    __device__ __inline__ size_t operator[](size_t i) const { return sizes[i]; }
    __device__ __inline__ size_t& operator[](size_t i) { return sizes[i]; }

   private:
    size_t* sizes;
    size_t ndims;
};

class cuIndex_t {
   public:
    cuIndex_t(const index_t& shape);
    ~cuIndex_t();

    I_H_D size_t size() const { return ndims; }
    __device__ __inline__ long operator[](size_t i) const { return items[i]; }
    __device__ __inline__ long& operator[](size_t i) { return items[i]; }

   private:
    long* items;
    size_t ndims;
};
}  // namespace cuda
}  // namespace blas

#endif  // BLAS_CUDA_CU_COMMON_H_