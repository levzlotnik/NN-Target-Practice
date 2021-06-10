#include "cu_common.h"

namespace blas { namespace cuda {

cuShape_t::cuShape_t(const shape_t& shape) : ndims(shape.size()) {
    CU_CHECK_ERR(cudaMalloc(&sizes, ndims));
    CU_CHECK_ERR(cudaMemcpy(sizes, shape.data(), ndims * sizeof(size_t),
                            cudaMemcpyKind::cudaMemcpyHostToDevice));
}

cuShape_t::~cuShape_t() { cudaFree(sizes); }

cuIndex_t::cuIndex_t(const index_t& shape) : ndims(shape.size()) {
    CU_CHECK_ERR(cudaMalloc(&items, ndims));
    CU_CHECK_ERR(cudaMemcpy(items, shape.data(), ndims * sizeof(size_t),
                            cudaMemcpyKind::cudaMemcpyHostToDevice));
}

cuIndex_t::~cuIndex_t() { cudaFree(items); }
}  // namespace cuda
}  // namespace blas