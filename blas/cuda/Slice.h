#ifndef BLAS_CUDA_CUSLICE_H_
#define BLAS_CUDA_CUSLICE_H_

#include "../Slice.h"
#include "cu_common.h"

namespace blas {
namespace cuda {

class Slice {
   public:
    I_H_D Slice() : Slice(0, 1, 1) {}

    I_H_D Slice(long b, long e, long stride = 1) : b(b), e(e), stride(stride) {
        _size = calc_size();
    }

    I_H_D size_t size() const { return _size; }

    I_H_D blasStatus check_idx(size_t idx) const {
        return idx > size() ? BLAS_CUDA_OOR : BLAS_CUDA_OK;
    }
    I_H_D long get(size_t idx) const { return b + idx * stride; }

    I_H_D Slice subslice(Slice relative_slice);

    long b, e, stride;

   private:
    size_t _size;

    I_H_D size_t calc_size() const {
        long d_ = e - b;
        if (d_ == 0) return 0;
        size_t d = d_ > 0 ? d_ : -d_;
        size_t s = stride > 0 ? stride : -stride;
        return (d - 1) / s + 1;
    }
};

class SliceGroup {
   public:
    SliceGroup(const blas::SliceGroup& sg);
    ~SliceGroup();

   private:
    Slice* slices;
    size_t nslices;
};
}  // namespace cuda
}  // namespace blas
#endif  // BLAS_CUDA_CUSLICE_H_