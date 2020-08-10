//
// Created by LevZ on 8/7/2020.
//

#include "TensorMath.h"

namespace blas {

    class matmul_shape_mismatch : public std::runtime_error {
    public:
        explicit matmul_shape_mismatch(const std::string &what) :
                std::runtime_error("matmul_shape_mismatch: " + what) {}

        matmul_shape_mismatch(const shape_t &s1, const shape_t &s2) :
                matmul_shape_mismatch(shape2str(s1) + ", " + shape2str(s2)) {}
    };

    shape_t check_matrix_matrix_mm(const shape_t &s1, const shape_t &s2) {
        if (s1.size() != 2 || s2.size() != 2)
            throw matmul_shape_mismatch("Matrix dims must be 2.");
        if (s1[1] != s2[0])
            throw matmul_shape_mismatch(s1, s2);
        return shape_t{s1[0], s2[1]};
    }

    shape_t check_shapes_bmm(const shape_t &s1, const shape_t &s2) {
        shape_t out_shape;
        shape_t s1_b, s2_b;
        out_shape.reserve(std::max(s1.size(), s2.size()));
        shape_t s1_last_dims(s1.end() - 2, s1.end());
        shape_t s2_last_dims(s2.end() - 2, s2.end());
        shape_t out_last_dims = check_matrix_matrix_mm(s1_last_dims, s2_last_dims);
        shape_t s1_first_dims(s1.begin(), s1.end() - 2);
        shape_t s2_first_dims(s2.begin(), s2.end() - 2);
        shape_t out_first_dims = broadcast_shapes(s1_first_dims, s2_first_dims);
        std::copy(out_first_dims.begin(), out_first_dims.end(), out_shape.begin());
        std::copy(out_last_dims.begin(), out_last_dims.end(), out_shape.end() - 2);
        return out_shape;
    }

    template<template<typename> class Tensor1, typename T>
    inline TensorView<T> promote_to_matrix(Tensor1<T>& t, int pos) {
        return t.dim() == 1 ? t.unsqueeze(pos) : t.view(t.shape);
    }

    template<typename T>
    inline Tensor<T> promote_to_matrix(TensorSliced<T>& t, int pos){
        if (t.dim() > 1)
            return t.contiguous();
        return pos == 0 ?
               t.reshape({1, t.shape[0]}):
               t.reshape({t.shape[0], 1});
    }

    template<template<typename> class Tensor1,
             template<typename> class Tensor2,
             typename T>
    inline void _unchecked_bmm(const Tensor1<T>& in1, const Tensor2<T>& in2, Tensor<T>& out) {
        shape_t in1_last_dims{in1.shape.end() - 2, in1.shape.end()};
        shape_t in2_last_dims{in2.shape.end() - 2, in2.shape.end()};
        shape_t in1_batch_dims(in1.shape.begin(), in1.shape.end()-2);
        shape_t in2_batch_dims(in2.shape.begin(), in2.shape.end()-2);
        // TODO - continue implementing
    }


    template<template<typename> class Tensor1,
            template<typename> class Tensor2, typename T>
    Tensor<T> bmm(const Tensor1<T> &t1, const Tensor2<T> &t2) {
        static_assert(std::is_base_of_v<Tensor<T>, Tensor1<T>> &&
                      std::is_base_of_v<Tensor<T>, Tensor2<T>>,
                      "No.");

        if (t1.shape.size() < 3 && t2.shape.size() < 3)
            throw std::runtime_error("'blas::bmm' requires at least one tensor of >2 dimensions.\n\t"
                                     "For a non-batch version of matrix multiplication use 'blas::mm'.");
        bool should_squeeze_result = std::min(t1.dim(), t2.dim()) == 1;
        auto in1 = promote_to_matrix(t1, 0);
        auto in2 = promote_to_matrix(t2, 1);
        shape_t out_shape_unsqueezed = check_shapes_bmm(in1.shape, in2.shape);
        shape_t out_shape(out_shape_unsqueezed);
        if (should_squeeze_result) {
            int squeeze_at = out_shape.size() - 1 - (t1.dim() == 1 ? 1 : 0);
            out_shape.erase(out_shape.begin() + squeeze_at);
        }
        Tensor<T> ret(out_shape);
        _unchecked_bmm(in1, in2, ret.view(out_shape_unsqueezed));
        return ret;
    }

    template<template<typename> class Tensor1,
            template<typename> class Tensor2, typename T>
    Tensor<T> bmm(const Tensor1<T> &t1, const Tensor2<T> &t2, Tensor<T> &out) {
        static_assert(std::is_base_of_v<Tensor<T>, Tensor1<T>> &&
                      std::is_base_of_v<Tensor<T>, Tensor2<T>>,
                      "No.");
        if (t1.shape.size() < 3 && t2.shape.size() < 3)
            throw std::runtime_error("'blas::bmm' requires at least one tensor of >2 dimensions.\n\t"
                                     "For a non-batch version of matrix multiplication use 'blas::mm'.");
        auto in1 = promote_to_matrix(t1, 0);
        auto in2 = promote_to_matrix(t2, 1);
        shape_t out_shape_unsqueezed = check_shapes_bmm(in1.shape, in2.shape);
        if (out.shape != out_shape_unsqueezed)
            throw shape_mismatch(out.shape, out_shape_unsqueezed);
        _unchecked_bmm(in1, in2, out.view(out_shape_unsqueezed));
        return out;
    }

}
