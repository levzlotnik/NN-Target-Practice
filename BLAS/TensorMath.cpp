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
        out_shape.resize(std::max(s1.size(), s2.size()));
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
    inline TensorView<T> promote(const Tensor1<T>& t, int pos, bool to_batched=false) {
        switch (t.dim()) {
            case 0:
                throw broadcast_failure("Cannot accept scalars for matmul of any kind.");
            case 1: {
                auto t_ = t.const_unsqueeze(pos);
                return to_batched ? t_.unsqueeze(0) : t_;
            }
            case 2: {
                auto t_ = t.const_view(t.shape);
                return to_batched ? t_.unsqueeze(0) : t_;
            }
            default:
                return t.const_view(t.shape);
        }
    }

    template<typename T>
    inline TensorSliced<T> promote(const TensorSliced<T>& t, int pos, bool to_batched=false){
        switch (t.dim()) {
            case 0:
                throw broadcast_failure("Cannot accept scalars for matmul of any kind.");
            case 1: {
                auto t_ = t.const_slice_unsqueeze(pos);
                return to_batched ? t_.slice_unsqueeze(0) : t_;
            }
            case 2: {
                return to_batched ? t.const_slice_unsqueeze(0) : t;
            }
            default:
                return t;
        }
    }

    template<template<typename> class Tensor1,
             template<typename> class Tensor2,
             typename T>
    inline void _unchecked_matmul(const Tensor1<T>& in1, const Tensor2<T>& in2, Tensor<T>& out) {
        size_t n = in1.shape[0], m = in2.shape[1], k = in1.shape[1];
        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j < m; ++j) {
                T accum = 0;
                for (size_t l=0; l < k; ++l) {
                    T a_il = Tensor1<T>::get(in1, i * k + l);
                    T b_lj = Tensor2<T>::get(in2, l * m + j);
                    accum += a_il * b_lj;
                }
                Tensor<T>::get(out, i * m + j) = accum;
            }
        }
    }

    template<template<typename> class Tensor1,
             template<typename> class Tensor2,
             typename T>
    inline void _unchecked_bmm(const Tensor1<T>& in1, const Tensor2<T>& in2, Tensor<T>& out) {
        shape_t in1_last_dims{in1.shape.end() - 2, in1.shape.end()};
        shape_t in2_last_dims{in2.shape.end() - 2, in2.shape.end()};
        shape_t out_last_dims{out.shape.end() - 2, out.shape.end()};

        shape_t in1_batch_dims{in1.shape.begin(), in1.shape.end()-2};
        shape_t in2_batch_dims{in2.shape.begin(), in2.shape.end()-2};
        shape_t out_batch_dims{out.shape.begin(), out.shape.end()-2};
        // We know that in1_last_dims and in2_last_dims are compatible for matmul-ing each other
        // so since this is the requirement for calling this function.
        // Hence we iterate over the inJ_batch_dims.
        Tensor<T> out_result(out_last_dims);
        // TODO - optimize this shit
        SliceGroup sg_in1 = SliceGroup::cover_shape(in1_batch_dims);
        for (const auto& in1_idx: sg_in1) {
            TensorSliced<T> in1_matrix = in1.unchecked_subscript_slice(in1_idx);
            auto [sg_in2, sg_out] = broadcast_index(in1_idx, in1_batch_dims, in2_batch_dims, out_batch_dims);
            for (const auto& in2_idx: sg_in2) {
                TensorSliced<T> in2_matrix = in2.unchecked_subscript_slice(in2_idx);
                _unchecked_matmul(in1_matrix, in2_matrix, out_result);
                // Inject into the output.
                for (const auto& out_idx : sg_out)
                    out.unchecked_subscript_slice(out_idx).copy_(out_result);
            }
        }
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
        auto in1 = promote(t1, 0, true);
        auto in2 = promote(t2, 1, true);
        shape_t out_shape_unsqueezed = check_shapes_bmm(in1.shape, in2.shape);
        shape_t out_shape(out_shape_unsqueezed);
        if (should_squeeze_result) {
            int squeeze_at = t1.dim() == 1 ? -2 : -1;
            out_shape.erase(out_shape.end() + squeeze_at);
        }
        Tensor<T> out(out_shape);
        TensorView<T> out_view = out.view(out_shape_unsqueezed);
        _unchecked_bmm(in1, in2, out_view);
        return out;
    }

    template<template<typename> class Tensor1,
            template<typename> class Tensor2, typename T>
    Tensor<T>& bmm(const Tensor1<T> &t1, const Tensor2<T> &t2, Tensor<T> &out) {
        static_assert(std::is_base_of_v<Tensor<T>, Tensor1<T>> &&
                      std::is_base_of_v<Tensor<T>, Tensor2<T>>,
                      "No.");
        if (t1.shape.size() < 3 && t2.shape.size() < 3)
            throw std::runtime_error("'blas::bmm' requires at least one tensor of >2 dimensions.\n\t"
                                     "For a non-batch version of matrix multiplication use 'blas::mm'.");
        auto in1 = promote(t1, 0, true);
        auto in2 = promote(t2, 1, true);
        shape_t out_shape_unsqueezed = check_shapes_bmm(in1.shape, in2.shape);
        shape_t out_shape_squeezed(out_shape_unsqueezed);
        bool should_squeeze_result = std::min(t1.dim(), t2.dim()) == 1;
        if (should_squeeze_result) {
            int squeeze_at = t1.dim() == 1 ? -2 : -1;
            out_shape_squeezed.erase(out_shape_squeezed.end() + squeeze_at);
        }
        if (out.shape != out_shape_squeezed)
            throw shape_mismatch(out.shape, out_shape_squeezed);
        TensorView<T> out_view = out.view(out_shape_unsqueezed);
        _unchecked_bmm(in1, in2, out_view);
        return out;
    }

    template<template<typename> class Tensor1,
             template<typename> class Tensor2,
             typename T>
    Tensor<T> matmul(const Tensor1<T> &t1, const Tensor2<T> &t2) {
        if (t1.shape.size() > 2 || t2.shape.size() > 2)
            throw std::runtime_error("'blas::matmul' requires at least the tensors to be of <=2 dimensions.\n\t"
                                     "For a batch version of matrix multiplication use 'blas::bmm'.");
        auto in1 = promote(t1, 0);
        auto in2 = promote(t2, 1);
        shape_t out_shape = check_matrix_matrix_mm(in1.shape, in2.shape);
        Tensor<T> out(out_shape);
        bool should_squeeze_result = std::min(t1.dim(), t2.dim()) == 1;
        _unchecked_matmul(in1, in2, out);
        if (should_squeeze_result) {
            int squeeze_at = t1.dim() == 1 ? -2 : -1;
            out.shape.erase(out.shape.end() + squeeze_at);
        }
        return out;
    }

    template<template<typename> class Tensor1,
            template<typename> class Tensor2,
            typename T>
    Tensor<T> &matmul(const Tensor1<T> &t1, const Tensor2<T> &t2, Tensor<T> &out) {
        if (t1.shape.size() > 2 || t2.shape.size() > 2)
            throw std::runtime_error("'blas::matmul' requires at least the tensors to be of <=2 dimensions.\n\t"
                                     "For a batch version of matrix multiplication use 'blas::bmm'.");
        auto in1 = promote(t1, 0);
        auto in2 = promote(t2, 1);
        shape_t out_shape_unsqueezed = check_matrix_matrix_mm(in1.shape, in2.shape);
        shape_t out_shape_squeezed(out_shape_unsqueezed);
        bool should_squeeze_result = std::min(t1.dim(), t2.dim()) == 1;
        if (should_squeeze_result) {
            int squeeze_at = t1.dim() == 1 ? -2 : -1;
            out_shape_squeezed.erase(out_shape_squeezed.end() + squeeze_at);
        }
        if (out.shape != out_shape_squeezed)
            throw shape_mismatch(out.shape, out_shape_squeezed);
        TensorView<T> out_view = out.view(out_shape_unsqueezed);
        _unchecked_matmul(in1, in2, out_view);
        return out;
    }

    template<template<typename> class Tensor1,
            template<typename> class Tensor2,
            typename T>
    Tensor<T> conv1d(const Tensor1<T> &input, const Tensor2<T> &kernels, ConvMode mode) NOT_IMPLEMENTED

    template<template<typename> class Tensor1,
            template<typename> class Tensor2,
            typename T>
    Tensor<T> conv2d(const Tensor1<T> &input, const Tensor2<T> &kernels, ConvMode mode) NOT_IMPLEMENTED


    template<template<typename> class Tensor1,
            template<typename> class Tensor2,
            typename T>
    Tensor<T>& conv1d(const Tensor1<T> &input, const Tensor2<T> &kernels, Tensor<T>& out, ConvMode mode) NOT_IMPLEMENTED

    template<template<typename> class Tensor1,
            template<typename> class Tensor2,
            typename T>
    Tensor<T>& conv2d(const Tensor1<T> &input, const Tensor2<T> &kernels, Tensor<T>& out, ConvMode mode) NOT_IMPLEMENTED


#define INSTANTIATE_MATRIX_OPS(dtype) \
    INSTANTIATE_MATMUL(dtype)         \
    INSTANTIATE_BMM(dtype)            \
    INSTANTIATE_CONV1D(dtype)         \
    INSTANTIATE_CONV2D(dtype)

#define TWO_ARG_FUNCTION(Tnsr1, Tnsr2, T, func) \
    template Tensor<T> func<Tnsr1, Tnsr2, T>(const Tnsr1<T>& t1, const Tnsr2<T>& t2); \
    template Tensor<T>& func<Tnsr1, Tnsr2, T>(const Tnsr1<T>& t1, const Tnsr2<T>& t2, Tensor<T>& out);

#define CONV_FUNCTION(Tnsr1, Tnsr2, T, func) \
    template Tensor<T> func<Tnsr1, Tnsr2, T>(const Tnsr1<T>& t1, const Tnsr2<T>& t2, ConvMode mode); \
    template Tensor<T>& func<Tnsr1, Tnsr2, T>(const Tnsr1<T>& t1, const Tnsr2<T>& t2, Tensor<T>& out, ConvMode mode);

#define APPLY_FUNCTION(T, func, macro) \
    macro(Tensor, Tensor, T, func) \
    macro(Tensor, TensorView, T, func) \
    macro(Tensor, TensorSliced, T, func) \
    macro(TensorView, Tensor, T, func) \
    macro(TensorView, TensorView, T, func) \
    macro(TensorView, TensorSliced, T, func) \
    macro(TensorSliced, Tensor, T, func) \
    macro(TensorSliced, TensorView, T, func) \
    macro(TensorSliced, TensorSliced, T, func)

#define INSTANTIATE_MATMUL(dtype) \
    APPLY_FUNCTION(dtype, matmul, TWO_ARG_FUNCTION)

#define INSTANTIATE_BMM(dtype) \
    APPLY_FUNCTION(dtype, bmm, TWO_ARG_FUNCTION)

#define INSTANTIATE_CONV1D(dtype) \
    APPLY_FUNCTION(dtype, conv1d, CONV_FUNCTION)

#define INSTANTIATE_CONV2D(dtype) \
    APPLY_FUNCTION(dtype, conv2d, CONV_FUNCTION)

    INSTANTIATE_MATRIX_OPS(double)
}

