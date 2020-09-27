//
// Created by LevZ on 7/12/2020.
//

#ifndef TARGETPRACTICE_TENSORMATH_H
#define TARGETPRACTICE_TENSORMATH_H

#include "Tensor.h"
#include "../common_math.h"

namespace blas {

#define DEF_TENSOR_TENSOR_OP(Tensor1, Tensor2, op) \
    template<typename T> \
    inline Tensor<T> operator op(const Tensor1<T>& t1, const Tensor2<T>& t2) { \
        return t1.apply_tensors(t2, [](T x, T y) -> T { return x op y; }); \
    }

#define DEF_TENSOR_TENSOR_OP_INPLACE(Tensor1, Tensor2, op) \
    template<typename T> \
    inline Tensor1<T>& operator op(Tensor1<T>& t1, const Tensor2<T>& t2) { \
        return t1.apply_tensors_(t2, [](T x, T y) -> T { return x op y; }); \
    }

#define DEF_TENSOR_TENSOR_INTERACTIVE_OPS(op) \
    DEF_TENSOR_TENSOR_OP(Tensor, Tensor, op) \
    DEF_TENSOR_TENSOR_OP(Tensor, TensorView, op) \
    DEF_TENSOR_TENSOR_OP(Tensor, TensorSliced, op) \
    DEF_TENSOR_TENSOR_OP(TensorView, Tensor, op) \
    DEF_TENSOR_TENSOR_OP(TensorView, TensorView, op) \
    DEF_TENSOR_TENSOR_OP(TensorView, TensorSliced, op) \
    DEF_TENSOR_TENSOR_OP(TensorSliced, Tensor, op) \
    DEF_TENSOR_TENSOR_OP(TensorSliced, TensorView, op) \
    DEF_TENSOR_TENSOR_OP(TensorSliced, TensorSliced, op) \

#define DEF_TENSOR_TENSOR_INTERACTIVE_OPS_INPLACE(op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(Tensor, Tensor, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(Tensor, TensorView, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(Tensor, TensorSliced, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(TensorView, Tensor, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(TensorView, TensorView, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(TensorView, TensorSliced, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(TensorSliced, Tensor, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(TensorSliced, TensorView, op) \
    DEF_TENSOR_TENSOR_OP_INPLACE(TensorSliced, TensorSliced, op) \

    MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_TENSOR_TENSOR_INTERACTIVE_OPS)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_TENSOR_TENSOR_INTERACTIVE_OPS_INPLACE)

#define DEF_TENSOR_SCALAR_OP(Tnsr, op) \
    template<typename T> \
    inline Tensor<T> operator op(const Tnsr<T>& t1, T x) { \
        return t1.apply(x, [](T x, T y) -> T { return x op y; }); \
    } \
    template<typename T> \
    inline Tensor<T> operator op(T x, const Tnsr<T>& t1) { \
        return t1.apply(x, [](T x, T y) -> T { return y op x; }); \
    }

#define DEF_TENSOR_SCALAR_OP_INPLACE(Tnsr, op) \
    template<typename T> \
    inline Tnsr<T>& operator op(Tnsr<T>& t1, T x) { \
        return t1.apply_(x, [](T x, T y) -> T { return x op y; }); \
    }

#define DEF_TENSOR_SCALAR_INTERACTIVE_OPS(op) \
    DEF_TENSOR_SCALAR_OP(Tensor, op) \
    DEF_TENSOR_SCALAR_OP(TensorView, op) \
    DEF_TENSOR_SCALAR_OP(TensorSliced, op)

#define DEF_TENSOR_SCALAR_INTERACTIVE_OPS_INPLACE(op) \
    DEF_TENSOR_SCALAR_OP_INPLACE(Tensor, op) \
    DEF_TENSOR_SCALAR_OP_INPLACE(TensorView, op) \
    DEF_TENSOR_SCALAR_OP_INPLACE(TensorSliced, op)

    MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_TENSOR_SCALAR_INTERACTIVE_OPS)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_TENSOR_SCALAR_INTERACTIVE_OPS_INPLACE)

#define MATH_FUNC_TENSOR_INLINE_INTERACTABLE(Tensor1, func) \
    template<typename T> \
    Tensor<T> func(const Tensor1<T>& t) { return t.func(); } \
    template<typename T> \
    void func(const Tensor1<T>& t, Tensor<T>& out) { return t.func(out); } \
    template<typename T> \
    void func(const Tensor1<T>& t, TensorView<T>& out) { return t.func(out); } \
    template<typename T> \
    void func(const Tensor1<T>& t, TensorSliced<T>& out) { return t.func(out); }

#define MATH_FUNC_TENSOR_INLINE(func) \
    MATH_FUNC_TENSOR_INLINE_INTERACTABLE(Tensor, func) \
    MATH_FUNC_TENSOR_INLINE_INTERACTABLE(TensorView, func) \
    MATH_FUNC_TENSOR_INLINE_INTERACTABLE(TensorSliced, func)

    MACRO_MATH_FUNCTIONS(MATH_FUNC_TENSOR_INLINE)

    template<typename T>
    T mse(const Tensor<T>& in1, const Tensor<T>& in2) {
        using bfd = common_math::binary_func_data<T>;
        auto i1 = Tensor<T>::const_elem_begin(in1), i2 = Tensor<T>::const_elem_begin(in2);
        auto e1 = Tensor<T>::const_elem_end(in1);
        T x1, x2, x_out;
        T n = in1.size;
        for(; i1 != e1; ++i1, ++i2) {
            x1 = *i1;
            x2 = *i2;
            x_out += bfd::pow(x1 - x2, T(2.0)) / n;
        }
        return x_out;
    }

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T> matmul(const Tensor1<T>& t1, const Tensor2<T>& t2);

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T>& matmul(const Tensor1<T>& t1, const Tensor2<T>& t2, Tensor<T>& out);

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    inline Tensor<T> mm(const Tensor1<T>& t1, const Tensor2<T>& t2){
        return matmul(t1, t2);
    }

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    inline Tensor<T>& mm(const Tensor1<T>& t1, const Tensor2<T>& t2, Tensor<T>& out){
        return matmul(t1, t2, out);
    }

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T> bmm(const Tensor1<T>& t1, const Tensor2<T>& t2);

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T>& bmm(const Tensor1<T>& t1, const Tensor2<T>& t2, Tensor<T>& out);

    enum ConvMode {
        SAME,
        VALID
    };
    // TODO - implement all of these.

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T> conv1d(const Tensor1<T>& input, const Tensor2<T>& kernels, ConvMode mode = ConvMode::VALID);

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T>& conv1d(const Tensor1<T>& input, const Tensor2<T>& kernels, Tensor<T>& out, ConvMode mode = ConvMode::VALID);

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T> conv2d(const Tensor1<T>& input, const Tensor2<T>& kernels, ConvMode mode = ConvMode::VALID);

    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
    extern Tensor<T>& conv2d(const Tensor1<T>& input, const Tensor2<T>& kernels, Tensor<T>& out, ConvMode mode = ConvMode::VALID);

//    template<template<typename> class  Tensor1, template<typename> class Tensor2, typename T>
//    Tensor<T> conv3d(const Tensor1<T>& input, const Tensor2<T>& kernels, ConvMode mode = ConvMode::VALID);

}

#endif //TARGETPRACTICE_TENSORMATH_H
