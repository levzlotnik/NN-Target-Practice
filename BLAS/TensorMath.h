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
    DEF_TENSOR_SCALAR_OP(TensorSliced, op) \


#define DEF_TENSOR_SCALAR_INTERACTIVE_OPS_INPLACE(op) \
    DEF_TENSOR_SCALAR_OP_INPLACE(Tensor, op) \
    DEF_TENSOR_SCALAR_OP_INPLACE(TensorView, op) \
    DEF_TENSOR_SCALAR_OP_INPLACE(TensorSliced, op) \


    MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_TENSOR_SCALAR_INTERACTIVE_OPS)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_TENSOR_SCALAR_INTERACTIVE_OPS_INPLACE)

#define MATH_FUNC_TENSOR_INLINE(func) \
    template<typename T> \
    Tensor<T> func(const Tensor<T>& t) { return t.func(); }

    MACRO_MATH_FUNCTIONS(MATH_FUNC_TENSOR_INLINE)

    template<typename T>
    Tensor<T> linspace(T s, T f, size_t num=50) {
        T step_size = (f - s) / T(num -1);
        shape_t shape = {num};
        Tensor<T> ret(shape);
        T v = s;
        for (int i=0; i < num; ++i, v += step_size)
            Tensor<T>::get(ret, i) = v;
        return ret;
    }

    template<typename T>
    Tensor<T> arange(T s, T f, T step=1.) {
        using std::to_string;
        long num = ::floor((f-s) / step);
        if (num < 1)
            throw std::runtime_error("Invalid step - can't get to " + to_string(s) + " from " + to_string(f) + ".");
        shape_t shape = {size_t(num)};
        Tensor<T> ret(shape);
        T v = s;
        for (int i=0; i < num; ++i, v += step)
            Tensor<T>::get(ret, i) = v;
        return ret;
    };

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
