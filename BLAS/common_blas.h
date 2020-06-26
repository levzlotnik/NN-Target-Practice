//
// Created by LevZ on 6/14/2020.
//

#ifndef TARGETPRACTICE_COMMON_BLAS_H
#define TARGETPRACTICE_COMMON_BLAS_H
#include <string>
#include <functional>
#include <cmath>

#define MACRO_BASIC_ARITHMETIC_OPERATORS(macro) \
    macro(+) \
    macro(-) \
    macro(*) \
    macro(/)

#define MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(macro) \
    macro(+=) \
    macro(-=) \
    macro(*=) \
    macro(/=)

static int normalize_index(int i, int n){
    if (i < -n || i >= n)
        throw std::out_of_range("index should be between " + std::to_string(-n) +
            " and " + std::to_string(n-1));
    return (i + n) % n;
}

static inline float relu(float x) { return x > 0 ? x : 0; }

#define MACRO_MATH_FUNCTIONS(macro) \
    macro(sqrt) \
    macro(floor) \
    macro(ceil) \
    macro(round) \
    macro(log) \
    macro(log2) \
    macro(log10) \
    macro(log1p) \
    macro(exp) \
    macro(exp2) \
    macro(abs) \
    macro(sin) \
    macro(cos) \
    macro(tan) \
    macro(asin) \
    macro(acos) \
    macro(atan) \
    macro(sinh) \
    macro(cosh) \
    macro(tanh) \
    macro(relu)


using UnaryOperation = std::function<float(float&)>;
using BinaryOperation = std::function<float(float&, float&)>;

#endif //TARGETPRACTICE_COMMON_BLAS_H
