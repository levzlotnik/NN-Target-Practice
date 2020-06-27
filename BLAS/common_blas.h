//
// Created by LevZ on 6/14/2020.
//

#ifndef TARGETPRACTICE_COMMON_BLAS_H
#define TARGETPRACTICE_COMMON_BLAS_H
#include <string>
#include <functional>
#include <cmath>
#include "../common.h"

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


using UnaryOperation = std::function<float(float&)>;
using BinaryOperation = std::function<float(float&, float&)>;


#endif //TARGETPRACTICE_COMMON_BLAS_H
