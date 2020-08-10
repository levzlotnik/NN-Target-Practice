//
// Created by LevZ on 6/14/2020.
//

#ifndef TARGETPRACTICE_COMMON_BLAS_H
#define TARGETPRACTICE_COMMON_BLAS_H
#include <string>
#include <functional>
#include <cmath>
#include "../common.h"
#include "../common_math.h"

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

static int normalize_index(long i, int n, bool inclusive=false){
    if (inclusive && i == n)
        return n;
    if (i < -n || i >= (n + inclusive))
        throw std::out_of_range("index should be between " + std::to_string(-n) +
            " and " + std::to_string(n-1));
    return (i + n) % n;
}


#endif //TARGETPRACTICE_COMMON_BLAS_H
