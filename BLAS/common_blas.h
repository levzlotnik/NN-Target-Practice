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

static long normalize_index(long i, long n, bool inclusive= false){
    if (inclusive && i == n)
        return n;
    if (i < -n || i >= (n + inclusive))
        throw std::out_of_range("index should be between " + std::to_string(-n) +
            " and " + std::to_string(n-1));
    return (i + n) % n;
}


using shape_t = std::vector<size_t>;
using index_t = std::vector<long>;

static index_t normalize_index(const index_t& index, const shape_t& shape, bool inclusive= false) {
    index_t idx{index};
    int i = 0;
    for (auto& x : idx)
        x = normalize_index(x, shape[i++], inclusive);
    return idx;
}


#endif //TARGETPRACTICE_COMMON_BLAS_H
