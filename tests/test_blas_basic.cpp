//
// Created by LevZ on 7/8/2020.
//

#include "../blas/blas.h"
#include "common.h"
#include <iostream>
using namespace std;
using namespace blas;

int main(){
    Tensor<double> t (
            {1, 2, 3,
             4, 5, 6},
            {2, 1, 3}
    );
    PRINT_EXPR(t);
    PRINT_EXPR(t.const_permute({0, 1, 2}));
    PRINT_EXPR(t.permute({1, 2, 0}));
    auto t1 = t({1, 2});
    t1 = 900.0;
    auto t2 = t * t1;
    PRINT_EXPR(t);
    PRINT_EXPR(t1);
    PRINT_EXPR(t1 / 100.0);
    Tensor<double> t3 (
            {3.14, 42,
                  69,   420.0,
                  57,   45},
            {1, 2, 3}
    );
    PRINT_EXPR(t3);
    // Reduce:
    PRINT_EXPR(t3.sum());
    PRINT_EXPR(t3.sum(1));
    PRINT_EXPR(t3.sum({0, 2}));

    PRINT_EXPR(t2 / 100.0 + t3);
    auto t4 = t2 / 100.0 + t3;
    PRINT_EXPR(log(t4));
    PRINT_EXPR(t4.log1p_());
    PRINT_EXPR(t1.log10_());
    PRINT_EXPR(t);
    PRINT_EXPR(t[0]);
    PRINT_EXPR((t[{0, 0, 2}]));
    auto t5 = linspace(-0.5, 0.5, 100);
    PRINT_EXPR((t5({0, 0, 5})));
    PRINT_EXPR((t5({0, 0, 5})({0, 0, 5})));
    PRINT_EXPR(t5({0, 0, 25}) - t5({0, 0, 5})({0, 0, 5}));
    auto t6 = arange<double>(1, 4).reshape({3, 1});
    auto t7 = arange<double>(3, 6).reshape({1, 3});
    PRINT_EXPR(t6 + t7);

    // Stress testing elemwise ops for profiling:
    Tensor<double> big {
            {1000, 1000}
    };
    float x = 0;
    for (auto it =  Tensor<double>::elem_begin(big);
              it != Tensor<double>::elem_end(big); ++it)
        *it = x++;
    const Tensor<double> big_copy = big;
    auto big_sub = big_copy[420];
    big *= big_sub;

    return 0;
}