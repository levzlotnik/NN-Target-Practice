//
// Created by LevZ on 7/8/2020.
//

#include "../BLAS/BLAS.h"
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

    // Stress testing for profiling:
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

    t = Tensor<double> {
            {1, 2, 3, 4,
                  5, 6, 7, 8},
            {2, 4}
    };
    auto t10 = t({{}, {2, 4}});
    PRINT_EXPR(matmul(t10, t));

    auto t11 = arange<double>(0, 1 * 2 * 1* 2).reshape({1, 2, 1, 2});
    PRINT_EXPR(bmm(t11, t));
    auto t12 = arange<double>(0, 2*1*2*1).reshape({2,1,2,1});
    PRINT_EXPR(bmm(t11, t12)); // TODO - this doesn't comply with numpy for some reason. debug it.
    return 0;
}