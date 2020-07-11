//
// Created by LevZ on 7/8/2020.
//

#include "../BLAS/BLAS.h"
#include <iostream>
using namespace std;
using namespace blas;

int main(){
    Tensor<double> t (
            {1, 2, 3,
                  4, 5, 6},
            {2, 1, 3}
    );
    cout << t << endl;
    auto t1 = t({1, 2});
    t1 = 900.0;
    auto t2 = t.apply_tensors(t1, [](double x, double y){ return x*y; });
    cout << "t = " << t << endl;
    cout << "t1 = " << t1 << endl;
    cout << "t2 = " << t2 << endl;
    auto t2_ = t2.apply(100, [](double x, double y){return x/y;});
    cout << "t2 / 100 = " << t2_ << endl;
    Tensor<double> t3 (
            {3.14, 42,
                  69, 420.0,
                  57, 45},
            {1, 2, 3}
    );
    cout << "t3 = " << t3 << endl;
    cout << "t2 / 100 + t3 = " << t2_.apply_tensors(t3, [](double x, double y){return x+y;});
    return 0;
}