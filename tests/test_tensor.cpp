//
// Created by LevZ on 7/8/2020.
//

#include "../BLAS/BLAS.h"
#include <iostream>
using namespace std;
using namespace blas;

int main(){
    Tensor<double>::precision = 3;
    Tensor<double> t (
            {1, 2, 3,
                  4, 5, 6},
            {2, 1, 3}
    );
    cout << t << endl;
    auto t1 = t({1, 2});
    t1 = 900.0;
    auto t2 = t * t1;
    cout << "t = " << t << endl;
    cout << "t1 = " << t1 << endl;
    cout << "t2 = " << t2 << endl;
    auto t2_ = t2 / 100.;
    cout << "t2 / 100 = " << t2_ << endl;
    Tensor<double> t3 (
            {3.14, 42,
                  69, 420.0,
                  57, 45},
            {1, 2, 3}
    );
    cout << "t3 = " << t3 << endl;
    auto t4 = t2_ + t3;
    cout << "t4 = t2 / 100 + t3 = " << t4 << endl;
    cout << "log(t4) = " << log(t4) << endl;
    cout << "t4.log1p_() = " << t4.log1p_() << endl;
    cout << "t1.log10_() = " << t1.log10_() << endl;
    cout << "t = " << t << endl;
    cout << "t[0] = " << t[0] << endl;
    cout << "t[0,0,2] = " << t[{0,0,2}] << endl;

    return 0;
}