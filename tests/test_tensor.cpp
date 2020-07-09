//
// Created by LevZ on 7/8/2020.
//

#include "../BLAS/BLAS.h"
#include "iostream"
using namespace std;

int main(){
    Tensor<double> t (
            {1, 2, 3, 4, 5, 6},
            {2, 1, 3}
    );
    auto t1 = t({1, 2});
    t1 = 900;
    cout << t << endl;
    return 0;
}