//
// Created by LevZ on 6/14/2020.
//

#include <ostream>
#include "../BLAS/BLAS.h"

float mul2(float& x){return x*2;}
float add(float& x, float& y) {return x+y;};

int main(){
    Matrix matrix(3, 4, 3.5);
    matrix.at(2,-1) /= 45.678;
    matrix.apply_(mul2);
    cout << matrix << endl;
    Matrix m2 = matrix.apply(mul2);
    m2.at(0, 0) *= 19;
    cout << "m2 = " << m2 << endl;
    Matrix m3 = m2.apply(matrix, add);
    cout << "m3 = " << m3 << endl;
    Matrix m4 = 4.0 / m3;
    cout << "m4 = " << m4 << endl;
    m4 *= 10;
    cout << "m4 = " << m4 << endl;
    Matrix eye3 = Matrix::eye(3);
    cout << "eye(3) = " << eye3 << endl;
    cout << "eye(3) @ m4 = " << eye3.matmul(m4) << endl;
    return 0;
}