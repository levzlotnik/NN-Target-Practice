//
// Created by LevZ on 6/14/2020.
//

#include <ostream>
#include "../BLAS/BLAS.h"

float mul2(float& x){return x*2;}
float add(float& x, float& y) {return x+y;};

int main(){
    float arr[] = {1., 2., 3.};
    Vector v1(3, arr, false, true);
    auto v2 = 1 / v1;
    cout << "v2 - v1 = " << v2-v1 << endl;
    cout << "Vector::arange(0, 5) = " << Vector::arange(0, 5) << endl;
    cout << "Vector::linspace(0, 5, 10) = " << Vector::linspace(0, 5, 10) << endl;
    cout << "Vector::concat({v1, v2}) = " << Vector::concat({v1, v2}) << endl;
    cout << "sin(Vector::arange(-3, 3) * PI) = " << sin(Vector::arange(-3, 3) * PI) << endl;
    cout << "Vector::zeros_like(v2) = " << Vector::zeros_like(v2) << endl;
    Matrix matrix(3, 4, 3.5f);
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
    Matrix eye3 = Matrix::eye(3, true);
    cout << "eye(3) = " << eye3 << endl;
    cout << "eye(3) @ m4 = " << matmul(eye3, m4) << endl;
    Matrix s_eye3 = eye3.to_sparse();
    cout << "s_eye(3) = " << eye3 << endl;
    cout << "s_eye(3) @ m4 = " << matmul(s_eye3, m4) << endl;
    Matrix sparse_diag = Matrix::diag(Vector::arange(0, 6), true);
    cout << "sparse_diag = " << sparse_diag << endl;
    m4 = m4.reshape(6, 2);
    cout << "m4 = " << m4 << endl;
    cout << "sparse_diag @ m4 = " << matmul(sparse_diag, m4) << endl;
    cout << "m4.T @ sparse_diag*.3f = " << matmul(m4.T(), sparse_diag*=.3f) << endl;
    m4 =
    {
        {0, 1, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 2},
        {0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 3}
    };
    Vector v = {2, 3.5, 3.14, 420, 69, 57};
    cout << "m4 = " << m4 << endl;
    cout << "v = " << v << endl;
    cout << "m4 @ v = " << matmul(m4.to_sparse(), v) << endl;
    cout << "m4 @ sparse_diag = " << matmul(m4, sparse_diag) << endl;
    return 0;
}