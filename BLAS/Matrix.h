//
// Created by LevZ on 6/14/2020.
//

#ifndef TARGETPRACTICE_MATRIX_H
#define TARGETPRACTICE_MATRIX_H

#include <iostream>
#include <string>
#include <functional>
#include "common_blas.h"
#include "Vector.h"
using namespace std;

class Matrix {
private:
    float* data;
public:
    int n, m;
    Matrix();

    Matrix(int n, int m);
    Matrix(int n, int m, float init);

    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;

    friend void swap(Matrix& m1, Matrix& m2) noexcept;
    Matrix& operator=(Matrix other);

    ~Matrix();

    inline float& at(int i, int j){
        i = normalize_index(i, n);
        j = normalize_index(j, m);
        return data[i*m + j];
    }

    inline float at(int i, int j) const {
        i = normalize_index(i, n);
        j = normalize_index(j, m);
        return data[i*m + j];
    }

    inline float& operator () (int i, int j){
        return at(i, j);
    }

    inline float operator () (int i, int j) const{
        return at(i, j);
    }


    friend ostream& operator<<(ostream& os, const Matrix& matrix);

    /* Inplace operations */
    Matrix& apply_(UnaryOperation op);
    Matrix& apply_(const Matrix& other, BinaryOperation op);
    Matrix& apply_(float scalar, BinaryOperation op);
    /* Out of place operations */
    Matrix apply(UnaryOperation op);
    Matrix apply(const Matrix& other, BinaryOperation op);
    Matrix apply(float scalar, BinaryOperation op);

#define DECL_MATRIX_OPERATOR(op) \
    Matrix operator op(const Matrix& other); \
    Matrix operator op(float scalar); \
    friend Matrix operator op(float scalar, const Matrix& matrix);

#define DECL_MATRIX_OPERATOR_INPLACE(op) \
    Matrix& operator op(const Matrix& other); \
    Matrix& operator op(float scalar);

    // Declare all basic element wise operations!
    MACRO_BASIC_ARITHMETIC_OPERATORS(DECL_MATRIX_OPERATOR)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DECL_MATRIX_OPERATOR_INPLACE)

    float reduce(float init_value, BinaryOperation op);
    float sum();
    float mean();
    float var();
    float std();
    float prod();
    float trace();

    Vector reduce_axis(float init_value, int axis, BinaryOperation op);
    Vector sum(int axis);
    Vector mean(int axis);
    Vector var(int axis);
    Vector std(int axis);
    Vector get_diag() const;
    Matrix& set_diag(Vector diag);

    Vector get_row(int i);
    void set_row(int i, const Vector& row);

    Vector flatten(bool copy=true);
    Matrix reshape(int new_n, int new_m);

    Matrix matmul(const Matrix& other);

    Matrix transpose();

    static Matrix zeros(int n, int m);
    static Matrix ones(int n, int m);
    static Matrix zeros_like(const Matrix& matrix);
    static Matrix ones_like(const Matrix& matrix);
    static Matrix eye(int n);

private:
    void check_shapes(const Matrix& other);
    string str_shape() const;

    Vector get_col(int i);
};


#endif //TARGETPRACTICE_MATRIX_H
