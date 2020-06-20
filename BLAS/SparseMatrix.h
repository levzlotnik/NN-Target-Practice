//
// Created by LevZ on 6/19/2020.
//

#ifndef TARGETPRACTICE_SPARSEMATRIX_H
#define TARGETPRACTICE_SPARSEMATRIX_H


#include "Matrix.h"

class SparseMatrix : public Matrix {
private:
    unordered_map<int, float> dok; // Dictionary of Keys represntation.
public:
    SparseMatrix() {};
    SparseMatrix(int n, int m);

    SparseMatrix(const SparseMatrix& other);
    SparseMatrix(SparseMatrix&& other) noexcept ;
    SparseMatrix(initializer_list<initializer_list<float>> list) :
        SparseMatrix(sparsify(Matrix(list))) {}

    friend void swap(SparseMatrix& sm1, SparseMatrix& sm2);
    SparseMatrix& operator=(SparseMatrix other);

    float at(int i, int j) const override;
    float& at(int i, int j) override;

    ~SparseMatrix() override = default;

    SparseMatrix& apply_(UnaryOperation op) override;
    SparseMatrix& apply_(const Matrix& other, BinaryOperation op) override;
    SparseMatrix& apply_(float scalar, BinaryOperation op) override;

    Matrix apply(UnaryOperation op) const override;
    Matrix apply(const Matrix& other, BinaryOperation op) const override;
    Matrix apply(float scalar, BinaryOperation op) const override;

    Matrix operator+(const Matrix& other) const override;
    Matrix operator-(const Matrix& other) const override;
    Matrix operator*(const Matrix& other) const override;
    Matrix operator/(const Matrix& other) const override;

#define DECL_SPARSEMATRIX_OPERATOR_SCALAR(op) \
    Matrix operator op(float scalar) const;

#define DECL_SPARSEMATRIX_OPERATOR_SCALAR_INPLACE(op) \
    SparseMatrix& operator op(float scalar);

MACRO_BASIC_ARITHMETIC_OPERATORS(DECL_SPARSEMATRIX_OPERATOR_SCALAR)
MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DECL_SPARSEMATRIX_OPERATOR_SCALAR_INPLACE)



    Matrix* clone() const override;

    Matrix to_dense() const;

    Matrix operator-() const override;

    friend Matrix matmul(const Matrix& mat1, const Matrix& mat2);
    friend Vector matmul(const Vector& v, const Matrix& m);
    friend Vector matmul(const Matrix& m, const Vector& v);

    static SparseMatrix sparsify(const Matrix& m);

    inline int nnz() const { return dok.size(); };

private:
    SparseMatrix transpose_sparse();

    static SparseMatrix sparse_matmul(SparseMatrix &sm1, SparseMatrix &sm2);
};

Matrix matmul(const Matrix& mat1, const Matrix& mat2);
Vector matmul(const Vector& v, const Matrix& m);
Vector matmul(const Matrix& m, const Vector& v);


#endif //TARGETPRACTICE_SPARSEMATRIX_H
