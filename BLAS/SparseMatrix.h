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
    SparseMatrix(int n, int m);

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

    Matrix* clone() const override;

    Matrix to_dense() const;

    Matrix operator-() const override;

    friend Matrix matmul(const Matrix& mat1, const Matrix& mat2);

private:
    SparseMatrix transpose_sparse();

    static SparseMatrix sparse_matmul(SparseMatrix &sm1, SparseMatrix &sm2);
};


#endif //TARGETPRACTICE_SPARSEMATRIX_H
