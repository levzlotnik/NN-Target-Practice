//
// Created by LevZ on 6/19/2020.
//

#include "SparseMatrix.h"

float SparseMatrix::at(int i, int j) const {
    i = normalize_index(i, n);
    j = normalize_index(j, m);
    int idx = i*m + j;
    auto it = dok.find(idx);
    return (it != dok.end() ? it->second : 0.0f);
}

float &SparseMatrix::at(int i, int j) {
    i = normalize_index(i, n);
    j = normalize_index(j, m);
    int idx = i*m + j;
    return dok[idx];
}

SparseMatrix &SparseMatrix::apply_(UnaryOperation op) {
    for (auto& i: dok)
        i.second = op(i.second);
    return (*this);
}

SparseMatrix &SparseMatrix::apply_(const Matrix &other, BinaryOperation op) {
    throw runtime_error("Inplace application with another matrix as an operator is unsupported for "
                        "SparseMatrix. Convert to "
                        "non-sparse representation with "
                        "'.to_dense()' to allow inplace application with another matrix.");
}

SparseMatrix &SparseMatrix::apply_(float scalar, BinaryOperation op) {
    for (auto& i: dok)
        i.second = op(i.second, scalar);
    return (*this);
}

pair<int, int> denorm_index(int i, int n, int m){
    return make_pair(i / m, i % m);
}

Matrix SparseMatrix::to_dense() const {
    Matrix res(n, m, 0);
    for (auto i: dok){
        auto idx = denorm_index(i.first, n, m);
        res(idx.first, idx.second) = i.second;
    }
    return res;
}

Matrix SparseMatrix::apply(UnaryOperation op) const {
    return SparseMatrix(*this).apply_(op).to_dense();
}

Matrix SparseMatrix::apply(const Matrix &other, BinaryOperation op) const {
    return this->to_dense().apply_(other, op);
}

Matrix SparseMatrix::apply(float scalar, BinaryOperation op) const {
    return SparseMatrix(*this).apply_(scalar, op).to_dense();
}

Matrix SparseMatrix::operator-() const {
    return apply([](float& x){return -x;});
}

Matrix SparseMatrix::operator+(const Matrix& other) const {
    check_shapes(other);
    Matrix res(other);
    for (auto i: dok){
        auto idx = denorm_index(i.first, n, m);
        res(idx.first, idx.second) += i.second;
    }
    return res;
}

Matrix SparseMatrix::operator-(const Matrix &other) const {
    check_shapes(other);
    Matrix res = -other;
    for (auto i: dok){
        auto idx = denorm_index(i.first, n, m);
        res(idx.first, idx.second) += i.second;
    }
    return res;
}

Matrix SparseMatrix::operator*(const Matrix &other) const {
    check_shapes(other);
    Matrix res(n, m, 0);
    for (auto i: dok){
        auto idx = denorm_index(i.first, n, m);
        res(idx.first, idx.second) = i.second * other(idx.first, idx.second);
    }
    return res;
}

Matrix SparseMatrix::operator/(const Matrix &other) const {
    check_shapes(other);
    Matrix res(n, m, 0);
    for (auto i: dok){
        auto idx = denorm_index(i.first, n, m);
        res(idx.first, idx.second) = i.second / other(idx.first, idx.second);
    }
    return res;
}

SparseMatrix::SparseMatrix(int n, int m) : Matrix() {
    sparse = true;
    this->n = n;
    this->m = m;
}

Matrix *SparseMatrix::clone() const {
    return new SparseMatrix(*this);
}

Matrix matmul(const Matrix &mat1, const Matrix &mat2) {
    if (mat1.m != mat2.n)
        throw runtime_error("Shape mismatch for matrix "
                            "multiplication: " + mat1.str_shape() + ", " + mat2.str_shape());

    if (mat1.sparse && mat2.sparse) {
        auto& smat1 = (SparseMatrix &) mat1;
        auto& smat2 = (SparseMatrix &) mat2;
        return SparseMatrix::sparse_matmul(smat1, smat2).to_dense();
    }
    Matrix res(mat1.n, mat2.m, 0);
    // TODO - handle sparse@dense (and similarily dense@sparse), and lastly dense@dense.
    return res;
}

SparseMatrix SparseMatrix::transpose_sparse() {
    SparseMatrix res(m, n);
    for (auto[idx, v] : dok) {
        auto[i, j] = denorm_index(idx, n, m);
        res(j, i) = v;
    }
    return res;
}


SparseMatrix SparseMatrix::sparse_matmul(SparseMatrix &sm1, SparseMatrix &sm2) {
    // TODO - convert sm1->CSR, sm2->CSC like defined in
    //  https://en.wikipedia.org/wiki/Sparse_matrix#Storing_a_sparse_matrix
    //  and use these formats to make the efficient matmul.
    return SparseMatrix(0, 0);
}



