//
// Created by LevZ on 6/19/2020.
//

#include "SparseMatrix.h"
#include <unordered_set>

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

#define DEF_SPARSEMATRIX_SCALAR_OP(op) \
    Matrix SparseMatrix::operator op(float scalar) const { \
        return apply(scalar, [](float& x, float& y){ return x op y;}); \
    }

#define DEF_SPARSEMATRIX_SCALAR_OP_INPLACE(op) \
    SparseMatrix& SparseMatrix::operator op(float scalar) { \
        return apply_(scalar, [](float& x, float& y){ return x op y;}); \
    }

MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_SPARSEMATRIX_SCALAR_OP_INPLACE)
MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_SPARSEMATRIX_SCALAR_OP)


SparseMatrix::SparseMatrix(int n, int m) : Matrix() {
    Matrix::sparse = true;
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
    if (mat1.sparse) {
        auto& smat1 = (SparseMatrix &) mat1;
        for (auto [idx, val]: smat1.dok){
            auto [i, k] = denorm_index(idx, smat1.n, smat1.m);
            for (int j=0; j < res.m; ++j)
                res(i, j) += val * mat2(k, j);
        }
    }
    else if (mat2.sparse){
        auto& smat2 = (SparseMatrix &) mat2;
        for (auto [idx, val]: smat2.dok){
            auto [k, j] = denorm_index(idx, smat2.n, smat2.m);
            for (int i=0; i < res.n; ++i)
                res(i, j) += mat1(i, k) * val;
        }
    }
    else {
        for (int i = 0; i < res.n; ++i)
            for (int j = 0; j < res.m; ++j)
                for (int k = 0; k < mat1.m; ++k)
                    res(i, j) += mat1(i, k) * mat2(k, j);
    }
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
    SparseMatrix res(sm1.n, sm2.m);
    // Heuristic -
    // strategy = arg min { nnz(one) * respective_dim(other) }
    // For instance:
    // sm1.nnz()=3, sm2.n_cols()=5 vs sm2.nnz()=5, sm1.n_rows()=2
    // the second one wins because we only need to do 10 ops instead of 15.
    // we assume that the index preprocessing time is negligble.
    bool strategy_right = sm1.nnz()*sm2.m <= sm2.nnz()*sm1.n;
    if (strategy_right) {
        // Preprocess columns to exclude 0 columns of sm2.
        unordered_set<int> cols;
        for (auto [idx, v] : sm2.dok)
            cols.insert(denorm_index(idx, sm2.n, sm2.m).second);

        for (auto [idx, v] : sm1.dok){
            auto [i, k] = denorm_index(idx, sm1.n, sm1.m);
            for (auto j: cols)
                res(i, j) += v * sm2(k, j);
        }
    }
    else {
        // Preprocess columns to exclude 0 columns of sm2.
        unordered_set<int> rows;
        for (auto [idx, v] : sm1.dok)
            rows.insert(denorm_index(idx, sm1.n, sm1.m).first);

        for (auto [idx, v] : sm2.dok){
            auto [k, j] = denorm_index(idx, sm2.n, sm2.m);
            for (auto i: rows)
                res(i, j) += sm1(i, k) * v;
        }
    }
    return res;
}

Vector matmul(const Matrix &m, const Vector &v) {
    if (m.m != v.n)
        throw runtime_error("Shape mismatch: matrix multiplication with " +
            m.str_shape() + ", (" + to_string(v.n) + ").");
    Vector res(m.n, 0.0);
    if (m.sparse){
        auto& sm = (SparseMatrix &) m;
        for (auto[idx, val]: sm.dok){
            auto[i, j] = denorm_index(idx, sm.n, sm.m);
            res[i] += val * v[j];
        }
    }
    else {
        for (int i=0; i < res.n; ++i)
            for (int j = 0; j < m.m; ++j)
                res[i] += m(i, j) * v[j];
    }
    return res;
}

Vector matmul(const Vector &v, const Matrix &m) {
    if (m.n != v.n)
        throw runtime_error("Shape mismatch: matrix multiplication with " +
            to_string(v.n) + ", (" + m.str_shape() + ").");
    Vector res(m.m, 0.0);
    if (m.sparse){
        auto& sm = (SparseMatrix &) m;
        for (auto[idx, val]: sm.dok) {
            auto[i, j] = denorm_index(idx, sm.n, sm.m);
            res[j] += v[i] * val;
        }
    }
    else {
        for (int i=0; i < v.n; ++i)
            for (int j=0; j < m.m; ++j)
                res[j] += v[i] * m(i, j);
    }
    return res;
}

SparseMatrix SparseMatrix::sparsify(const Matrix &m) {
    if (m.sparse)
        return (SparseMatrix&) m;
    SparseMatrix res(m.n, m.m);
    for (int i=0; i < m.n; ++i)
        for (int j=0; j < m.m; ++j) {
            auto t = m(i, j);
            if (t != 0)
                res(i, j) = t;
        }
    return res;
}

SparseMatrix::SparseMatrix(const SparseMatrix& other) : Matrix(other), dok(other.dok){
    sparse = other.sparse;
}

SparseMatrix::SparseMatrix(SparseMatrix &&other) noexcept {
    swap(*this, other);
}

void swap(SparseMatrix &sm1, SparseMatrix &sm2) {
    using std::swap;
    swap(sm1.dok, sm2.dok);
    swap(sm1.sparse, sm2.sparse);
    swap(sm1.n, sm2.n);
    swap(sm1.m, sm2.m);
}

SparseMatrix &SparseMatrix::operator=(SparseMatrix other) {
    swap(*this, other);
    return (*this);
}



