//
// Created by LevZ on 6/14/2020.
//

#include "Matrix.h"
#include <algorithm>
#include <string>
#include <cassert>
#include <unordered_set>

Matrix::Matrix(int n, int m, bool sparse) : n(n), m(m), sparse(sparse) {
    if (n <= 0 || m <= 0)
        throw runtime_error("Matrix sizes must be positive.");
    if(sparse)
        data = nullptr;
    else
        data = new float[n*m];
}

Matrix::Matrix(int n, int m, float init): Matrix(n, m, false) {
    std::fill_n(data, n*m, init);
}

Matrix::Matrix(const Matrix &other):
    Matrix(other.n, other.m, other.sparse)
{
    if (sparse){
        sparse_dok = unordered_map<int, float>(other.sparse_dok);
    }
    else {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                data[i * m + j] = other.at(i, j);
    }
}

Matrix::~Matrix() {
    delete [] data;
}

ostream &operator<<(ostream &os, const Matrix& matrix) {
    os << '[';
    for(int i=0; i<matrix.n; ++i){
        os << '[';
        for(int j=0; j<matrix.m; ++j){
            os << matrix.at(i, j);
            if(j < matrix.m - 1)
                os << ", ";
        }
        os << ']';
        if(i < matrix.n - 1)
            os << ", " << endl;
    }
    os << ']';
    return os;
}

Matrix &Matrix::apply_(UnaryOperation op) {
    if (sparse) {
        for (auto& [k, v]: sparse_dok)
            v = op(v);
    }
    else {
        for (int i = 0; i < n * m; ++i) {
            data[i] = op(data[i]);
        }
    }
    return (*this);
}

Matrix &Matrix::apply_(const Matrix& other, BinaryOperation op) {
    check_shapes(other);
    if (sparse)
        throw runtime_error("Inplace application with another matrix as an operator is unsupported for "
                            "SparseMatrix. Convert to "
                            "non-sparse representation with "
                            "'.to_dense()' to allow inplace application with another matrix.");

    for(int i=0; i < n*m; ++i){
        data[i] = op(data[i], other.data[i]);
    }
    return (*this);
}

Matrix &Matrix::apply_(float scalar, BinaryOperation op) {
    if (sparse) {
        for (auto& [k, v]: sparse_dok)
            v = op(v, scalar);
    }
    else {
        for (int i = 0; i < n * m; ++i) {
            data[i] = op(data[i], scalar);
        }
    }
    return (*this);
}

Matrix Matrix::apply(UnaryOperation op) const {
    return Matrix(*this).apply_(op);
}

Matrix Matrix::apply(const Matrix &other, BinaryOperation op) const {
    return Matrix(*this).apply_(other, op);
}

Matrix Matrix::apply(float scalar, BinaryOperation op) const {
    return Matrix(*this).apply_(scalar, op);
}

#define DEF_MATRIX_OPERATOR_MATRIX_INPLACE(op) \
    Matrix& Matrix::operator op(const Matrix& other) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply_(other, oper); \
    }

#define DEF_MATRIX_OPERATOR_SCALAR_INPLACE(op) \
    Matrix& Matrix::operator op(float scalar) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply_(scalar, oper); \
    }

#define DEF_MATRIX_OPERATOR_INPLACE(op) \
    DEF_MATRIX_OPERATOR_MATRIX_INPLACE(op) \
    DEF_MATRIX_OPERATOR_SCALAR_INPLACE(op)

#define DEF_MATRIX_OPERATOR_MATRIX(op) \
    Matrix Matrix::operator op(const Matrix& other) const { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply(other, oper); \
    }

#define DEF_MATRIX_OPERATOR_SCALAR(op) \
    Matrix Matrix::operator op(float scalar) const { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply(scalar, oper); \
    }

#define DEF_SCALAR_OPERATOR_MATRIX(op) \
    Matrix operator op(float scalar, const Matrix& matrix) { \
        const BinaryOperation oper = [](float& x, float& y) {return y op x;}; \
        return matrix.apply(scalar, oper); \
    }

#define DEF_MATRIX_OPERATOR(op) \
    DEF_MATRIX_OPERATOR_MATRIX(op) \
    DEF_MATRIX_OPERATOR_SCALAR(op) \
    DEF_SCALAR_OPERATOR_MATRIX(op)

// Define all basic element wise operations!
MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_MATRIX_OPERATOR_INPLACE)
MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_MATRIX_OPERATOR)

void Matrix::check_shapes(const Matrix &other) const {
    if(n != other.n || m != other.m)
        throw runtime_error("Shapes mismatch: " + str_shape() + ", " + other.str_shape());
}

string Matrix::str_shape() const{
    return "(" + to_string(n) + ", " + to_string(m) + ")";
}

float Matrix::reduce(float init_value, BinaryOperation op) {
    for (int i=0; i < n; ++i)
        for (int j=0; j < m; ++j)
            init_value = op(init_value, this->at(i, j));
    return init_value;
}

float Matrix::sum() {
    if (sparse){
        float s = 0;
        for (auto [k, v] : sparse_dok)
            s += v;
        return s;
    }
    return reduce(0, [](float& x, float& y){return x+y;});
}

float Matrix::prod() {
    if (sparse && sparse_dok.size() != n*m)
        return 0;
    return reduce(1, [](float& x, float& y){return x*y;});
}

float Matrix::trace() {
    if(n!=m)
        throw runtime_error("Matrix isn't square (n,m should be equal), shape= " + str_shape());
    float t = 0;
    for(int i=0; i<n; i+=n)
        t += this->at(i, i);
    return t;
}

pair<int, int> denorm_index(int idx, int n, int m) {
    if (idx >= n*m)
        throw out_of_range(to_string(idx) + " >= " + to_string(n*m) + ".");
    if (idx < 0)
        throw out_of_range(to_string(idx) + " <= 0");
    return {idx / m, idx % m};
}

Matrix Matrix::transpose() {
    Matrix res(m, n, this->sparse);
    if (this->sparse){
        for (auto [k, v]: sparse_dok){
            auto [i, j] = denorm_index(k, n, m);
            res.at(j, i) = v;
        }
    }
    else {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                res.at(j, i) = this->at(i, j);
    }
    return res;
}

Matrix Matrix::zeros(int n, int m) {
    return Matrix(n, m, 0.0f);
}

Matrix Matrix::ones(int n, int m) {
    return Matrix(n, m, 1.0f);
}

Matrix Matrix::zeros_like(const Matrix &matrix) {
    return zeros(matrix.n, matrix.n);
}

Matrix Matrix::ones_like(const Matrix &matrix) {
    return ones(matrix.n, matrix.n);
}

Matrix Matrix::eye(int n, bool sparse) {
    Matrix res(n, n, true);
    for (int i = 0; i < n; ++i)
        res.at(i, i) = 1;
    if (sparse)
        return res;
    return res.to_dense();
}

void swap(Matrix &m1, Matrix &m2) noexcept {
    using std::swap;
    swap(m1.data, m2.data);
    swap(m1.n, m2.n);
    swap(m1.m, m2.m);
    swap(m1.sparse, m2.sparse);
    swap(m1.sparse_dok, m2.sparse_dok);
}

Matrix &Matrix::operator=(Matrix other) {
    swap(*this, other);
    return (*this);
}

Matrix::Matrix(Matrix &&other) noexcept : Matrix() {
    swap(*this, other);
}

Matrix::Matrix() : n(0), m(0), data(nullptr), sparse(false){

}

float Matrix::mean() {
    return sum() / float(m*n);
}

Vector Matrix::flatten(bool copy) {
    return Vector(n*m, data, false, copy);
}

Matrix Matrix::reshape(int new_n, int new_m) {
    Matrix matrix(*this);
    matrix.n = new_n;
    matrix.m = new_m;
    return matrix;
}

Vector Matrix::get_row(int i) {
    return Vector(m, &this->at(i, 0), false, true);
}

void Matrix::set_row(int i, const Vector &row) {
    if (row.n != m)
        throw runtime_error("Shape mismatch: "+ to_string(m)+ ", " + to_string(row.n));
    for (int j = 0; j < m; ++j)
        this->at(i, j) = row.at(j);

}

Vector Matrix::reduce_axis(float init_value, int axis, BinaryOperation op) {
    if (axis != 0 && axis != 1)
        throw out_of_range("Axes are only 0, 1.");
    int k = axis == 0 ? n : m;
    int t = axis == 0 ? m : n;
    auto fn = axis == 0 ? &Matrix::get_row : &Matrix::get_col;
    Vector res(t, init_value);
    for (int i = 0; i < k; ++i)
        res.apply_((this->*fn)(i), op);
    return res;
}

Vector Matrix::get_col(int i) {
    Vector res(n);
    for(int j = 0; j < n; ++j)
        res[j] = this->at(j, i);

    return res;
}

Vector Matrix::sum(int axis) {
    return reduce_axis(0, axis, [](float& x, float& y) {return x+y;});
}

Vector Matrix::mean(int axis) {
    float k = (axis == 0 ? n : m);
    return sum(axis) / k;
}

float Matrix::std() {
    return flatten(false).std();
}

Vector Matrix::var(int axis) {
    if (axis != 0 && axis != 1)
        throw out_of_range("Axes are only 0, 1.");
    int k = axis == 0 ? n : m;
    int t = axis == 0 ? m : n;
    auto fn = axis == 0 ? &Matrix::get_row : &Matrix::get_col;
    Vector M2(t, 0.0), mu((this->*fn)(0)), delta(t);
    for (int i = 1; i < k; ++i) {
        auto v = (this->*fn)(i);
        delta = v - mu;
        mu += delta / float(i+1);
        M2 += (v - mu) * delta;
    }
    return M2 / float(k);
}

float Matrix::var() {
    return flatten(false).var();
}

Vector Matrix::std(int axis) {
    return this->var(axis).apply_([](float& x){return sqrt(x);});
}

Vector Matrix::get_diag() const {
    Vector res(min(n, m));
    for(int i=0; i < res.n; ++i)
        res[i] = this->at(i, i);
    return res;
}

Matrix &Matrix::set_diag(Vector diag) {
    if (diag.n != min(n, m))
        throw out_of_range("Shape mismatch: attempting to set matrix diagonal of shape (" +
            to_string(min(n, m)) + ") with vector of shape (" + to_string(diag.n) + ").");
    for (int i=0; i < diag.n; ++i)
        this->at(i, i) = diag[i];
    return (*this);
}

Matrix Matrix::operator-() const {
    return apply([](float& x){return -x;});
}

Matrix *Matrix::clone() const {
    return new Matrix(*this);
}

Matrix Matrix::diag(const Vector &v, bool sparse) {
    Matrix res;
    if (sparse)
        res = Matrix(v.n, v.n, true);
    else
        res = Matrix(v.n, v.n, 0.0f);
    res.set_diag(v);
    return res;
}

Matrix Matrix::T() {
    return transpose();
}

Matrix::Matrix(initializer_list<initializer_list<float>> list2d) :
        Matrix(list2d.size(), list2d.begin()->size(), false)
{
    // Check sizes identical:
    for (auto list1d: list2d) {
        if (list1d.size() != m)
            throw invalid_argument("The initializer lists must have uniform shapes.");
    }
    // Fill in matrix:
    int i=0, j=0;
    for (auto l1d : list2d) {
        j = 0;
        for (auto x: l1d) {
            data[i*m + j] = x;
            ++j;
        }
        ++i;
    }
}

Matrix Matrix::to_dense() {
    if (!sparse)
        return Matrix(*this);
    Matrix res(n, m, 0.0f);
    for (auto [idx, v]: sparse_dok){
        auto [i, j] = denorm_index(idx, n, m);
        res.at(i, j) = v;
    }
    return res;
}

Matrix Matrix::to_sparse() {
    if (sparse)
        return Matrix(*this);
    Matrix res(n, m, true);
    float v;
    for (int i=0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if ((v=this->at(i, j)) != 0)
                res.at(i, j) = v;

    return res;
}

Matrix matmul(const Matrix &mat1, const Matrix &mat2) {
    if (mat1.m != mat2.n)
        throw runtime_error("Shape mismatch for matrix "
                            "multiplication: " + mat1.str_shape() + ", " + mat2.str_shape());

    if (mat1.sparse && mat2.sparse) {
        return Matrix::sparse_matmul(mat1, mat2).to_dense();
    }
    Matrix res(mat1.n, mat2.m, 0.0f);
    if (mat1.sparse) {
        for (auto [idx, val]: mat1.sparse_dok){
            auto [i, k] = denorm_index(idx, mat1.n, mat1.m);
            for (int j=0; j < res.m; ++j)
                res(i, j) += val * mat2(k, j);
        }
    }
    else if (mat2.sparse){
        for (auto [idx, val]: mat2.sparse_dok){
            auto [k, j] = denorm_index(idx, mat2.n, mat2.m);
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

Matrix Matrix::sparse_matmul(Matrix sm1, Matrix sm2) {
    Matrix res(sm1.n, sm2.m);
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
        for (auto [idx, v] : sm2.sparse_dok)
            cols.insert(denorm_index(idx, sm2.n, sm2.m).second);

        for (auto [idx, v] : sm1.sparse_dok){
            auto [i, k] = denorm_index(idx, sm1.n, sm1.m);
            for (auto j: cols)
                res(i, j) += v * sm2(k, j);
        }
    }
    else {
        // Preprocess columns to exclude 0 columns of sm2.
        unordered_set<int> rows;
        for (auto [idx, v] : sm1.sparse_dok)
            rows.insert(denorm_index(idx, sm1.n, sm1.m).first);

        for (auto [idx, v] : sm2.sparse_dok){
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
        for (auto[idx, val]: m.sparse_dok){
            auto[i, j] = denorm_index(idx, m.n, m.m);
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
        for (auto[idx, val]: m.sparse_dok) {
            auto[i, j] = denorm_index(idx, m.n, m.m);
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

Matrix::Matrix(const vector<Vector>& list_vectors, bool sparse) :
    Matrix(list_vectors.size(), list_vectors[0].n, sparse)
{
    for (int i=1; i < n; ++i)
        if (list_vectors[i].n != m)
            throw invalid_argument("The rows must have uniform shapes.");

    float v;
    for (int i=0; i < n; ++i)
        for (int j=0; j < m; ++j)
            if ((v=list_vectors[i][j]) != 0 || !sparse)
                at(i, j) = v;

}
