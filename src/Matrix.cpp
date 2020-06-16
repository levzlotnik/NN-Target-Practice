//
// Created by LevZ on 6/14/2020.
//

#include "../include/Matrix.h"
#include <algorithm>
#include <string>

Matrix::Matrix(int n, int m): n(n), m(m) {
    if (n <= 0 || m <= 0)
        throw runtime_error("Matrix sizes must be positive.");
    data = new float[n*m];
}

Matrix::Matrix(int n, int m, float init): Matrix(n, m) {
    std::fill_n(data, n*m, init);
}

Matrix::Matrix(const Matrix &other): Matrix(other.n, other.m) {
    std::copy_n(other.data, n*m, data);
}

Matrix::~Matrix() {
    delete [] data;
}

ostream &operator<<(ostream &os, const Matrix& matrix) {
    os << '[';
    for(int i=0; i<matrix.n; ++i){
        os << '[';
        for(int j=0; j<matrix.m; ++j){
            os << const_cast<Matrix &>(matrix).at(i, j);
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
    for(int i=0; i < n*m; ++i){
        data[i] = op(data[i]);
    }
    return (*this);
}

Matrix &Matrix::apply_(const Matrix& other, BinaryOperation op) {
    check_shapes(other);
    for(int i=0; i < n*m; ++i){
        data[i] = op(data[i], other.data[i]);
    }
    return (*this);
}

Matrix &Matrix::apply_(float scalar, BinaryOperation op) {
    for(int i=0; i < n*m; ++i){
        data[i] = op(data[i], scalar);
    }
    return (*this);
}

Matrix Matrix::apply(UnaryOperation op) {
    Matrix res(n, m);
    for(int i=0; i<n*m; ++i){
        res.data[i] = op(data[i]);
    }
    return res;
}

Matrix Matrix::apply(const Matrix &other, BinaryOperation op) {
    check_shapes(other);
    Matrix res(n, m);
    for(int i=0; i<n*m; ++i){
        res.data[i] = op(data[i], other.data[i]);
    }
    return res;
}

Matrix Matrix::apply(float scalar, BinaryOperation op) {
    Matrix res(n, m);
    for(int i=0; i<n*m; ++i){
        res.data[i] = op(data[i], scalar);
    }
    return res;
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
    Matrix Matrix::operator op(const Matrix& other) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply(other, oper); \
    }

#define DEF_MATRIX_OPERATOR_SCALAR(op) \
    Matrix Matrix::operator op(float scalar) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply(scalar, oper); \
    }

#define DEF_SCALAR_OPERATOR_MATRIX(op) \
    Matrix operator op(float scalar, const Matrix& matrix) { \
        Matrix res(matrix.n, matrix.m);\
        for (int i=0; i<res.n * res.m; ++i) \
            res.data[i] = scalar op matrix.data[i]; \
        return res; \
    }

#define DEF_MATRIX_OPERATOR(op) \
    DEF_MATRIX_OPERATOR_MATRIX(op) \
    DEF_MATRIX_OPERATOR_SCALAR(op) \
    DEF_SCALAR_OPERATOR_MATRIX(op)

// Define all basic element wise operations!
MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_MATRIX_OPERATOR_INPLACE)
MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_MATRIX_OPERATOR)

void Matrix::check_shapes(const Matrix &other) {
    if(n != other.n || m != other.m)
        throw runtime_error("Shapes mismatch: " + str_shape() + ", " + other.str_shape());
}

string Matrix::str_shape() const{
    return "(" + to_string(n) + ", " + to_string(m) + ")";
}

float Matrix::reduce(float init_value, BinaryOperation op) {
    for (int i = 0; i < n * m ; ++i)
        init_value = op(init_value, data[i]);
    return init_value;
}

float Matrix::sum() {
    return reduce(0, [](float& x, float& y){return x+y;});
}

float Matrix::prod() {
    return reduce(1, [](float& x, float& y){return x*y;});
}

float Matrix::trace() {
    if(n!=m)
        throw runtime_error("Matrix isn't square (n,m should be equal), shape= " + str_shape());
    float t = 0;
    for(int i=0; i<n*n; i+=n)
        t += data[i];
    return t;
}

Matrix Matrix::matmul(const Matrix &other) {
    if (m != other.n)
        throw runtime_error("Shape mismatch for matrix multiplication: " + str_shape() + ", " + other.str_shape());
    Matrix res(n, other.m, 0);
    for (int i=0; i<res.n; ++i)
        for (int j=0; j<res.m; ++j)
            for (int k=0; k < m; ++k)
                res.at(i, j) += this->at(i, k) * const_cast<Matrix&>(other).at(k, j);

    return res;
}

Matrix Matrix::transpose() {
    Matrix res(m, n);
    for (int i =0; i < n; ++i)
        for (int j=0; j < m; ++j)
            res.at(j, i) = this->at(i, j);

    return res;
}

Matrix Matrix::zeros(int n, int m) {
    return Matrix(n, m, 0);
}

Matrix Matrix::ones(int n, int m) {
    return Matrix(n, m, 1);
}

Matrix Matrix::zeros_like(const Matrix &matrix) {
    return zeros(matrix.n, matrix.n);
}

Matrix Matrix::ones_like(const Matrix &matrix) {
    return ones(matrix.n, matrix.n);
}

Matrix Matrix::eye(int n) {
    Matrix res(n, n, 0);
    for (int i = 0; i < n; ++i)
        res.at(i, i) = 1;
    return res;
}

void swap(Matrix &m1, Matrix &m2) noexcept {
    using std::swap;
    swap(m1.data, m2.data);
    swap(m1.n, m2.n);
    swap(m1.m, m2.m);
}

Matrix &Matrix::operator=(Matrix other) {
    swap(*this, other);
    return (*this);
}

Matrix::Matrix(Matrix &&other) noexcept : Matrix() {
    swap(*this, other);
}

Matrix::Matrix() : n(0), m(0), data(nullptr){

}

template<class Distribution, typename ... Argc>
Matrix Matrix::random_sample_seeded(int n, int m, uint32_t seed, Argc ... argc) {
    Matrix res(n, m);
    static default_random_engine generator(seed);
    static Distribution dist(argc...);
    for (int i = 0; i < n*m; ++i)
        res.data[i] = float(dist(generator));
    return res;
}

template<class Distribution, typename ... Argc>
Matrix Matrix::random_sample(int n, int m, Argc ... argc) {
    Matrix res(n, m);
    static random_device generator;
    static Distribution dist(argc...);
    for (int i = 0; i < n*m; ++i)
        res.data[i] = float(dist(generator));
    return res;
}

Matrix Matrix::randn(float mu, float sigma, int n, int m, bool seeded, uint32_t seed) {
    using Dist = normal_distribution<float>;
    if (seeded)
        return random_sample_seeded<Dist>(n, m, seed, mu, sigma);
    return random_sample<Dist>(n, m, mu, sigma);
}

Matrix Matrix::uniform(float lower, float upper, int n, int m, bool seeded, uint32_t seed) {
    using Dist = uniform_real_distribution<float>;
    if (seeded)
        return random_sample_seeded<Dist>(n, seed, lower, upper);
    return random_sample<Dist>(n, m, lower, upper);
}

Matrix Matrix::randint(int lower, int upper, int n, int m, bool seeded, uint32_t seed) {
    using Dist = uniform_int_distribution<int>;
    if (seeded)
        return random_sample_seeded<Dist>(n, m, seed, lower, upper);
    return random_sample<Dist>(n, m, lower, upper);
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
        this->at(i, j) = const_cast<Vector&>(row).at(j);

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



