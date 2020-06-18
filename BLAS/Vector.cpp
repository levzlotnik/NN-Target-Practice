//
// Created by LevZ on 6/14/2020.
//

#include "Vector.h"
#include <algorithm>

Vector::Vector() : n(0), data(nullptr){

}

Vector::Vector(int n, float* d, bool create_if_null, bool copy) : n(n){
    if (create_if_null && d == nullptr){
        data = new float[n];
        return;
    }
    if (copy) {
        if (d == nullptr && n > 0)
            throw runtime_error("Cannot copy from nullptr.");
        data = new float[n];
        for (int i = 0; i < n; ++i)
            data[i] = d[i];
    }
    else {
        data = d;
        delete_data = false;
    }
}

Vector::Vector(int n, float init): Vector(n) {
    std::fill_n(data, n, init);
}

Vector::Vector(const Vector &other): Vector(other.n, other.data, false, true) {

}

void swap(Vector &v1, Vector &v2) noexcept {
    using std::swap;
    swap(v1.data, v2.data);
    swap(v1.n, v2.n);
    swap(v1.delete_data, v2.delete_data);
}

Vector::Vector(Vector &&other) noexcept : Vector() {
    swap(*this, other);
}


Vector &Vector::operator=(Vector other) {
    swap(*this, other);
    return (*this);
}

Vector::~Vector() {
    if (delete_data)
        delete [] data;
}

void Vector::check_shapes(const Vector &other) const {
    if(n != other.n)
        throw runtime_error("Shape mismatch: " + to_string(n) + ", " + to_string(other.n));
}

ostream &operator<<(ostream& os, const Vector& vector) {
    os << '[';
    for (int i = 0; i < vector.n; ++i){
        os << vector.data[i];
        if (i < vector.n - 1)
            os << ", ";
    }
    os << ']';
    return os;
}

Vector &Vector::apply_(UnaryOperation op) {
    for (int i=0; i < n; ++i)
        data[i] = op(data[i]);
    return (*this);
}

Vector &Vector::apply_(const Vector &other, BinaryOperation op) {
    check_shapes(other);
    for (int i = 0; i < n; ++i)
        data[i] = op(data[i], other.data[i]);
    return (*this);
}

Vector &Vector::apply_(float scalar, BinaryOperation op) {
    for (int i = 0; i < n; ++i)
        data[i] = op(data[i], scalar);
    return (*this);
}

Vector Vector::apply(UnaryOperation op) const {
    Vector res(n);
    for (int i = 0; i < n; ++i)
        res.data[i] = op(data[i]);
    return res;
}

Vector Vector::apply(const Vector &other, BinaryOperation op) const{
    check_shapes(other);
    Vector res(n);
    for (int i = 0; i < n; ++i)
        res.data[i] = op(data[i], other.data[i]);
    return res;
}

Vector Vector::apply(float scalar, BinaryOperation op) const {
    Vector res(n);
    for (int i = 0; i < n; ++i)
        res.data[i] = op(data[i], scalar);
    return res;
}

#define DEF_VECTOR_OPERATOR_VECTOR_INPLACE(op) \
    Vector& Vector::operator op(const Vector& other) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply_(other, oper); \
    }

#define DEF_VECTOR_OPERATOR_SCALAR_INPLACE(op) \
    Vector& Vector::operator op(float scalar) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply_(scalar, oper); \
    }

#define DEF_VECTOR_OPERATOR_INPLACE(op) \
    DEF_VECTOR_OPERATOR_VECTOR_INPLACE(op) \
    DEF_VECTOR_OPERATOR_SCALAR_INPLACE(op)

#define DEF_VECTOR_OPERATOR_VECTOR(op) \
    Vector Vector::operator op(const Vector& other) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply(other, oper); \
    }

#define DEF_VECTOR_OPERATOR_SCALAR(op) \
    Vector Vector::operator op(float scalar) { \
        const BinaryOperation oper = [](float& x, float& y) {return x op y;}; \
        return apply(scalar, oper); \
    }

#define DEF_SCALAR_OPERATOR_VECTOR(op) \
    Vector operator op(float scalar, const Vector& vector) { \
        Vector res(vector.n);\
        for (int i=0; i<res.n; ++i) \
            res.data[i] = scalar op vector.data[i]; \
        return res; \
    }

#define DEF_VECTOR_OPERATOR(op) \
    DEF_VECTOR_OPERATOR_VECTOR(op) \
    DEF_VECTOR_OPERATOR_SCALAR(op) \
    DEF_SCALAR_OPERATOR_VECTOR(op)

// Define all basic element wise operations!
MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DEF_VECTOR_OPERATOR_INPLACE)
MACRO_BASIC_ARITHMETIC_OPERATORS(DEF_VECTOR_OPERATOR)

float Vector::dot(const Vector &other) {
    check_shapes(other);
    float res = 0;
    for (int i = 0; i < n; ++i)
        res += data[i] * other.data[i];
    return res;
}

float Vector::reduce(BinaryOperation op, float init_val) {
    for (int i = 0; i < n; ++i)
        init_val = op(init_val, data[i]);
    return init_val;
}

float Vector::sum() {
    return reduce([](float& x, float& y){return x+y;});
}

Vector Vector::zeros(int n) {
    return Vector(n, 0.0);
}

Vector Vector::ones(int n) {
    return Vector(n, 1.0);
}

Vector Vector::zeros_like(const Vector &other) {
    return zeros(other.n);
}

Vector Vector::ones_like(const Vector &other) {
    return ones(other.n);
}


float Vector::mean() {
    return sum() / n;
}

float Vector::std() {
    return sqrt(var());
}

float Vector::var() {
    // Welford's online algorithm
    if (n < 2)
        return 0;
    float M2 = 0, mu = data[0], delta;
    for (int i=1; i < n; ++i){
        delta = data[i] - mu;
        mu += delta / float(i+1);
        M2 += (data[i] - mu) * delta;
    }
    return M2 / float(n);
}

float *Vector::begin() {
    return data;
}

float *Vector::end() {
    return data+n;
}

float *Vector::rbegin() {
    return data + n - 1;
}

float *Vector::rend() {
    return data - 1;
}

Vector Vector::arange(float a, float b, float step) {
    int n_steps = floor((b - a) / step);
    if (n_steps < 1)
        throw runtime_error("Invalid step - can't get to " + to_string(b) + " from " + to_string(a) + ".");
    Vector res(n_steps);
    for (float& x : res){
        x = a;
        a += step;
    }
    return res;
}

Vector Vector::linspace(float a, float b, int num) {
    float step = (b-a) / float(num);
    return arange(a, b, step);
}

Vector Vector::concat(std::vector<Vector> vectors) {
    int size = std::accumulate(vectors.begin(), vectors.end(), 0,
            [](int a, Vector v){return a + v.n;});
    Vector res(size);
    float* beg = res.begin();
    for (auto vec : vectors){
        std::copy(vec.begin(), vec.end(), beg);
        beg += vec.n;
    }
    return res;
}

Vector Vector::slice(int beg, int end, int step) const {
    if (step == 0)
        throw runtime_error("Cannot use step = 0.");
    beg = normalize_index(beg, n);
    end = normalize_index(end, n);
    int size = (end - beg) / step;
    if (size == 0)
        throw runtime_error("The slice size cannot be 0.");
    if (size < 0)
        throw runtime_error("The slice and step directions contradict each other: slice direction =" +
            to_string(end-beg) + ", step = " + to_string(step));
    Vector res(size);
    for (int i=0; i < size; ++i)
        res.data[i] = this->data[beg + i*step];
    return res;
}

void Vector::sliced_set(Vector v, int beg, int end, int step) {
    if (step == 0)
        throw runtime_error("Cannot use step = 0.");
    beg = normalize_index(beg, n);
    end = normalize_index(end, n);
    int size = (end - beg) / step;
    if (size == 0)
        throw runtime_error("The slice size cannot be 0.");
    if (size < 0)
        throw runtime_error("The slice and step directions contradict each other: slice direction =" +
                            to_string(end-beg) + ", step = " + to_string(step));
    for (int i = 0; i < v.n; ++i)
        data[beg + i*step] = v.data[i];
}


