//
// Created by LevZ on 6/14/2020.
//

#ifndef TARGETPRACTICE_VECTOR_H
#define TARGETPRACTICE_VECTOR_H


#include <iostream>
#include <string>
#include <functional>
#include "common_blas.h"
#include <random>
#include <type_traits>

using namespace std;

class Vector {
private:
    float* data;
    bool delete_data = true;

public:
    int n;

    float* begin();
    float* end();
    float* rbegin();
    float* rend();

    const float* begin() const;
    const float* end() const;
    const float* rbegin() const;
    const float* rend() const;

    explicit Vector(int n, float* data= nullptr, bool create_if_nullptr=true, bool copy=false);
    Vector(int n, float init);
    Vector(initializer_list<float> list);

    friend void swap(Vector& v1, Vector& v2) noexcept;
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;

    Vector& operator=(Vector other);

    ~Vector();

    inline float item() const {
        if (n > 1)
            throw runtime_error("Can only call `.item()` for a vector of a single value.");
        return data[0];
    }

    inline float operator[](int i) const {
        i = normalize_index(i, n);
        return data[i];
    }

    inline float& operator [](int i) {
        i = normalize_index(i, n);
        return data[i];
    }

    inline float& at(int i){
        return (*this)[i];
    }
    inline float at(int i) const {
        return (*this)[i];
    }

    Vector slice(int beg, int end, int step=1) const;
    void sliced_set(Vector v, int beg, int end, int step=1);

    inline int shape() const { return n; }

    friend ostream& operator<<(ostream& os, const Vector& vector);

    /* Inplace operations */
    Vector& apply_(UnaryOperation op);
    Vector& apply_(const Vector& other, BinaryOperation op);
    Vector& apply_(float scalar, BinaryOperation op);
    /* Out of place operations */
    Vector apply(UnaryOperation op) const;
    Vector apply(const Vector& other, BinaryOperation op) const;
    Vector apply(float scalar, BinaryOperation op) const;

    Vector& pow_(float ex);
    Vector pow(float ex);

    Vector& fill_(float scalar);

#define DECL_VECTOR_OPERATOR(op) \
    Vector operator op(const Vector& other) const; \
    Vector operator op(float scalar) const; \
    friend Vector operator op(float scalar, const Vector& matrix);

#define DECL_VECTOR_OPERATOR_INPLACE(op) \
    Vector& operator op(const Vector& other); \
    Vector& operator op(float scalar);

    // Declare all basic element wise operations!
    MACRO_BASIC_ARITHMETIC_OPERATORS(DECL_VECTOR_OPERATOR)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DECL_VECTOR_OPERATOR_INPLACE)

    float dot(const Vector& other);
    float reduce(BinaryOperation op);
    float sum();
    float mean();
    float var();
    float std();

    static Vector zeros(int n);
    static Vector ones(int n);
    static Vector zeros_like(const Vector& other);
    static Vector ones_like(const Vector& other);
    static Vector arange(float a, float b, float step=1);
    static Vector linspace(float a, float b, int num=50);

    static Vector concat(std::vector<Vector> vectors);

    Vector* clone() const;

    Vector();

private:
    void check_shapes(const Vector& other) const;

};


#endif //TARGETPRACTICE_VECTOR_H
