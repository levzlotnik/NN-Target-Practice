//
// Created by LevZ on 6/14/2020.
//

#ifndef BLAS_VECTOR_H
#define BLAS_VECTOR_H


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
    Vector();

public:
    int n;

    float* begin();
    float* end();
    float* rbegin();
    float* rend();

    explicit Vector(int n, float* data= nullptr, bool create_if_nullptr=true, bool copy=false);
    Vector(int n, float init);

    friend void swap(Vector& v1, Vector& v2) noexcept;
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;

    Vector& operator=(const Vector& other);
    Vector& operator=(Vector&& other) noexcept ;

    ~Vector();

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

    friend ostream& operator<<(ostream& os, const Vector& vector);

    /* Inplace operations */
    Vector& apply_(UnaryOperation op);
    Vector& apply_(const Vector& other, BinaryOperation op);
    Vector& apply_(float scalar, BinaryOperation op);
    /* Out of place operations */
    Vector apply(UnaryOperation op);
    Vector apply(const Vector& other, BinaryOperation op);
    Vector apply(float scalar, BinaryOperation op);


#define DECL_VECTOR_OPERATOR(op) \
    Vector operator op(const Vector& other); \
    Vector operator op(float scalar); \
    friend Vector operator op(float scalar, const Vector& matrix);

#define DECL_VECTOR_OPERATOR_INPLACE(op) \
    Vector& operator op(const Vector& other); \
    Vector& operator op(float scalar);

    // Declare all basic element wise operations!
    MACRO_BASIC_ARITHMETIC_OPERATORS(DECL_VECTOR_OPERATOR)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DECL_VECTOR_OPERATOR_INPLACE)

    float dot(const Vector& other);
    float reduce(BinaryOperation op, float init_val=0);
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

private:
    template <class Distribution, typename ... Argc >
    static Vector random_sample_seeded(int n, uint32_t seed, Argc ... argc);

    template <class Distribution, typename ... Argc >
    static Vector random_sample(int n, Argc ... argc);

public:
    static Vector randn(float mu, float sigma, int n, bool seeded=false, uint32_t seed=0);
    static Vector uniform(float lower, float upper, int n, bool seeded=false, uint32_t seed=0);
    static Vector randint(int lower, int upper, int n, bool seeded=false, uint32_t seed=0);
    
private:
    void check_shapes(const Vector& other) const;

};


#endif //BLAS_VECTOR_H