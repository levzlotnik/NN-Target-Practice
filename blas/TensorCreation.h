//
// Created by LevZ on 9/11/2020.
//

#ifndef TARGETPRACTICE_TENSORCREATION_H
#define TARGETPRACTICE_TENSORCREATION_H

#include "all_tensors.h"

namespace blas
{
    template<typename T>
    inline Tensor<T> zeros(const shape_t& shape) {
        return Tensor<T>(shape).fill_(T(0));
    }

    template<typename T>
    inline Tensor<T> zeros_like(const Tensor<T>& t) {
        return zeros<T>(t.shape);
    }

    template<typename T>
    inline Tensor<T> ones(const shape_t& shape) {
        return Tensor<T>(shape).fill_(T(1));
    }

    template<typename T>
    inline Tensor<T> ones_like(const Tensor<T>& t) {
        return ones<T>(t.shape);
    }

    template<typename T>
    inline Tensor<T> linspace(T s, T f, size_t num=50) {
        T step_size = (f - s) / T(num -1);
        shape_t shape = {num};
        Tensor<T> ret(shape);
        T v = s;
        for (int i=0; i < num; ++i, v += step_size)
            Tensor<T>::get(ret, i) = v;
        return ret;
    }

    template<typename T>
    inline Tensor<T> arange(T s, T f, T step=1.) {
        using std::to_string;
        long num = ::floor((f-s) / step);
        if (num < 1)
            throw std::runtime_error("Invalid step - can't get to " + to_string(s) + " from " + to_string(f) + ".");
        shape_t shape = {size_t(num)};
        Tensor<T> ret(shape);
        T v = s;
        for (int i=0; i < num; ++i, v += step)
            Tensor<T>::get(ret, i) = v;
        return ret;
    };

    void seed(uint64_t s);

    template<typename T>
    inline Tensor<T> uniform(T lower, T upper, const shape_t& shape);

    template<typename T>
    inline Tensor<T> randn(T mu, T sigma, const shape_t& shape);

    template<typename T>
    inline Tensor<T> randn(const shape_t& shape) {
        return randn(T(0), T(1), shape);
    }

    template<typename T>
    inline Tensor<T> uniform(const Tensor<T>& lower, const Tensor<T>& upper, const shape_t& shape) NOT_IMPLEMENTED

    template<typename T>
    inline Tensor<T> randn(const Tensor<T>& mu, const Tensor<T>& sigma, const shape_t& shape) NOT_IMPLEMENTED

    template<typename T>
    inline Tensor<T> uniform_like(const Tensor<T>& t, T lower = 0, T upper = 1){
        return uniform<T>(lower, upper, t.shape);
    }

    template<typename T>
    inline Tensor<T> randn_like(const Tensor<T>& t, T mu = 0, T sigma = 1) {
        return randn<T>(mu, sigma, t.shape);
    }
}

#endif //TARGETPRACTICE_TENSORCREATION_H
