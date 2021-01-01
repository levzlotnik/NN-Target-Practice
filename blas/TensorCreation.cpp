//
// Created by LevZ on 9/12/2020.
//
#include "TensorCreation.h"
#include <random>

namespace blas {
    template <typename T>
    using _urd = std::uniform_real_distribution<T>;
    template <typename T>
    using _uid = std::uniform_int_distribution<T>;

    template <typename T>
    using _nd = std::normal_distribution<T>;

    template <typename T>
    class uniform_driver_type;
    template <> class uniform_driver_type<double> : public _urd<double> { using _urd<double>::_urd; };
    template <> class uniform_driver_type<float> : public _urd<float> { using _urd<float>::_urd; };
    template <> class uniform_driver_type<long> : public _uid<long> { using _uid<long>::_uid; };

    template <typename T>
    using normal_driver_type = _nd<T>;

    class RandomState {
    public:
        static inline void seed(uint64_t s) {
            instance().engine.seed(s);
        }
        static inline std::mt19937_64& get_engine() {
            return instance().engine;
        }

    private:
        static RandomState& instance() {
            static RandomState _instance;
            return _instance;
        }
        RandomState() : engine{std::random_device()()} {}
        std::mt19937_64 engine;
    };


    inline void seed(uint64_t s) {
        RandomState::seed(s);
    }

    template<typename T>
    Tensor<T> uniform(T lower, T upper, const shape_t &shape) {
        Tensor<T> ret(shape);
        uniform_driver_type<T> driver{lower, upper};
        auto& gen = RandomState::get_engine();
        auto iter = Tensor<T>::elem_begin(ret), end = Tensor<T>::elem_end(ret);
        for (; iter != end; ++iter)
            *iter = driver(gen);
        return ret;
    }

    template<typename T>
    Tensor<T> randn(T mu, T sigma, const shape_t &shape) {
        Tensor<T> ret(shape);
        normal_driver_type<T> driver{mu, sigma};
        auto& gen = RandomState::get_engine();
        auto iter = Tensor<T>::elem_begin(ret), end = Tensor<T>::elem_end(ret);
        for (; iter != end; ++iter)
            *iter = driver(gen);
        return ret;
    }

#define INSTANTIATE_CREATION_OPS(T) \
    template Tensor<T> randn<T>(T, T, const shape_t&); \
    template Tensor<T> uniform<T>(T, T, const shape_t&);

    INSTANTIATE_CREATION_OPS(double)
    INSTANTIATE_CREATION_OPS(float)

}
