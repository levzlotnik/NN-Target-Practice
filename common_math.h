//
// Created by LevZ on 6/27/2020.
//

#ifndef TARGETPRACTICE_COMMON_MATH_H
#define TARGETPRACTICE_COMMON_MATH_H

#include <string>
#include <unordered_map>
#include <cmath>

using std::pair;
using std::tuple;
template<typename T>
inline T relu(T x) { return x > 0 ? x : 0; }

template<typename T>
inline T absl(T x) { return x > 0 ? x : -x; }

// Apply a macro to all math functions.
//   Note: Only add a macro function here if you can add it to the
//         unary_elemwise_mapping!
#define MACRO_MATH_FUNCTIONS(macro) \
    macro(sin) \
    macro(cos) \
    macro(tan) \
    macro(sqrt) \
    macro(log) \
    macro(log2) \
    macro(log10) \
    macro(log1p) \
    macro(absl) \
    macro(exp2) \
    macro(floor) \
    macro(ceil) \
    macro(round) \
    macro(exp) \
    macro(relu)

#define MACRO_BASIC_ARITHMETIC_OPERATORS(macro) \
    macro(+) \
    macro(-) \
    macro(*) \
    macro(/)

#define MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(macro) \
    macro(+=) \
    macro(-=) \
    macro(*=) \
    macro(/=)

#define MACRO_BASIC_ARITHMETIC_OPERATORS_NAMED(macro) \
    macro(+, "add")                         \
    macro(-, "sub")                         \
    macro(*, "mul")                         \
    macro(/, "div")


namespace common_math {

    template<typename T> using unary_op = std::function<T(T)>;
    template<typename T> using binary_op = std::function<T(T, T)>;

    template<typename T>
    struct unary_func_data {

#define DEF_STATIC_UNARY_FUNC(func) \
    static inline constexpr T func(T x) { return ::func(x); }

        MACRO_MATH_FUNCTIONS(DEF_STATIC_UNARY_FUNC)

        static pair<unary_op<T>, unary_op<T>> get_function_data(std::string func_name) {
            using namespace std;
            static const T ln2 = unary_func_data::log(2);
            static const T ln10 = unary_func_data::log(10);
            static const T half = 0.5;
            static const unordered_map<string, pair<unary_op<T>, unary_op<T>>>
                    unary_elemwise_mapping
                    {
                            /* {func,       {name, derivative}} */
                            {"sin",   {unary_func_data::sin,   [](T x) -> T { return cos(x); }}},
                            {"cos",   {unary_func_data::cos,   [](T x) -> T { return -sin(x); }}},
                            {"tan",   {unary_func_data::tan,   [](T x) -> T { return 1 / (cos(x) * cos(x)); }}},
                            {"sqrt",  {unary_func_data::sqrt,  [=](T x) -> T { return half / sqrt(x); }}},
                            {"log",   {unary_func_data::log,   [](T x) -> T { return 1 / x; }}},
                            {"log2",  {unary_func_data::log2,  [=](T x) -> T {
                                return 1 / (x * ln2);
                            }}},
                            {"log10", {unary_func_data::log10, [=](T x) -> T {
                                return 1 / (x * ln10);
                            }}},
                            {"log1p", {unary_func_data::log1p, [](T x) -> T { return 1 / (1 + x); }}},
                            {"absl",  {unary_func_data::absl,  [](T x) -> T { return x < 0 ? T(-1) : T(1); }}},
                            {"exp",   {unary_func_data::exp,   unary_func_data::exp}},
                            {"exp2",  {unary_func_data::exp2,  [=](T x) -> T {
                                return ln2 * exp2(x);
                            }}},
                            {"relu",  {unary_func_data::relu,  [](T x) -> T { return x > 0 ? T(1) : 0; }}}
                            // TODO - implement more, if needed.
                    };
            return unary_elemwise_mapping.at(func_name);
        }
    };

    template
    struct unary_func_data<double>;
    template
    struct unary_func_data<float>;

    template<typename T>
    using jac_binary_op = std::function<T(T in1, T in2, T out)>;

    template<typename T>
    struct binary_func_data {
        static inline T add(T x, T y) { return x + y; };

        static inline T mul(T x, T y) { return x * y; };

        static inline T sub(T x, T y) { return x - y; };

        static inline T div(T x, T y) { return x / y; };

        static inline T pow(T x, T y) { return ::pow(x, y); }

        static tuple<binary_op<T>, jac_binary_op<T>, jac_binary_op<T>> get_function_data(const std::string& func_name) {
            using u = unary_func_data<T>;
            static const std::unordered_map<std::string,
                         tuple<binary_op<T>, jac_binary_op<T>, jac_binary_op<T>>>
                    binary_elemwise_mapping =
                    {
                            {"add", {add,
                                            [](T i1, T i2, T o) -> T { return 1; },
                                            [](T i1, T i2, T o) -> T { return 1; }}},
                            {"mul", {mul,
                                            [](T i1, T i2, T o) -> T { return i2; },
                                            [](T i1, T i2, T o) -> T { return i1; }}},
                            {"sub", {sub,
                                            [](T i1, T i2, T o) -> T { return 1; },
                                            [](T i1, T i2, T o) -> T { return -1; }}},
                            {"div", {div,
                                            [](T i1, T i2, T o) -> T { return 1 / i2; },
                                            [](T i1, T i2, T o) -> T { return -o / i2; }}},
                            {"pow", {pow,
                                            [](T i1, T i2, T o) -> T { return o * i2 / i1;},
                                            [](T i1, T i2, T o) -> T { return o * u::log(i1); }}}
                    };
            return binary_elemwise_mapping.at(func_name);
        }
    };

    template
    struct binary_func_data<double>;
    template
    struct binary_func_data<float>;
};

#endif //TARGETPRACTICE_COMMON_MATH_H
