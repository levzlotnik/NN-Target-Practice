//
// Created by LevZ on 7/11/2020.
//

#include "Tensor.h"

// TODO - implement

namespace blas {
    /**
     * Applies an element-wise function on all elements and stores into destination.
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor1 &apply_(Tensor1 &dst, const Tensor2 &src, std::function<T(T, T)> op) {

    }

    /**
     * Applies an element-wise function on all elements and stores into destination, using broadcast mechanics.
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor1 &apply_broadcast_(Tensor1 &dst, const Tensor2 &src, std::function<T(T, T)> op) {

    }

    /**
     * Applies an element-wise function on all elements and stores into a new tensor..
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor<T> apply(const Tensor1 &src1, const Tensor2 &src2, std::function<T(T, T)> op) {

    }

    /**
     * Applies an element-wise function on all elements and stores into a new tensor, using broadcast mechanics.
     * @tparam Tensor1
     * @tparam Tensor2
     * @tparam T
     * @param dst
     * @param src
     * @param op
     * @return
     */
    template<class Tensor1, class Tensor2, typename T>
    Tensor<T> apply_broadcast(const Tensor1 &src1, const Tensor2 &src2, std::function<T(T, T)> op) {

    }

}