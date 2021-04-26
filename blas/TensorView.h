#ifndef BLAS_TENSORVIEW_H_
#define BLAS_TENSORVIEW_H_

#include "Tensor.h"

namespace blas {
/**
 * A view of a tensor - doesn't copy data from original tensor.
 * @tparam T : stored data type.
 */
template <typename T>
class TensorView : public Tensor<T> {
    friend class Tensor<T>;

    TensorView(T* data, const std::vector<size_t>&);

    explicit TensorView(Tensor<T> t);

   public:
    ~TensorView() override = default;  // Doesn't delete the data.
    using eiterator = T*;
    using ceiterator = const T*;

    static T& get(TensorView<T>& tv, size_t true_idx) {
        return tv.data[true_idx];
    }
    static T get(const TensorView<T>& tv, size_t true_idx) {
        return tv.data[true_idx];
    }

    inline Tensor<T> contiguous() const override {
        Tensor<T> t(this->shape);
        t.copy_(*this);
        return t;
    }

    // We don't need this here because it operates the same exact way as for
    // Tensor.
    // // Declares all interactions between the different tensor types.
    // MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE,
    // TensorView, T)

    DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorView)
    DEF_COPY_FILL_TEMPLATES(TensorView, T)
};
}
#endif // BLAS_TENSORVIEW_H_