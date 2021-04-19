#ifndef TENSORTRANSPOSED_H
#define TENSORTRANSPOSED_H

#include "Tensor.h"

namespace blas {
template <typename T>
class TensorTransposed : public Tensor<T> {
    friend class Tensor<T>;
    const shape_t old_strides;

   public:
    ~TensorTransposed() override = default;  // doesn't delete data
    TensorTransposed(const Tensor<T>& t, const shape_t& permute_indexes) : 
        Tensor<T>::Tensor(t.data, t.shape), old_strides(t.strides)
    {
        size_t new_shape(t.dim());
        size_t new_strides(t.dim());
        for (int i = 0; i < t.dim(); ++i){
            size_t new_i = permute_indexes[i];
            Tensor<T>::strides[i] = old_strides[new_i];
            Tensor<T>::shape[i] = t.shape[new_i];
        }
    }
    TensorTransposed(Tensor<T>& t, const shape_t& permute_indexes): 
        Tensor<T>::Tensor(t.data, t.shape), old_strides(t.strides)
    {
        size_t new_shape(t.dim());
        size_t new_strides(t.dim());
        for (int i = 0; i < t.dim(); ++i){
            size_t new_i = permute_indexes[i];
            Tensor<T>::strides[i] = old_strides[new_i];
            Tensor<T>::shape[i] = t.shape[new_i];
        }
    }

    template <typename T_>
    struct elem_iterator {
        T_* data;
        SliceGroup::const_iterator sg_iterator;
        const shape_t shape;
        const shape_t strides;

        using difference_type = ptrdiff_t;
        using value_type = T_;
        using reference = T_&;
        using pointer = T_*;
        using iterator_category = std::forward_iterator_tag;
        
        inline reference operator*() {
            const auto& idx = *sg_iterator;
            auto true_idx = ravel_index(idx, shape, strides);
            return data[true_idx];
        }

        inline elem_iterator& operator++() {
            ++sg_iterator;
            return *this;
        }

        inline elem_iterator operator++(int) {
            elem_iterator temp(*this);
            ++(*this);
            return temp;
        }

        inline bool operator==(const elem_iterator& other) {
            return sg_iterator == other.sg_iterator;
        }

        inline bool operator!=(const elem_iterator& other) {
            return !(*this == other);
        }
    };
    using eiterator = elem_iterator<T>;
    using ceiterator = elem_iterator<const T>;
    inline eiterator elem_begin(TensorTransposed& ts) {

    }
    static inline eiterator elem_end(TensorTransposed& ts);
    static inline ceiterator const_elem_begin(const TensorTransposed& ts);
    static inline ceiterator const_elem_end(const TensorTransposed& ts);
};
}
#endif // TENSORTRANSPOSED_H