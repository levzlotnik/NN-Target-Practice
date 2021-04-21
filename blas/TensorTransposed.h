#ifndef BLAS_TENSORTRANSPOSED_H_
#define BLAS_TENSORTRANSPOSED_H_

#include "Tensor.h"

namespace blas {
template <typename T>
class TensorTransposed : public Tensor<T> {
    friend class Tensor<T>;
    const shape_t old_strides;
    SliceGroup sg_convenience;

   public:
    ~TensorTransposed() override = default;  // doesn't delete data
    TensorTransposed(const Tensor<T>& t, const shape_t& permute_indexes)
        : Tensor<T>::Tensor(t.get_data_ptr(), t.shape), old_strides(t.strides) {
        size_t new_shape(t.dim());
        size_t new_strides(t.dim());
        for (int i = 0; i < t.dim(); ++i) {
            size_t new_i = permute_indexes[i];
            this->strides[i] = old_strides[new_i];
            this->shape[i] = t.shape[new_i];
        }
        this->requires_deletion = false;
        sg_convenience = SliceGroup::cover_shape(this->shape);
    }

    TensorTransposed(Tensor<T>& t, const shape_t& permute_indexes)
        : Tensor<T>::Tensor(t.get_data_ptr(), t.shape), old_strides(t.strides) {
        size_t new_shape(t.dim());
        size_t new_strides(t.dim());
        for (int i = 0; i < t.dim(); ++i) {
            size_t new_i = permute_indexes[i];
            this->strides[i] = old_strides[new_i];
            this->shape[i] = t.shape[new_i];
        }
        this->requires_deletion = false;
        sg_convenience = SliceGroup::cover_shape(this->shape);
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
    static inline eiterator elem_begin(TensorTransposed& tt) {
        return eiterator{tt.data, tt.sg_convenience.begin(), tt.shape,
                         tt.strides};
    }
    static inline eiterator elem_end(TensorTransposed& tt) {
        return eiterator{tt.data, tt.sg_convenience.end(), tt.shape,
                         tt.strides};
    }
    static inline ceiterator const_elem_begin(const TensorTransposed& tt) {
        return ceiterator{tt.data, tt.sg_convenience.begin(), tt.shape,
                          tt.strides};
    }
    static inline ceiterator const_elem_end(const TensorTransposed& tt) {
        return ceiterator{tt.data, tt.sg_convenience.begin(), tt.shape,
                          tt.strides};
    }
    static T& get(TensorTransposed<T>& t, size_t true_idx) {
        // Translate true_idx into tranposed idx
        index_t unraveled = unravel_index(true_idx, t.shape, t.size);
        size_t transposed_idx = ravel_index(unraveled, t.shape, t.strides);
        return t.data[transposed_idx];
    }
    static T get(const TensorTransposed<T>& t, size_t true_idx) {
        // Translate true_idx into tranposed idx
        index_t unraveled = unravel_index(true_idx, t.shape, t.size);
        size_t transposed_idx = ravel_index(unraveled, t.shape, t.strides);
        return t.data[transposed_idx];
    }

    DECL_ALL_REDUCE_OVERRIDES()

    MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE,
                                TensorTransposed, T)

    DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorTransposed)
    DEF_COPY_FILL_TEMPLATES(TensorTransposed, T)

    Tensor<T> contiguous() const override;
    
    ostream& print_to_os(ostream& os, bool rec_start) const override;
};
}  // namespace blas
#endif  // BLAS_TENSORTRANSPOSED_H_