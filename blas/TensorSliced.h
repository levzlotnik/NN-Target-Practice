#ifndef BLAS_TENSORSLICED_H_
#define BLAS_TENSORSLICED_H_

namespace blas{
/**
 * A slice of the tensor - doesn't copy data from the original tensor.
 * @tparam T
 */
template <typename T>
class TensorSliced : public Tensor<T> {
   private:
    friend class Tensor<T>;
    shape_t underlying_tensor_shape;
    const size_t underlying_tensor_size;

    TensorSliced(T* data, const shape_t& shape, const SliceGroup& slice_group);

    TensorSliced(const Tensor<T>& t, const SliceGroup& slice_group);

   public:
    ~TensorSliced() override = default;
    TensorSliced(const TensorSliced& other);
    TensorSliced(TensorSliced&& other) noexcept = default;

    DECL_INTERACTIVE_ACTION_TENSOR_UNIQUE_OVERRIDE(TensorSliced)
    DEF_COPY_FILL_TEMPLATES(TensorSliced, T)

    template <typename T_>
    struct eliterator {
        T_* data;
        SliceGroup::const_iterator sg_iterator;
        const shape_t shape;
        const size_t size;

        using difference_type = ptrdiff_t;
        using value_type = T_;
        using reference = T_&;
        using pointer = T_*;
        using iterator_category = std::forward_iterator_tag;

        inline reference operator*() {
            const auto& idx = *sg_iterator;
            auto true_idx = ravel_index(idx, shape, size);
            return data[true_idx];
        }

        inline eliterator& operator++() {
            ++sg_iterator;
            return *this;
        }

        inline eliterator operator++(int) {
            eliterator temp(*this);
            ++(*this);
            return temp;
        }

        inline bool operator==(const eliterator& other) {
            return sg_iterator == other.sg_iterator;
        }

        inline bool operator!=(const eliterator& other) {
            return !(*this == other);
        }
    };
    using eiterator = eliterator<T>;
    using ceiterator = eliterator<const T>;
    static inline eiterator elem_begin(TensorSliced& ts);
    static inline eiterator elem_end(TensorSliced& ts);
    static inline ceiterator const_elem_begin(const TensorSliced& ts);
    static inline ceiterator const_elem_end(const TensorSliced& ts);

    /* We mark these as forbidden because we cannot create TensorView out of
     * TensorSlice.*/
    MARK_FORBIDDEN(TensorView<T> unchecked_subscript(long idx) const override)
    MARK_FORBIDDEN(TensorView<T> unchecked_subscript(const index_t& index)
                       const override)
    TensorSliced<T> unchecked_slice(const Slice& slice) const override;
    TensorSliced<T> unchecked_slice_group(
        const SliceGroup& slice_group) const override;
    MARK_FORBIDDEN(TensorView<T> view(const vector<long>& new_shape) override)

    DECL_ALL_REDUCE_OVERRIDES()

    Tensor<T> reshape(const vector<long>& new_shape) const override;

    SliceGroup slice_group;

   protected:
    ostream& print_to_os(ostream& os, bool rec_start) const override;

   public:
    static T& get(TensorSliced<T>& ts, size_t true_idx);
    static T get(const TensorSliced<T>& ts, size_t true_idx);

    MACRO_INTERACTABLE_TENSORTYPES(DECL_INTERACTIVE_ACTIONS_TENSOR_OVERRIDE,
                                   TensorSliced, T)

    Tensor<T> contiguous() const override;

    TensorSliced<T> slice_unsqueeze(int i);
    const TensorSliced<T> const_slice_unsqueeze(int i) const;
    TensorSliced<T> slice_squeeze(int i);
    const TensorSliced<T> const_slice_squeeze(int i) const;
    TensorSliced<T> slice_squeeze(vector<int> dims);
    const TensorSliced<T> const_slice_squeeze(vector<int> dims) const;
};
}
#endif // BLAS_TENSORSLICED_H_