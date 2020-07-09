//
// Created by LevZ on 7/8/2020.
//

#ifndef TARGETPRACTICE_TENSOR_H
#define TARGETPRACTICE_TENSOR_H

#include <type_traits>
#include <vector>
#include <stdexcept>
#include "common_blas.h"

using std::vector;
using std::tuple;
using std::initializer_list;
using std::ostream;

template <typename T>
class TensorView;

template<typename T>
class Tensor;

using shape_t = std::vector<size_t>;

std::string shape2str(const shape_t& shape){
    std::string ret = "(";
    for (int i=0; i< shape.size(); ++i){
        ret += std::to_string(shape[i]);
        if (i < shape.size()-1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}

class shape_mismatch : public std::out_of_range {
public:
    shape_mismatch(const shape_t& s1, const shape_t& s2, const std::string& action="") :
            std::out_of_range(
                    "Shape Mismatch: " + shape2str(s1) + ", " + shape2str(s2) +
                    (action.empty() ? "" : " for action {" + action + "}") + "."
            ) {}
};

class broadcast_failure : public std::out_of_range {
public:
    broadcast_failure(const shape_t& s1, const shape_t& s2, const std::string& action="") :
            std::out_of_range(
                    "Cannot Broadcast shapes: " + shape2str(s1) + ", " + shape2str(s2) +
                    (action.empty() ? "" : " for action {" + action + "}") + "."
            ) {}
};

class Slice {
private:
    class const_iterator {
        using index_t = int;
        index_t pos;
        int stride;
    public:
        using difference_type=long; // unused
        using value_type=index_t;
        using pointer=const index_t*; // unused
        using reference=const index_t&;
        using iterator_category=std::input_iterator_tag;

        const_iterator(index_t pos, int stride) : pos(pos), stride(stride) {}
        inline const_iterator& operator++() { pos += stride; return *this;}
        inline bool operator ==(const const_iterator& other) const { return pos == other.pos; }
        inline bool operator !=(const const_iterator& other) const { return !(*this == other) ; }
        inline reference operator *() { return pos; }
    };
public:
    int b, e, stride;
    Slice(int b, int e, int stride=1);
    explicit Slice(const tuple<int, int, int>& tup) :
            b(std::get<0>(tup)), e(std::get<1>(tup)), stride(std::get<2>(tup)) {}
    constexpr Slice(initializer_list<int> lst);
    constexpr void check_stride() const;

    inline constexpr size_t size() const { return (e - b - 1) / stride; }

    inline const_iterator begin() const { return const_iterator(b, stride); }
    inline const_iterator end() const { return const_iterator(e, stride); }
};
/**
 * An object that represents an index iterator over a Slice of a tensor.
 */
class SliceGroup {
private:
    class const_iterator {
        using index_t = vector<int>;
        vector<Slice> slices;
        index_t pos;
        size_t elems_passed;
    public:
        using difference_type=long; // unused
        using value_type=index_t;
        using pointer=const index_t*; // unused
        using reference=const index_t&;
        using iterator_category=std::input_iterator_tag;

        const_iterator(index_t pos, vector<Slice> slices, size_t elems_passed = 0);
        const_iterator& operator++();
        inline bool operator ==(const const_iterator& other) const { return elems_passed == other.elems_passed; }
        inline bool operator !=(const const_iterator& other) const { return !(*this == other) ; }
        inline reference operator *() { return pos; }
    };
public:
    explicit SliceGroup(const vector<tuple<int, int, int>>& slices);
    SliceGroup(initializer_list<initializer_list<int>> lst);

    inline const_iterator begin() const { return const_iterator(get_init_pos(), slices, 0); }
    inline const_iterator end() const { return const_iterator(get_init_pos(), slices, size()); }
    inline size_t size() const;
    vector<Slice> slices;

private:
    inline vector<int> get_init_pos() const { vector<int> res; for (auto s: slices) res.push_back(s.b); return res; }
};

template<typename T>
class Tensor {
public:
    Tensor();
    explicit Tensor(T scalar);
    explicit Tensor(const shape_t& shape);
    Tensor(std::vector<T> data, const shape_t& shape);
    Tensor(T* data, const shape_t& shape);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept ;
    virtual ~Tensor();
    friend void swap(Tensor& t1, Tensor& t2);
    Tensor& operator=(Tensor other);

    template<typename InputIt>
    Tensor& copy_(InputIt b, InputIt e, typename InputIt::iterator_category *p=0);
    Tensor& copy_(const Tensor& other);
    Tensor& fill_(T scalar);

    template <typename Tnsr> class subtensor_iterator {
    private:
        T* baseline_data_ptr;
        size_t stride;
        shape_t shape;
        size_t pos;
    public:
        using difference_type=long;
        using value_type=Tnsr;
        using pointer=Tnsr*;
        using reference=Tnsr&;
        using iterator_category=std::random_access_iterator_tag;

        subtensor_iterator(T* data_ptr, size_t stride, size_t pos, shape_t shape);

        inline subtensor_iterator& operator+=(difference_type n) { pos += n; return *this; }
        friend inline subtensor_iterator operator+(difference_type n, subtensor_iterator i)
            { subtensor_iterator temp{i}; return temp+=n; }
        friend inline subtensor_iterator operator+(subtensor_iterator i, difference_type n)
            { subtensor_iterator temp{i}; return temp+=n; }
        inline subtensor_iterator& operator-=(difference_type n) { pos -= n; return *this; }
        friend inline subtensor_iterator operator-(subtensor_iterator i, difference_type n)
            { subtensor_iterator temp{i}; return temp-=n; }
        inline difference_type operator-(subtensor_iterator other) {
            if (baseline_data_ptr != other.baseline_data_ptr)
                throw std::out_of_range("Can't operate on different baseline pointers.");
            return pos - other.pos;
        }
        inline reference operator*() {
            return Tnsr(&baseline_data_ptr[stride * pos], shape);
        }
        inline reference operator[](difference_type n) { return *(*this + n); }
        inline bool operator <(subtensor_iterator other) { return (*this - other) < 0; }
        inline bool operator ==(subtensor_iterator other) { return (*this - other) == 0; }
        inline bool operator !=(subtensor_iterator other) { return (*this - other) != 0; }
        inline bool operator >(subtensor_iterator other) { return other < *this; }
        inline bool operator <=(subtensor_iterator other) { return !(*this > other); }
        inline bool operator >=(subtensor_iterator other) { return !(*this < other); }
        inline subtensor_iterator& operator++() {return (*this += 1); }
        inline subtensor_iterator operator++(int) { subtensor_iterator temp{*this}; ++(*this); return temp; }
        inline subtensor_iterator& operator--() {return (*this -= 1); }
        inline subtensor_iterator operator--(int) { subtensor_iterator temp{*this}; --(*this); return temp; }
    };
    using iterator = subtensor_iterator<TensorView<T>>;
    using const_iterator = subtensor_iterator<Tensor<T>>;

    // Checked indexing
    Tensor at(const std::vector<int>& index) const;
    template<typename ... Args>
    inline Tensor at(Args... args) const {
        return at({args...});
    }

    TensorView<T>& at(std::vector<int> index);
    template<typename ... Args>
    inline TensorView<T>& at(Args... args) {
        return at({args...});
    }
    Tensor operator[](int idx) const;
    TensorView<T>& operator[](int idx);
    Tensor operator[](const vector<int>& index) const;
    TensorView<T>& operator[](const vector<int>& index);
    Tensor operator()(const Slice& slice) const;
    TensorView<T>& operator()(const Slice& slice);
    Tensor operator()(const SliceGroup& slice) const;
    TensorView<T>& operator()(const SliceGroup& slice);
private:
    Tensor unchecked_subscript(int idx) const;
    TensorView<T>& unchecked_subscript(int idx);
    Tensor unchecked_subscript(const vector<int>& index) const;
    TensorView<T>& unchecked_subscript(const vector<int>& index);
    Tensor unchecked_slice(const Slice& slice) const;
    TensorView<T>& unchecked_slice(const Slice& slice);
    Tensor unchecked_slice_group(const SliceGroup& slice_group) const;
    TensorView<T>& unchecked_slice_group(const SliceGroup& slice_group);
public:


    // Iterate over subtensors.
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    // Iterate over elements.
    inline T* elem_begin() { return data; }
    inline T* elem_end() { return data + size; }
    inline const T* elem_begin() const { return data; }
    inline const T* elem_end() const { return data + size; }

    Tensor reshape(shape_t new_shape);
    TensorView<T> view(shape_t new_shape);
    inline size_t dim() const { return shape.size(); }

    /*Elementwise Operations*/
    Tensor& apply_(UnaryOperation op);
    Tensor& apply_(const Tensor& other, BinaryOperation op);
    Tensor& apply_broadcasted_(const Tensor& other, BinaryOperation op);
    Tensor& apply_(float scalar, BinaryOperation op);

    /* Out of place operations */
    Tensor apply(UnaryOperation op) const;
    Tensor apply(const Tensor& other, BinaryOperation op) const;
    Tensor apply_broadcasted(const Tensor& other, BinaryOperation op);
    Tensor apply(float scalar, BinaryOperation op) const;

    Tensor operator-() const;

#define DECL_TENSOR_OPERATOR(op) \
    Tensor operator op(const Tensor& other) const; \
    Tensor operator op(float scalar) const; \
    friend Tensor operator op(float scalar, const Tensor& tensor);

#define DECL_TENSOR_OPERATOR_INPLACE(op) \
    Tensor& operator op(const Tensor& other); \
    Tensor& operator op(float scalar);
    
    // Declare all basic element wise operations!
    MACRO_BASIC_ARITHMETIC_OPERATORS(DECL_TENSOR_OPERATOR)
    MACRO_BASIC_ARITHMETIC_INPLACE_OPERATORS(DECL_TENSOR_OPERATOR_INPLACE)
    friend ostream& operator <<(ostream& os, const Tensor& t);

protected:
    static_assert(std::is_arithmetic_v<T>, "Must be an arithmetic type.");
    T* data;
    shape_t shape;
    shape_t strides;
    size_t size;

    ostream& print_to_os(ostream& os, bool rec_start);

    shape_t slice2shape(const Slice &slice) const;

    Slice normalize_slice(const Slice &slice, int max_size=-1) const;
    SliceGroup normalize_slice_group(const SliceGroup &group) const;
};

/**
 * A view of a tensor - doesn't copy data around.
 * @tparam T : stored data type.
 */
template<typename T>
class TensorView : public Tensor<T> {
    friend class Tensor<T>;
    TensorView(T* data, const std::vector<size_t>&);
    explicit TensorView(Tensor<T> t);

public:
    ~TensorView() = default; // Doesn't delete the data.
    TensorView(const TensorView& tv) = default;
    TensorView& operator=(const Tensor<T>& t); // COPIES DATA.
    TensorView& operator=(Tensor<T>&&) = delete;
    TensorView& operator=(TensorView&& t) noexcept = delete;
    TensorView& operator=(T scalar) { this->fill_(scalar); return *this; }
};


shape_t shape2strides(const shape_t &shape);

/**
 *
 * Translates a dims-format index to a tuple of true index and the number of elements associated with it.
 * Checks for out-of-range errors.
 * @param idx: Unraveled index
 * @param shape: Shape of tensor
 * @param size : the total size of the tensor. Optional, if (-1) the size will be determined from the shape.
 * @return pair of (true_index, num_elements).
 */
std::pair<size_t, size_t> ravel_index_checked(const std::vector<int>& idx, const shape_t& shape, int size= -1);

/**
 * Same as last function but doesn't check for errors and only returns the true index.
 * @param idx : see ravel_index_checked
 * @param shape : see ravel_index_checked
 * @param size : see ravel_index_checked
 * @return
 */
size_t ravel_index(std::vector<size_t> idx, const shape_t& shape, int size=-1);

/**
 * The invers operation of ravel index - translates a true index into dims-format index.
 * @param true_idx : true array index.
 * @param shape : see ravel_index_checked
 * @param size : see ravel_index_checked
 * @return unraveled index for the current tensor
 */
std::vector<size_t> unravel_index(size_t true_idx, const shape_t& shape, int size=-1);

#endif //TARGETPRACTICE_TENSOR_H
