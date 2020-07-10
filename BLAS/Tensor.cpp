//
// Created by LevZ on 7/8/2020.
//

#include "Tensor.h"

#include <utility>
#include <numeric>
#include <algorithm>


#define MAX_ROW_STRING_SIZE 50
#define MAX_EXPANSION_STRING_SIZE 3

using std::cout;
using std::endl;


using namespace blas;
template<typename T>
Tensor<T>::Tensor() : data(nullptr), size(0) {

}

template<typename T>
Tensor<T>::Tensor(T scalar) : data(new T(scalar)), size(1) {
}


inline size_t shape2size(std::vector<size_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{});
}

template<typename T>
Tensor<T>::Tensor(std::vector<T> data, const std::vector<size_t>& shape) : Tensor(data.data(), shape){

}


template<typename T>
Tensor<T>::Tensor(const Tensor &other) :
data(new T[other.size]), shape(other.shape), size(other.size), strides(other.strides){
    for (int i=0; i < size; ++i)
        this->data[i] = other.data[i];
}

template<typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept : Tensor() {
    using std::swap;
    swap(*this, other);
}

template<typename T>
Tensor<T>::~Tensor() {
    if (requires_deletion)
        delete [] data;
}


template<typename T>
Tensor<T> &Tensor<T>::operator=(Tensor&& other)  noexcept {
    swap(*this, other);
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::at(const index_t& index) const {
    auto [idx, elements] = ravel_index_checked(index, shape, size);
    std::vector<size_t> remaining_shape = elements == 1 ?
                                          std::vector<size_t>{1, 1}:
                                          std::vector<size_t>{shape.begin() + index.size(), shape.end()};
    return Tensor(&data[idx], remaining_shape);
}

template<typename T>
TensorView<T> Tensor<T>::at(const index_t& index) {
    auto [idx, elements] = ravel_index_checked(index, shape, size);
    std::vector<size_t> remaining_shape = elements == 1 ?
                                          std::vector<size_t>{1, 1}:
                                          std::vector<size_t>{shape.begin() + index.size(), shape.end()};
    return TensorView(&data[idx], remaining_shape);
}

shape_t shape2strides(const shape_t &shape){
    std::vector<size_t> res;
    res.reserve(shape.size());
    size_t stride = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{});
    for (auto s: shape){
        stride /= s;
        res.push_back(stride);
    }
    return res;
}


template<typename T>
Tensor<T>::Tensor(T *data, const std::vector<size_t>& shape) :
size(shape2size(shape)), data(new T[shape2size(shape)]), shape(shape), strides(shape2strides(shape)) {
    for (int i=0; i < size; ++i)
        this->data[i] = data[i];
}

template<class Tensor1, class Tensor2>
Tensor1& copy_(Tensor1& dst, const Tensor2& src) {
    using sceiterator = typename Tensor2::ceiterator;
    using deiterator = typename Tensor1::eiterator;
    if (dst.shape != src.shape)
        throw shape_mismatch(dst.shape, src.shape, "copy_");
    sceiterator src_it = Tensor2::const_elem_begin(src);
    deiterator dst_it = Tensor1::elem_begin(dst);
    while (dst_it != Tensor1::elem_end(dst))
        *dst_it++ = *src_it++;
    return dst;
}


template<typename T>
typename Tensor<T>::iterator Tensor<T>::begin() {
    if (shape == shape_t{1})
        throw std::out_of_range("Cannot iterate over a scalar tensor.");
    size_t stride = strides[0];
    shape_t remaining_shape{shape.begin()+1, shape.end()};
    return Tensor::iterator(data, stride, 0, remaining_shape);
}

template<typename T>
typename Tensor<T>::iterator Tensor<T>::end() {
    if (shape == shape_t{1})
        throw std::out_of_range("Cannot iterate over a scalar tensor.");
    size_t stride = strides[0];
    size_t pos = shape[0];
    shape_t remaining_shape{shape.begin()+1, shape.end()};
    return Tensor::iterator(data, stride, pos, remaining_shape);
}

template<typename T>
typename Tensor<T>::const_iterator Tensor<T>::begin() const {
    if (shape == shape_t{1})
        throw std::out_of_range("Cannot iterate over a scalar tensor.");
    size_t stride = strides[0];
    shape_t remaining_shape{shape.begin()+1, shape.end()};
    return Tensor::const_iterator(data, stride, 0, remaining_shape);
}

template<typename T>
typename Tensor<T>::const_iterator Tensor<T>::end() const {
    if (shape == shape_t{1})
        throw std::out_of_range("Cannot iterate over a scalar tensor.");
    size_t stride = strides[0];
    size_t pos = shape[0];
    shape_t remaining_shape{shape.begin()+1, shape.end()};
    return Tensor::const_iterator(data, stride, pos, remaining_shape);
}

template<typename T>
Tensor<T> Tensor<T>::reshape(shape_t new_shape) {
    if (shape2size(new_shape) != size)
        throw shape_mismatch(shape, new_shape, "reshape");
    return Tensor(data, new_shape);
}

template<typename T>
TensorView<T> Tensor<T>::view(shape_t new_shape) {
    if (shape2size(new_shape) != size)
        throw shape_mismatch(shape, new_shape, "reshape");
    return TensorView<T>(data, new_shape);
}



std::pair<size_t, size_t>
ravel_index_checked(const std::vector<int>& idx, const std::vector<size_t>& shape, int size) {
    using std::to_string;
    if (idx.size() > shape.size())
        throw std::out_of_range("Index cannot be of larger dimension of the shape.");
    size_t index=0;
    size_t stride = size > 0 ? size :
            std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{});
    int curr_index;
    for(int i=0; i < idx.size(); ++i){
        stride /= shape[i];
        curr_index = idx[i];
        if (curr_index < -shape[i] || curr_index >= shape[i])
            throw std::out_of_range("Index out of range in dimension "
                + to_string(i) + ": idx=" + to_string(curr_index) + ", shape=" + to_string(shape[i]) + ".");
        curr_index = (curr_index + shape[i]) % shape[i];
        index += stride * curr_index;
    }
    return {index, stride};
}


template<typename T>
TensorView<T>::TensorView(T *data, const std::vector<size_t>& shape) : Tensor<T>() {
    Tensor<T>::data = data;
    Tensor<T>::shape = shape;
    Tensor<T>::strides = shape2strides(shape);
    Tensor<T>::size = shape2size(shape);
    Tensor<T>::requires_deletion = false;
}


template<typename T>
TensorView<T>::TensorView(Tensor<T> t) : TensorView<T>(t.get_data_ptr(), t.shape) {

}


template<typename T>
template<typename Tnsr>
Tensor<T>::subtensor_iterator<Tnsr>::subtensor_iterator(T *data_ptr, size_t stride, size_t pos, shape_t shape) :
baseline_data_ptr(data_ptr), stride(stride), pos(pos), shape(std::move(shape)) {

}


size_t blas::ravel_index(const index_t &idx, const shape_t &shape, int size) {
    size_t index=0;
    size_t stride = size > 0 ? size :
                    std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{});
    int curr_index;
    for(int i=0; i < idx.size(); ++i){
        stride /= shape[i];
        curr_index = idx[i];
        index += stride * curr_index;
    }
    return index;
}

std::vector<size_t> unravel_index(size_t true_idx, const shape_t &shape, int size) {
    int stride = size > 0 ?
            size :
            std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{});
    std::vector<size_t> res(shape.size());
    for (int i = 0; i < res.size(); ++i){
        stride /= shape[i];
        res[i] = true_idx / stride;
        true_idx %= stride;
    }
    return res;
}

std::string shape2str(const shape_t &shape) {
    std::string ret = "(";
    for (int i=0; i< shape.size(); ++i){
        ret += std::to_string(shape[i]);
        if (i < shape.size()-1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}

template<class Tens, typename T>
Tens &fill_(Tens &dst, T scalar) {
    for (auto it = Tens::elem_begin(dst); it != Tens::elem_end(dst); ++it)
        *it = scalar;
    return dst;
}


inline shape_t bradcast_shapes(const shape_t &s1, const shape_t &s2) {
    if (s1.empty() || s2.empty())
        return s1.empty() ? s2 : s1;
    shape_t result(std::max(s1.size(), s2.size()));
    for (int i=0; i < result.size(); ++i) {
        size_t e1 = i >= s1.size() ? 1 : s1[s1.size() - 1 - i];
        size_t e2 = i >= s2.size() ? 1 : s2[s2.size() - 1 - i];
        if (e1 != e2 && e1 != 1 && e2 != 1)
            throw broadcast_failure(s1, s2);
        result[result.size()-1-i] = std::max(e1, e2);
    }
    return result;
}

/**
 * Pulls an applicable index for unbroadcasted tensor using broadcasted destination
 * @param brdcst_dst_idx : the broadcasted destination index
 * @param brdcst_dst_shp : the broadcasted destination shape
 * @param unbrdcst_src : the unbroadcasted source shape
 * @return
 */
shape_t pull_unbroadcasted_index(const shape_t& brdcst_dst_idx ,const shape_t& brdcst_dst_shp,
                                 const shape_t& unbrdcst_src) {
    shape_t res(unbrdcst_src.size());
    for (int i=0; i < res.size(); ++i){
        res[res.size() - 1 - i] = unbrdcst_src[res.size() - 1 - i] == 1 ?
                0 : brdcst_dst_idx[brdcst_dst_idx.size() - 1- i];
    }
    return res;
}


template<typename T>
ostream &Tensor<T>::print_to_os(ostream &os, bool rec_start) const {
    using std::to_string;
    using std::endl;
    os << (rec_start ? "Tensor" : "");
    os << "[";
    if (dim() <= 1) {
        size_t expected_string_size = sizeof(", ") * (size-1) + to_string(data[0]).size() * size;
        if (expected_string_size < MAX_ROW_STRING_SIZE)
            for (int i=0; i < size; ++i) {
                os << data[i];
                if (i < size - 1)
                    os << ", ";
            }
        else
            os << data[0] << ",..., " << data[size - 1];

    }
    else {
        if (shape[0] < MAX_EXPANSION_STRING_SIZE) {
            int i = 0;
            for (const auto &sub: *this) {
                sub.print_to_os(os, false);
                if (++i < shape[0]) os << endl << "      ";
            }
        }
        else {
            this->operator[](0).print_to_os(os, false);
            os << endl << "..." << endl;
            this->operator[](-1).print_to_os(os, false);
        }
    }
    os << "]";
    return os;
}

template<typename T>
Tensor<T> Tensor<T>::operator[](int idx) const {
    idx = normalize_index(idx, shape[0]);
    return *(begin() + idx);
}

template<typename T>
TensorView<T> Tensor<T>::operator[](int idx) {
    idx = normalize_index(idx, shape[0]);
    return begin()[idx];
}

template<typename T>
Tensor<T> Tensor<T>::operator[](const index_t &index) const {
    return at(index);
}

template<typename T>
TensorView<T> Tensor<T>::operator[](const index_t &index) {
    return at(index);
}

template<typename T>
Tensor<T> Tensor<T>::operator()(const Slice &slice) const {
    auto s = normalize_slice(slice, -1);
    return unchecked_slice(s);
}

template<typename T>
TensorSliced<T> Tensor<T>::operator()(const Slice &slice) {
    auto s = normalize_slice(slice, -1);
    return unchecked_slice(s);
}

template<typename T>
Tensor<T> Tensor<T>::operator()(const SliceGroup &slice_group) const {
    auto sg = normalize_slice_group(slice_group);
    return unchecked_slice_group(sg);
}

template<typename T>
TensorSliced<T> Tensor<T>::operator()(const SliceGroup &slice_group) {
    auto sg = normalize_slice_group(slice_group);
    return unchecked_slice_group(sg);
}

template<typename T>
Tensor<T> Tensor<T>::operator-() const {
    return apply( [](T x) -> T { return -x ;} );
}

template<typename T>
Slice Tensor<T>::normalize_slice(const Slice &slice, int max_size) const {
    max_size = max_size <= 0 ? shape[0] : max_size;
    auto s = slice;
    s.b = normalize_index(slice.b, max_size);
    s.e = normalize_index(slice.e, max_size, true);
    return slice;
}

template<typename T>
shape_t Tensor<T>::slice2shape(const Slice &slice) const {
    shape_t result(shape);
    result[0] = slice.size();
    return result;
}

template<typename T>
Tensor<T>::Tensor(const shape_t &shape) : shape(shape), size(shape2size(shape)), data(new T[size]) {

}

template<typename T>
SliceGroup Tensor<T>::normalize_slice_group(const SliceGroup &group) const {
    auto g = group;
    for (int i=0; i < group.slices.size(); ++i)
        g.slices[i] = normalize_slice(group.slices[i], shape[i]);
    return g;
}

template<typename T>
Tensor<T> Tensor<T>::unchecked_subscript(int idx) const {
    return Tensor();
}

template<typename T>
TensorView<T> Tensor<T>::unchecked_subscript(int idx) {
    return TensorView(*this);
}

template<typename T>
Tensor<T> Tensor<T>::unchecked_subscript(const index_t &index) const {
    return Tensor();
}

template<typename T>
TensorView<T> Tensor<T>::unchecked_subscript(const index_t &index) {
    return TensorView(*this);
}

template<typename T>
TensorView<T> Tensor<T>::optimized_unchecked_subscript(int idx) const {
    TensorView<T> x = const_cast<Tensor*>(this)->unchecked_subscript(idx);
    return static_cast<TensorView<T>>(x);
}

template<typename T>
TensorView<T> Tensor<T>::optimized_unchecked_subscript(const index_t &index) const {
    TensorView<T> x = const_cast<Tensor*>(this)->unchecked_subscript(index);
    return static_cast<TensorView<T>>(x);
}

template<typename T>
Tensor<T> Tensor<T>::unchecked_slice(const Slice &slice) const {
    shape_t new_shape(shape);
    new_shape[0] = slice.size();
    Tensor res(new_shape);
    int i = 0;
    for (auto idx: slice){
        Tensor sub = this->optimized_unchecked_subscript(idx);
        res.unchecked_subscript(i++) = sub;
    }
    return res;
}

template<typename T>
TensorSliced<T> Tensor<T>::unchecked_slice(const Slice &slice) {
    return TensorSliced(*this);
}

template<typename T>
Tensor<T> Tensor<T>::unchecked_slice_group(const SliceGroup &slice_group) const {
    return Tensor();
}

template<typename T>
TensorSliced<T> Tensor<T>::unchecked_slice_group(const SliceGroup &slice_group) {
    return TensorSliced(*this);
}

template<typename T>
T *Tensor<T>::get_data_ptr() {
    return data;
}

Slice::Slice(int b, int e, int stride) : b(b), e(e), stride(stride){
    check_stride();
}

constexpr Slice::Slice(std::initializer_list<int> lst) : b(0), e(0), stride(0) {
    using std::tuple;
    switch (lst.size()) {
        case 1: {
            int b_ = *lst.begin();
            std::tie(b, e, stride) = tuple{b_, b_ + 1, 1};
            break;
        }
        case 2: {
            b = *lst.begin();
            e = *(lst.begin() + 1);
            stride = 1;
            break;
        }
        case 3: {
            b = *lst.begin();
            e = *(lst.begin() + 1);
            stride = *(lst.begin() + 2);
        }
        default:
            throw std::invalid_argument("Slice takes 1 to 3 arguments.");
    }
    check_stride();
}

constexpr void Slice::check_stride() const {
    if (e == b)
        return;
    if ( (e - b) * stride <= 0 )// stride cannot reach end from beginning.
        throw std::out_of_range("Iteration cannot reach end with current stride.");
}

SliceGroup::SliceGroup(const vector<tuple<int, int, int>>& slices) {
    for (auto tup: slices)
        this->slices.emplace_back(tup);
}

SliceGroup::SliceGroup(initializer_list<initializer_list<int>> lst) {
    for (auto l: lst)
        this->slices.emplace_back(l);
}

size_t SliceGroup::size() const {
    return std::accumulate(slices.begin(), slices.end(), 1,
            [](size_t x, Slice s) { return x * s.size(); });
}

SliceGroup::const_iterator::const_iterator(index_t pos, vector<Slice> slices,
                                           size_t elems_passed):
       pos(std::move(pos)), slices(std::move(slices)), elems_passed(elems_passed)
{

}

inline bool past_end(int pos, int stride, int end) {
    return stride > 0 ? pos >= end : pos <= end;
}

SliceGroup::const_iterator &SliceGroup::const_iterator::operator++() {
    ++elems_passed;
    for (int i=pos.size() -1; i >= 0; --i){
        auto s = slices[i];
        if (s.size() > 1) {
            pos[i] += s.stride;
            if (past_end(pos[i], s.stride, s.e))
                pos[i] = s.b;
            else
                break;
        }
    }
    return *this;
}

SliceGroup::const_iterator &SliceGroup::const_iterator::operator+=(difference_type x) {

    return *this;
}


template<typename T>
typename TensorSliced<T>::ceiterator TensorSliced<T>::elem_begin(TensorSliced<T>& ts) {
    return typename TensorSliced<T>::eiterator{ ts.data, ts.slice_group.begin(), ts.shape, ts.size };
}

template<typename T>
typename TensorSliced<T>::ceiterator TensorSliced<T>::elem_end(TensorSliced<T> &ts) {
    return typename TensorSliced<T>::eiterator{ ts.data, ts.slice_group.end(), ts.shape, ts.size };
}


template<typename T>
typename TensorSliced<T>::ceiterator TensorSliced<T>::const_elem_begin(const TensorSliced<T>& ts) {
    return typename TensorSliced<T>::ceiterator{ ts.data, ts.slice_group.begin(), ts.shape, ts.size };
}

template<typename T>
typename TensorSliced<T>::ceiterator TensorSliced<T>::const_elem_end(const TensorSliced<T> &ts) {
    return typename TensorSliced<T>::ceiterator{ ts.data, ts.slice_group.end(), ts.shape, ts.size };
}

template<typename T>
TensorSliced<T>::TensorSliced(T *data, const shape_t &shape, const SliceGroup &slice_group) :
    Tensor<T>(), slice_group(slice_group)
{
    Tensor<T>::data = data;
    Tensor<T>::shape = shape;
    Tensor<T>::strides = shape2strides(shape);
    Tensor<T>::size = shape2size(shape);
    Tensor<T>::requires_deletion = false;
}

template<typename T>
TensorSliced<T>::TensorSliced(Tensor<T> t, const SliceGroup &slice_group) {

}

template<typename T>
T &TensorSliced<T>::get(TensorSliced<T> &ts, size_t true_idx) {
    index_t vec_idx = ts.slice_group.translate_true_idx(true_idx);
    size_t data_relative_index = ravel_index(vec_idx, ts.shape, ts.size);
    return ts.data[data_relative_index];
}

template<typename T>
T TensorSliced<T>::get(const TensorSliced<T> &ts, size_t true_idx) {
    index_t vec_idx = ts.slice_group.translate_true_idx(true_idx);
    size_t data_relative_index = ravel_index(vec_idx, ts.shape, ts.size);
    return ts.data[data_relative_index];
}


#define INSTANTIATE_TEMPLATE_TENSOR(dtype) \
    template class blas::Tensor<dtype>;

INSTANTIATE_TEMPLATE_TENSOR(double)