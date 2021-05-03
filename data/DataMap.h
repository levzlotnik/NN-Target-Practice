#ifndef DATA_DATAMAP_H_
#define DATA_DATAMAP_H_

#include "RuntimeDataType.h"
#include "blas/Tensor.h"
#include <type_traits>

namespace data {

using blas::Tensor;
using blas::TensorView;

struct Metadata {
    RunTimeType rt_type;
};

class Column {
   private:
    Metadata metadata;
    Tensor<uint8_t> data;
    friend class DataMap;
    uint8_t* get_ptr_at(size_t idx);
    const uint8_t* get_ptr_at(size_t idx) const;

   public:
    Metadata get_meta() const { return metadata; }
    template <typename T>
    inline const TensorView<T> get() const {
        metadata.rt_type.check_dtype<T>();
        return data.view_as<T>();
    }

    template <typename T>
    inline TensorView<T> get() {
        metadata.rt_type.check_dtype<T>();
        return data.view_as<T>();
    }
};

class Row {
   private:
    vector<uint8_t*> data;
    vector<Metadata> metadata;
    unordered_map<string, size_t> table;
    friend class DataMap;
    Row(unordered_map<string, size_t> t, vector<Metadata> m)
        : metadata(m), table(t), data(m.size()) {}

   public:
    template <typename T>
    inline T& at(const string& name) {
        size_t idx = table.at(name);
        metadata[idx].rt_type.check_dtype<T>();
        return *data[idx];
    }

    template <typename T>
    inline T at(const string& name) const {
        size_t idx = table.at(name);
        metadata[idx].rt_type.check_dtype<T>();
        return *data[idx];
    }
};

class DataMap {
   private:
    vector<Column> columns;
    unordered_map<string, size_t> table_idxs;
    size_t size;

   public:
    template<bool is_lvalue>
    class IndexLocator {
       private:
        using ref_type = typename std::conditional<is_lvalue, DataMap&, const DataMap&>::type;
        ref_type owner_ref;

       public:
        IndexLocator(ref_type dm) : owner_ref(dm) {}
        const Row operator[](size_t row_idx) const;
        Row operator[](size_t row_idx);
    };

    IndexLocator<true> iloc() { return IndexLocator<true>(*this); }
    IndexLocator<false> iloc() const { return IndexLocator<false>(*this); }
};
}  // namespace data

#endif  // DATA_DATAMAP_H_