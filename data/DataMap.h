#ifndef DATA_DATAMAP_H_
#define DATA_DATAMAP_H_

#include "RuntimeDataType.h"
#include "blas/Tensor.h"

namespace data {

using blas::Tensor;
using blas::TensorView;

struct ColumnMetadata {
    RunTimeType rt_type;
};

class Column {
   private:
    ColumnMetadata metadata;
    Tensor<uint8_t> data;

   public:
    template<typename T>
    inline const TensorView<T> get() const {
        return data.view_as<T>();
    }

    template<typename T>
    inline TensorView<T> get() {
        return data.view_as<T>();
    }
    
};

class DataMap {
   private:
    unordered_map<string, Column> table;

   public:
    template <typename T>
    inline const TensorView<T> get(const string& col_name) const {
        const Column& col = table.at(col_name);
        return col.get<T>();
    }
    template <typename T>
    inline TensorView<T> get(const string& col_name) {
        Column& col = table.at(col_name);
        return col.get<T>();
    }

};
}  // namespace data

#endif  // DATA_DATAMAP_H_