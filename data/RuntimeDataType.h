#ifndef DATA_RUNTIMEDATATYPE_H_
#define DATA_RUNTIMEDATATYPE_H_
#include <memory>

#include "common.h"

namespace data {
typedef enum { LONG, FLOAT = 1, SINGLE = 1, DOUBLE = 2 } DataType;

inline string dtype2str(DataType dtype) {
    switch (dtype) {
        case LONG:
            return "long";
        case FLOAT:
            return "float";
        case DOUBLE:
            return "double";
        default:
            throw std::invalid_argument("No such data type.");
    }
}


struct RunTimeTypeCheckBase {
    virtual void check_dtype(DataType dtype) const = 0;
    virtual size_t dtype_size() const = 0;

    template <typename T>
    void check_dtype() const;
};

class RunTimeType {
   private:
    std::shared_ptr<RunTimeTypeCheckBase> type_checker;
    RunTimeType() = default;
    DataType dtype;

   public:
    RunTimeType(DataType dtype);

    template <typename T>
    static RunTimeType from_dtype();

    inline DataType get() const { return dtype; }
    inline size_t get_size() const { return type_checker->dtype_size(); }

    inline string to_str() const { return dtype2str(dtype); }

    inline void check_dtype(DataType dtype) const {
        type_checker->check_dtype(dtype);
    }

    template <typename T>
    inline void check_dtype() const {
        type_checker->check_dtype<T>();
    }
};

}  // namespace data
#endif  // DATA_RUNTIMEDATATYPE_H_