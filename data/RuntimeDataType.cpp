#include "RuntimeDataType.h"

namespace data {
template <typename T>
struct Compile2RuntimeTypeConverter {
    static_assert(false, "Wrong type.");
    inline static constexpr DataType value = DataType::NOTYPE;
};

template <>
struct Compile2RuntimeTypeConverter<long> {
    inline static constexpr DataType value = DataType::LONG;
};

template <>
struct Compile2RuntimeTypeConverter<float> {
    inline static constexpr DataType value = DataType::FLOAT;
};

template <>
struct Compile2RuntimeTypeConverter<double> {
    inline static constexpr DataType value = DataType::DOUBLE;
};

template <typename T>
void RunTimeTypeCheckBase::check_dtype<T>() const {
    return check_dtype(Compile2RuntimeTypeConverter<T>::value);
}

template <typename T>
class RunTimeTypeCheck : public RunTimeTypeCheckBase {
    void check_dtype(DataType dtype) const override;
    size_t dtype_size() const override { return sizeof(T); }
};

template class RunTimeTypeCheck<long>;
template class RunTimeTypeCheck<float>;
template class RunTimeTypeCheck<double>;

template <typename T>
void RunTimeTypeCheck<T>::check_dtype(DataType dtype) const {
    DataType expected = Compile2RuntimeTypeConverter<T>::value;
    if (expected != dtype)
        throw std::runtime_error("Wrong Data Type. Expected '" +
                                 dtype2str(expected) + "' but got '" +
                                 dtype2str(dtype));
}

RunTimeTypeCheckBase* get_from_dtype(DataType dtype) {
    switch (dtype) {
        case LONG:
            return new RunTimeTypeCheck<long>();
        case FLOAT:
            return new RunTimeTypeCheck<float>();
        case DOUBLE:
            return new RunTimeTypeCheck<double>();
        default:
            throw std::invalid_argument("Wrong dtype.");
    }
}

RunTimeType::RunTimeType(DataType dtype)
    : type_checker(get_from_dtype(dtype)) {}

template <typename T>
RunTimeType RunTimeType::from_dtype<T>() {
    RunTimeType obj;
    obj.type_checker = new RunTimeTypeCheck<T>();
    return obj;
}

}  // namespace data
