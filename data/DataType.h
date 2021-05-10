#include <stdexcept>
#include <type_traits>
#include "common.h"

namespace data {

enum DType { NO_TYPE, UINT8, LONG, FLOAT, DOUBLE };

inline string dtype2str(DType dtype) {
    switch (dtype) {
        case UINT8:
            return "uint8_t";
        case LONG:
            return "long";
        case FLOAT:
            return "float";
        case DOUBLE:
            return "double";
        default:
            throw std::invalid_argument("Invalid dtype.");
    }
}

inline size_t dtype2size(DType dtype) {
    switch (dtype) {
        case UINT8:
            return sizeof(uint8_t);
        case LONG:
            return sizeof(long);
        case FLOAT:
            return sizeof(float);
        case DOUBLE:
            return sizeof(double);
        default:
            throw std::invalid_argument("Invalid dtype.");
    }
}

struct DTypeConvert {
    template <typename CompileTimeDtype>
    static constexpr DType from_v = DType::NO_TYPE;

    template <DType someBadDataType>
    struct to {
        using type = std::false_type;
    };

    template <>
    static constexpr DType from_v<uint8_t> = DType::UINT8;
    template <>
    static constexpr DType from_v<long> = DType::LONG;
    template <>
    static constexpr DType from_v<float> = DType::FLOAT;
    template <>
    static constexpr DType from_v<double> = DType::DOUBLE;

   private:
    template <>
    struct to<DType::UINT8> {
        using type = uint8_t;
    };
    template <>
    struct to<DType::LONG> {
        using type = long;
    };
    template <>
    struct to<DType::FLOAT> {
        using type = float;
    };
    template <>
    struct to<DType::DOUBLE> {
        using type = double;
    };

   public:
    template <DType RunTimeDtype>
    using to_t = typename to<RunTimeDtype>::type;
};


}  // namespace data