#include "DataType.h"
#include "blas/Tensor.h"
#include "common.h"

namespace data {
class Column {
   private:
    blas::Tensor<uint8_t> _data;
    DType dtype;

   public:

};

class DataFrame {};

}  // namespace data