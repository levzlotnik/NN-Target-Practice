#include "DataMap.h"

namespace data {
const uint8_t* Column::get_ptr_at(size_t idx) const {
    return &data.get_data_ptr()[idx * metadata.rt_type.get_size()];
}

uint8_t* Column::get_ptr_at(size_t idx) {
    return &data.get_data_ptr()[idx * metadata.rt_type.get_size()];
}

template <bool lv>
const Row DataMap::IndexLocator<lv>::operator[](size_t row_idx) const {
    if (row_idx >= owner_ref.size) throw std::out_of_range("Out of range.");
    vector<Metadata> m(owner_ref.table_idxs.size());
    for (const auto& [name, idx] : owner_ref.table_idxs)
        m[idx] = owner_ref.columns[idx].get_meta();
    Row row{owner_ref.table_idxs, m};
    for (const auto& [name, idx] : row.table)
        row.data[idx] = owner_ref.columns[idx].get_ptr_at(row_idx);
}

template <bool lv>
Row DataMap::IndexLocator<lv>::operator[](size_t row_idx) {
    static_assert(!lv, "Cannot allow lvalue for const DataMap&. ");
    if (row_idx >= owner_ref.size) throw std::out_of_range("Out of range.");
    vector<Metadata> m(owner_ref.table_idxs.size());
    for (const auto& [name, idx] : owner_ref.table_idxs)
        m[idx] = owner_ref.columns[idx].get_meta();
    Row row{owner_ref.table_idxs, m};
    for (const auto& [name, idx] : row.table)
        row.data[idx] = owner_ref.columns[idx].get_ptr_at(row_idx);
}

template <typename T, template <typename> class Tnsr>
Column::Column(const Metadata& meta, const Tnsr<T>& val)
    : metadata(meta), data({val.size() * meta.rt_type.get_size()}) {
    TensorView<T> buffer = data.view_as<T>();
    buffer.copy_(val);
}

template <typename T, template <typename> class Tnsr>
void DataMap::emplace(const string& name, const Tnsr<T>& val) {
    Metadata meta{RunTimeType::from_dtype<T>()};
    columns.emplace_back(meta, val);
    table_idxs.emplace(name, columns.size() - 1);
}
}  // namespace data