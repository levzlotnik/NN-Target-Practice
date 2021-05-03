#include "DataMap.h"

namespace data 
{
    const uint8_t* Column::get_ptr_at(size_t idx) const {
        return &data.get_data_ptr()[idx * metadata.rt_type.get_size()];
    }

    uint8_t* Column::get_ptr_at(size_t idx) {
        return &data.get_data_ptr()[idx * metadata.rt_type.get_size()];
    }

    template <bool lv>
    const Row DataMap::IndexLocator<lv>::operator[](size_t row_idx) const {
        if (row_idx >= owner_ref.size)
            throw std::out_of_range("Out of range.");
        vector<Metadata> m(owner_ref.table_idxs.size());
        for (const auto& [name, idx]: owner_ref.table_idxs)
            m[idx] = owner_ref.columns[idx].get_meta();
        Row row{owner_ref.table_idxs, m};
        for (const auto& [name, idx]: row.table)
            row.data[idx] = owner_ref.columns[idx].get_ptr_at(row_idx);
    }

    template<bool lv>
    Row DataMap::IndexLocator<lv>::operator[](size_t row_idx) {
        static_assert(!lv, "Cannot allow lvalue for const DataMap&. ");
        if (row_idx >= owner_ref.size)
            throw std::out_of_range("Out of range.");
        vector<Metadata> m(owner_ref.table_idxs.size());
        for (const auto& [name, idx]: owner_ref.table_idxs)
            m[idx] = owner_ref.columns[idx].get_meta();
        Row row{owner_ref.table_idxs, m};
        for (const auto& [name, idx]: row.table)
            row.data[idx] = owner_ref.columns[idx].get_ptr_at(row_idx);
    }
}