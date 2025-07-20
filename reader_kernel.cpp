#include "dataflow_api.h"

void kernel_main() {
    // compile time args
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t rank = get_compile_time_arg_val(1);

    // runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile  = get_arg_val<uint32_t>(2);

    uint32_t tiled_shape[rank];
    for (uint32_t i = 3; i < rank + 3; ++i) {
    }

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const DataFormat data_format = get_dataformat(cb_id);
    const uint32_t tile_size = get_tile_size(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = data_format
    };


    for (uint32_t tile_id = start_tile; tile_id < end_tile; ++tile_id) {
        // 1. Compute multi-dimensional index for the source tile
        uint32_t src_multi_dim_idx[rank];
        size_t remaining = tile_id;
        for (uint32_t i = 0; i < rank; ++i) {
            src_multi_dim_idx = remaining % src_multi_dim_idx[i];
        }

        // 2. Calculate destination multi-dimensional index

        // 3. Convert destination multi-dimensional index to linear index
        uint32_t dst_linear_idx = 0;
        for (uint32_t i = 0; i < N; ++i) {
            dst_linear_idx += src_multi_dim_idx[i] * src_strides[i];
        }

        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read_tile(tile_id, s0, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
