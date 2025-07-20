#include "dataflow_api.h"
#include "debug/dprint.h"


inline uint32_t calc_src_tile_index(
    uint32_t src_tile_id,
    uint32_t rank,
    uint32_t* dims_to_flip,
    uint32_t* input_tile_shape,
    uint32_t* input_tile_strides) {

    // 1. Compute multi-dimensional index for the source tile
    // 2. Convert multi-dimensional index to linear index

    size_t remaining = src_tile_id;
    std::vector<uint32_t> src_multi_dim(rank, 0);
    std::vector<uint32_t> dst_multi_dim(rank, 0);

    for (uint32_t i = 0; i < rank; ++i) {
        size_t dim = rank - i;
        src_multi_dim[dim] = remaining % input_tile_shape[i];

        bool should_flip = std::find(
            dims_to_flip.begin(), dims_to_flip.end(), dim) != dims_to_flip.end();
        if (should_flip) {
            // calculate dst tile multi dimension coordinate
            dst_multi_dim[dim] = input_tile_shape[dim] - src_multi_dim[dim] - 1;
        } else {
            dst_multi_dim[dim] = src_multi_dim[dim];
        }

        remaining /= input_tile_shape[i];
    }

    // dst tile multi dimension coordinate -> dst_linear_tile_id
    uint32_t dst_tile_id = 0;
    for (uint32_t j = 0; j < rank; ++j) {
        dst_tile_id += dst_multi_dim[j] * input_tile_strides[j];
    }
    return dst_tile_id;
}  

void kernel_main() {
    // ------------------------------------------------------------------------
    // 1) Compile-time arguments
    // ------------------------------------------------------------------------
    constexpr bool src0_is_dram = static_cast<bool>(get_compile_time_arg_val(0));
    constexpr uint32_t RANK = get_compile_time_arg_val(1);
    constexpr uint32_t element_size = get_compile_time_arg_val(2);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(3);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(4);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(5);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(6);

    // ------------------------------------------------------------------------
    // 2) Runtime arguments
    // ------------------------------------------------------------------------
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile  = get_arg_val<uint32_t>(2);

    uint32_t input_tile_shape[RANK], input_tile_strides[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        input_tile_shape[i] = get_arg_val<uint32_t>(i + 3);
        input_tile_strides[i] = get_arg_val<uint32_t>(i + RANK + 3);
    }

    // ------------------------------------------------------------------------
    // 3) Derived constants
    // ------------------------------------------------------------------------
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const DataFormat data_format = get_dataformat(cb_id);
    const uint32_t tile_size = get_tile_size(cb_id);
    const InterleavedAddrGenFast<src_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    for (uint32_t tile_id = start_tile; tile_id < end_tile; ++tile_id) {
        cb_reserve_back(cb_id, onetile);

        // tile_id = dst_tile_id because we writing to output in order
        // calculate src_tile_id that should be placed 

        // inline uint32_t calc_src_tile_index(
        //     uint32_t src_tile_id,
        //     uint32_t rank,
        //     uint32_t* dims_to_flip,
        //     uint32_t* input_tile_shape,
        //     uint32_t* input_tile_strides) {
        uint32_t src_tile_linear_id = calc_src_tile_index(
            tile_id, RANK, dims_to_flip, input_tile_shape, input_tile_strides);
        
        uint32_t dest_tr0_l1 = get_write_ptr(cb_id);
        uint64_t src_bank_addr = get_noc_addr(src_tile_linear_id, s0);

        // Intra tile flip
        uint64_t subtile_addr = src_bank_addr;
        for (uint32_t sub = 0; sub < 4; sub++) {

            // Read subtile line by line
            for (uint32_t c16 = 0; c16 < FACE_HEIGHT; c16++) {
                noc_async_read(subtile_addr, dest_tr0_l1, SUBTILE_LINE_BYTES);
                if (is_horizontal_flip) {
                    // flip h_order
                }
            }
            DPRINT << uint32_t( reinterpret_cast<uint16_t*>( dest_tr0_l1 )[0] ) << ENDL();

            // Intra subtile flip

            if (sub == 1) {
                // subtile_offset += (HW2 << 4);
            }
            subtile_src_addrs += subtile_offset;
        }


        cb_push_back(cb_id, onetile);
    }
}
