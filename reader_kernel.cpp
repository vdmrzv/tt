#include "dataflow_api.h"
#include "debug/dprint.h"

inline uint32_t calc_src_tile_index(
    uint32_t dst_tile_id,
    uint32_t rank,
    uint32_t* dims_to_flip,
    uint32_t* tiled_shape,
    uint32_t* tile_strides) {

    size_t remaining = dst_tile_id;
    uint32_t src_multi_dim[rank];
    uint32_t dst_multi_dim[rank];
    uint32_t num_dims_to_flip = 2; // TODO fix this

    // 1. Convert output tile linear index to multi-dimensional index
    for (uint32_t i = 0; i < rank; ++i) {  
        uint32_t dim = rank - 1 - i;
        dst_multi_dim[dim] = remaining % tiled_shape[dim];
        remaining /= tiled_shape[dim];
    }

    // 2. Based on 1) compute multi-dimensional index for the source tile
    for (uint32_t i = 0; i < rank; ++i) {
        bool should_flip = false;
        for (uint32_t j = 0; j < num_dims_to_flip; j++) {
            if (dims_to_flip[j] == i) {
                should_flip = true;
                break;
            }
        }

        if (should_flip) {
            src_multi_dim[i] = tiled_shape[i] - dst_multi_dim[i] - 1;
        } else {
            src_multi_dim[i] = dst_multi_dim[i];
        }
    }

    // DPRINT << "DST_MULTI_DIM: ";
    // for (uint32_t i = 0; i < rank; ++i)
    //     DPRINT << dst_multi_dim[i] << ", ";
    // DPRINT << ENDL();

    // DPRINT << "SRC_MULTI_DIM: ";
    // for (uint32_t i = 0; i < rank; ++i)
    //     DPRINT << src_multi_dim[i] << ", ";
    // DPRINT << ENDL();

    // 3. Convert source tile multi-dimensional index to linear index
    uint32_t src_tile_id = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        src_tile_id += src_multi_dim[i] * tile_strides[i];
    }
    return src_tile_id;
}

void kernel_main() {
    // ------------------------------------------------------------------------
    // 1) Compile-time arguments
    // ------------------------------------------------------------------------
    constexpr bool src_is_dram = static_cast<bool>(get_compile_time_arg_val(0));
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

    uint32_t tiled_shape[RANK], tile_strides[RANK];
    for (uint32_t i = 0; i < RANK; i++) {
        tiled_shape[i] = get_arg_val<uint32_t>(i + 3);
        tile_strides[i] = get_arg_val<uint32_t>(i + RANK + 3);
    }

    // ------------------------------------------------------------------------
    // 3) Derived constants
    // ------------------------------------------------------------------------
    constexpr uint32_t FACE_HW = FACE_HEIGHT * FACE_WIDTH;
    constexpr uint32_t FACE_HW_BYTES = FACE_HW * element_size;
    constexpr uint32_t NUM_FACES_H = TILE_HEIGHT / FACE_HEIGHT;
    constexpr uint32_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;
    constexpr uint32_t NUM_FACES = NUM_FACES_H * NUM_FACES_W;
    constexpr uint32_t SUBTILE_LINE_BYTES = FACE_WIDTH * element_size;
    constexpr uint32_t FACE_H_STRIDE_BYTES = NUM_FACES_W * FACE_HW_BYTES;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const DataFormat data_format = get_dataformat(cb_id);
    const uint32_t tile_size = get_tile_size(cb_id);
    const InterleavedAddrGenFast<src_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    uint32_t dims_to_flip[2] = {2, 3}; // TODO fix this

    for (uint32_t tile_id = start_tile; tile_id < end_tile; ++tile_id) {
        cb_reserve_back(cb_id, onetile);
        uint32_t l1_buf_addr = get_write_ptr(cb_id);
        uint32_t l1_buf_base_addr = l1_buf_addr; // save base address for debug print
 
        uint32_t src_tile_id = calc_src_tile_index(
            tile_id, RANK, dims_to_flip, tiled_shape, tile_strides);
        // DPRINT << tile_id << " <- " << src_tile_id << ENDL();
 
        uint64_t tile_base_addr = get_noc_addr(src_tile_id, s0, 0);

        // read faces in reverse order
        for (int32_t face = NUM_FACES - 1; face >= 0; face--) {
            uint64_t face_addr = tile_base_addr + face * FACE_HW_BYTES;

            // read rows in reverse order
            for (int32_t face_row = FACE_HEIGHT - 1; face_row >= 0; face_row--) {
                uint64_t face_row_addr = face_addr + face_row * SUBTILE_LINE_BYTES;
                noc_async_read(face_row_addr, l1_buf_addr, SUBTILE_LINE_BYTES);
                noc_async_read_barrier();

                // Flip elements within the row in L1
                uint32_t* row_data = reinterpret_cast<uint32_t*>(l1_buf_addr);
                for (uint32_t i = 0; i < FACE_WIDTH / 2; ++i) {
                    uint32_t temp = row_data[i];
                    row_data[i] = row_data[FACE_WIDTH - 1 - i];
                    row_data[FACE_WIDTH - 1 - i] = temp;
                }
                l1_buf_addr += SUBTILE_LINE_BYTES;
            }
        }

        // DPRINT << "debug print" << ENDL();
        // for (uint32_t face = 0; face < 4; face++) {
        //     for (uint32_t face_row = 0; face_row < FACE_HEIGHT; face_row++) {
        //         for (uint32_t face_col = 0; face_col < FACE_WIDTH; ++face_col) {
        //             DPRINT << uint32_t(reinterpret_cast<uint32_t*>(l1_buf_base_addr)[face_col]) << ", ";
        //         }
        //         l1_buf_base_addr += SUBTILE_LINE_BYTES;
        //         DPRINT << ENDL();
        //     }
        //     DPRINT << ENDL();
        // }
        noc_async_read_barrier();
        cb_push_back(cb_id, onetile);
    }
}
