#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fmt/core.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
// #include "ttnn/operations/matmul/matmul.hpp"
// #include "ttnn/operations/reduction/argmax/argmax.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// uint32_t calc_src_tile_index(
//     std::vector<uint32_t>& dims_to_flip
//     uint32_t src_tile_id,
//     uint32_t rank
//     ttnn::Shape& input_tile_shape,
//     ttnn::SmallVector<uint32_t>& input_tile_strides
// ) {
//     size_t remaining = src_tile_id;
//     std::vector<uint32_t> src_multi_dim(rank, 0);
//     std::vector<uint32_t> dst_multi_dim(rank, 0);

//     for (uint32_t i = 0; i < rank; ++i) {
//         size_t dim = rank - i;
//         src_multi_dim[dim] = remaining % input_tile_shape[i];

//         bool should_flip = std::find(
//             dims_to_flip.begin(), dims_to_flip.end(), dim) != dims_to_flip.end();
//         if (should_flip) {
//             // calculate dst tile multi dimension coordinate
//             dst_multi_dim[dim] = input_tile_shape[dim] - src_multi_dim[dim] - 1;
//         } else {
//             dst_multi_dim[dim] = src_multi_dim[dim];
//         }

//         remaining /= input_tile_shape[i];
//     }

//     // dst tile multi dimension coordinate -> dst_linear_tile_id
//     uint32_t dst_tile_id = 0;
//     for (uint32_t j = 0; j < rank; ++j) {
//         dst_tile_id += dst_multi_dim[j] * input_tile_strides[j];
//     }
//     return dst_tile_id;
// }


uint32_t tile_volume(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    return tile_shape[0] * tile_shape[1];
}

uint32_t get_num_tiles(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.padded_shape();
    auto tile_vol = tile_volume(input_tensor);
    return shape.volume() / tile_vol;
}

static ttnn::SmallVector<uint32_t> get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttnn::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    return tiled_shape;
}

static ttnn::SmallVector<uint32_t> get_tile_strides(const ttnn::SmallVector<uint32_t>& shape) {
    ttnn::SmallVector<uint32_t> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static std::vector<uint32_t> compute_strides(const std::vector<uint32_t>& shape) {
    size_t n = shape.size();
    std::vector<uint32_t> strides(n);
    strides[n - 1] = 1;
    for (int i = n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Pretty print a tensor stored as a flat vector
template <typename T>
void pprint(const std::vector<T>& tensor, const std::vector<uint32_t>& dims) {
    size_t ndim = dims.size();
    size_t total_elems = tensor.size();

    // Helper: recursively print
    std::function<void(size_t, size_t, std::string)> print_recursive;
    print_recursive = [&](size_t dim, size_t offset, std::string indent) {
        if (dim == ndim - 1) {
            // Innermost dimension: print elements in one line
            std::cout << indent << "[";
            for (uint32_t i = 0; i < dims[dim]; ++i) {
                std::cout << tensor[offset + i];
                if (i != dims[dim] - 1)
                    std::cout << ", ";
            }
            std::cout << "]";
        } else {
            // Outer dimensions: print nested brackets
            std::cout << indent << "[\n";
            size_t step = 1;
            for (size_t j = dim + 1; j < ndim; ++j)
                step *= dims[j];
            for (uint32_t i = 0; i < dims[dim]; ++i) {
                print_recursive(dim + 1, offset + i * step, indent + "  ");
                if (i != dims[dim] - 1)
                    std::cout << ",\n";
                else
                    std::cout << "\n";
            }
            std::cout << indent << "]";
        }
    };

    // Start recursion
    print_recursive(0, 0, "");
    std::cout << std::endl;
}

void tensor_flip_cpu(
    const std::vector<uint32_t>& src,
    std::vector<uint32_t>& dst,
    const std::vector<uint32_t>& tensor_shape,
    const std::vector<uint32_t>& dims_to_flip) 
{
    const size_t numel = src.size();
    dst.resize(numel);
    auto strides = compute_strides(tensor_shape);
    const size_t ndim = tensor_shape.size();

    for (size_t idx = 0; idx < numel; ++idx) {
        size_t linear = idx, dst_linear = 0;
        for (size_t dim = 0; dim < ndim; ++dim) {
            uint32_t coord = linear / strides[dim];
            linear %= strides[dim];
            // flip coordinate if needed
            if (std::find(dims_to_flip.begin(), dims_to_flip.end(), dim) != dims_to_flip.end()) {
                coord = tensor_shape[dim] - 1 - coord;
            }
            dst_linear += coord * strides[dim];
        }
        dst[dst_linear] = src[idx];
    }
}

int main(int argc, char** argv) {
    constexpr uint32_t N = 1;

    constexpr uint32_t C = 2;
    constexpr uint32_t H = 128;
    constexpr uint32_t W = 224;

    // constexpr uint32_t C = 1;
    // constexpr uint32_t H = 32;
    // constexpr uint32_t W = 32;

    constexpr uint32_t NUMEL = N * C * H * W;
    constexpr uint32_t ELEMENT_SIZE = sizeof(uint32_t);
    constexpr uint32_t TILE_SIZE = ELEMENT_SIZE * TILE_HW;

    const std::vector<uint32_t> input_shape = {N, C, H, W};
    const std::vector<uint32_t> dims_to_flip = {2, 3};

    std::vector<uint32_t> src_vec(NUMEL, 0);
    std::vector<uint32_t> result_tt(NUMEL, 0);
    std::vector<uint32_t> result_cpu(NUMEL, 0);

    std::mt19937 gen(69);
    std::uniform_int_distribution<int> dist(0, 9);
    for (auto& v : src_vec) v = dist(gen);

    tensor_flip_cpu(src_vec, result_cpu, input_shape, dims_to_flip);

    // fmt::print("src_vec:\n");
    // pprint(src_vec, {1, 1, 32, 32});

    src_vec = tilize_nfaces(src_vec, N * C * H, W);
    // fmt::print("src_vec tilized:\n");
    // pprint(src_vec, {1, 1, 32, 32});

    // ------------------------------------------------------------------------
    // TT part
    // ------------------------------------------------------------------------
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    Program program = CreateProgram();
    CommandQueue& cq = device->command_queue();

    ttnn::PageConfig page_config(ttnn::Layout::TILE);
    ttnn::MemoryConfig memory_config(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
    ttnn::TensorLayout layout_config(ttnn::DataType::UINT32, page_config, memory_config);
    ttnn::TensorSpec tensor_spec(ttnn::Shape(input_shape), layout_config);
    ttnn::Tensor input_tensor = ttnn::Tensor::from_vector(src_vec, tensor_spec);
    input_tensor = input_tensor.to_device(device);
    ttnn::Tensor output_tensor = ttnn::Tensor::from_vector(result_tt, tensor_spec);
    output_tensor = output_tensor.to_device(device);

    // fmt::print("input_tensor\n");
    // input_tensor.print();
    // pprint(std::vector<uint32_t>(input_tensor.to_vector<uint32_t>()), input_shape);

    // ttnn::Tensor sliced_input_tensor = ttnn::slice(
    //     ttnn::DefaultQueueId,
    //     input_tensor,
    //     ttnn::SmallVector<uint32_t>{0, 0, 0, 32}, // Start
    //     ttnn::SmallVector<uint32_t>{1, 1, 32, 64}, // End
    //     ttnn::SmallVector<uint32_t>{1, 1, 1, 1}); // Step
    // Synchronize(device);

    // fmt::print("sliced_input_tensor\n");
    // sliced_input_tensor.print();
    // pprint(std::vector<uint32_t>(sliced_input_tensor.to_vector<uint32_t>()), {1, 1, 32, 32});

    uint32_t rank = input_tensor.logical_shape().rank();
    uint32_t num_tiles = get_num_tiles(input_tensor);
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();
    ttnn::SmallVector<uint32_t> input_tile_shape = get_tiled_shape(input_tensor);
    ttnn::SmallVector<uint32_t> input_tile_strides = get_tile_strides(input_tile_shape);

    fmt::print("input_shape: {}\n", input_shape);
    fmt::print("input_tile_shape: {}\n", input_tile_shape);
    fmt::print("input_tile_strides: {}\n", input_tile_strides);

    // ------------------------------------------------------------------------
    // 4) Split work to all available cores
    // ------------------------------------------------------------------------
    auto core_grid = device->compute_with_storage_grid_size();
    CoreRangeSet custom_core_range = CoreRangeSet(CoreRange({0, 0}, {3, 6})); // 4x7 = 28 cores  

    auto [num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2] = split_work_to_cores(core_grid, num_tiles);

    fmt::print("core_grid: {}\n", core_grid);
    fmt::print("num_cores: {}\n", num_cores);
    fmt::print("all_cores: {}\n", all_cores);
    fmt::print("core_group_1: {}\n", core_group_1);
    fmt::print("core_group_2: {}\n", core_group_2);
    fmt::print("num_tiles_per_core_group_1: {}\n", num_tiles_per_core_group_1);
    fmt::print("num_tiles_per_core_group_2: {}\n", num_tiles_per_core_group_2);

    // ------------------------------------------------------------------------
    // 3) Configure Circular Buffers
    // ------------------------------------------------------------------------
    const auto cb_data_format = tt::DataFormat::UInt32;
    uint32_t cb_size = 2 * TILE_SIZE; // double buffering

    auto cb_inp = CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(cb_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, TILE_SIZE));

    // ------------------------------------------------------------------------
    // 2) Set compile-time arguments and create kernels
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)input_tensor.buffer()->is_dram(),
        rank,
        sizeof(uint32_t),
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]  
    };

    auto reader_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tensor_flip/kernels/reader_kernel.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args)
    );

    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)output_tensor.buffer()->is_dram()};
    auto writer_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tensor_flip/kernels/writer_kernel.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args)
    );

    auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    // ------------------------------------------------------------------------
    // 5) Set Runtime Arguments for Kernels
    // ------------------------------------------------------------------------
    std::vector<uint32_t> reader_runtime_args = {input_tensor.buffer()->address(), 0, 0};
    std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(), 0, 0};

    reader_runtime_args.insert(
        reader_runtime_args.end(), input_tile_shape.begin(), input_tile_shape.end());
    reader_runtime_args.insert(
        reader_runtime_args.end(), input_tile_strides.begin(), input_tile_strides.end());

    uint32_t start_tile = 0;
    uint32_t end_tile = 0;
    for (const auto& [ranges, tiles_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                end_tile += tiles_per_core;

                reader_runtime_args[1] = start_tile;
                reader_runtime_args[2] = end_tile;
                SetRuntimeArgs(program, reader_id, core, reader_runtime_args);

                writer_runtime_args[1] = start_tile;
                writer_runtime_args[2] = end_tile;
                SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

                start_tile += tiles_per_core;
            }
        }
    }

    // fmt::print("all_close: {}\n", ttnn::allclose<uint32_t>(input_tensor.cpu(), output_tensor.cpu(), 1e-5f, 1e-5f));
    // fmt::print("enqueue program\n");

    EnqueueProgram(cq, program, false);
    Finish(cq);

    // fmt::print("finished execution\n");
    // fmt::print("all_close: {}\n", ttnn::allclose<uint32_t>(input_tensor.cpu(), output_tensor.cpu(), 1e-5f, 1e-5f));

    CloseDevice(device);
    return 0;
}
