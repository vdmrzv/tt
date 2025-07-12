#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fmt/core.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

// Pretty print a tensor stored as a flat vector
void pprint(std::vector<int>& tensor, std::vector<uint32_t> dims) {
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

void flip_vector(std::vector<int>& src, std::vector<int>& dst) {
    for (int i = 0; i < src.size(); ++i) {
        dst[dst.size() - i - 1] = src[i];
    }
}

void flip_cpu(
    std::vector<int>& src,
    std::vector<int>& dst,
    std::vector<uint32_t> dims,
    std::vector<uint32_t> dims_to_permute) {

    uint32_t stride;
    for (const auto& dim_idx : dims_to_permute) {
        int dim_size = dims[dim_idx];
        if (dim_size > 1) {
            stride = std::accumulate(dims.begin() + dim_idx + 1, dims.end(), 1, std::multiplies<int>());
            std::cout << "dim: " << dim_idx << "\t" << "stride: " << stride << std::endl;

            // last dimension
            // if (dim_idx = dims[dims.size()]) {
            //     for (int i = 0; i < ; i += stride) {
            //         for (int j = 0; j < src.size(); ++j) {
            //            dst[dst.size() - i * stride - 1] = src[i];
            //         }
            //     }
            // }

            for (int i = 0; i < dim_size; ++i) {
                // dst[i * stride] = 
            }
        }
    }
}

int main(int argc, char** argv) {
    // cpu part
    constexpr uint32_t N = 1;
    constexpr uint32_t H = 32;
    constexpr uint32_t W = 64;
    constexpr uint32_t C = 3;
    constexpr uint32_t numel = N * H * W * C;

    // Create random input vectors for matrices A and B
    std::mt19937 gen(69);
    std::uniform_int_distribution<int> dist(0, 10);

    std::vector<int> src(numel, 0);
    std::vector<uint32_t> shape = {N, H, W, C};
    for (auto& v : src) v = dist(gen);

    std::vector<int> dst_cpu(numel, 0);
    std::vector<uint32_t> dims_to_permute = {0, 1, 2, 3};

    pprint(src, shape);
    flip_cpu(src, dst_cpu, shape, dims_to_permute);

    // tt part
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    Program program = CreateProgram();
    CommandQueue& cq = device->command_queue();

    // Get the compute grid size to determine how many cores are available
    auto core_grid = device->compute_with_storage_grid_size();
    const uint32_t num_tiles = numel / TILE_HW;
    const uint32_t tile_size = TILE_HW * sizeof(uint32_t);
    const uint32_t dram_buffer_size = num_tiles * tile_size;

    auto [num_cores, // number of cores utilized
        all_cores, // set of all cores used
        core_group_1, // Primary group: handles more tiles per core
        core_group_2, // Secondary group: handles fewer tiles per core
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2
    ] = split_work_to_cores(core_grid, num_tiles);

    fmt::print("core_grid: {}\n", core_grid);
    fmt::print("num_tiles: {}\n", num_tiles);
    fmt::print("all_cores: {}\n", core_grid);
    fmt::print("core_group_1: {}\n", core_group_1);
    fmt::print("core_group_2: {}\n", core_group_2);
    fmt::print("num_tiles_per_core_group_1: {}\n", num_tiles_per_core_group_1);
    fmt::print("num_tiles_per_core_group_2: {}\n", num_tiles_per_core_group_2);

    // Page size is the granularity at which data is distributed across DRAM banks in interleaved mode
    // Setting page_size to tile_size is the most common configuration for memory buffers in Metalium

    // Create DRAM buffers for input and output tensors
    InterleavedBufferConfig dram_config {
        .device = device,
        .size = dram_buffer_size,
        .page_size = tile_size,
        .buffer_type = BufferType::DRAM
    };

    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    fmt::print("src_dram_buffer: {}\n", src_dram_buffer->address());
    fmt::print("dst_dram_buffer: {}\n", dst_dram_buffer->address());
    return 0;

    // Configure Circular Buffers
    // circular buffers exist in SRAM/L1 of each tensix core
    const auto cb_data_format = tt::DataFormat::UInt32;

    // each circular buffer has to have index
    // circular buffer index for input tensor
    uint32_t src_cb_index = CBIndex::c_0;


    // Create kernels


    return 0;
}
