#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Matrix dimensions must be multiples of 16 for WMMA
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)


template <bool ReLU>
__global__ void linear_layer_forward(const half* a, const half* b, const float* bias, float* c, 
                                   int M, int K, int N) {
    // Each warp computes a 16x16 output tile
    // Calculate the warp's position
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = blockIdx.y;  // Each block.y handles one 16xN tile

    // Initialize fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator with bias values
    wmma::load_matrix_sync(c_frag, bias + warpN * WMMA_N, N, wmma::mem_row_major);

    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load the inputs
        wmma::load_matrix_sync(a_frag, a + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(b_frag, b + i * N + warpN * WMMA_N, N);
        
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (ReLU) {
        // ReLU
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = max(0.0f, c_frag.x[i]);
        }
    }

    // Store the output
    wmma::store_matrix_sync(c + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
} 
