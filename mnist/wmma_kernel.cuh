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

__global__ void wmma_matrix_multiply(const half* a, const half* b, float* c, 
                                   int M, int K, int N) {
    // Each warp computes a 16x16 output tile
    // Calculate the warp's position
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = blockIdx.y;  // Each block.y handles one 16xN tile

    // Initialize accumulator with zeros
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load the inputs
        wmma::load_matrix_sync(a_frag, a + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(b_frag, b + i * N + warpN * WMMA_N, N);
        
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output
    wmma::store_matrix_sync(c + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
} 