#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <random>

using namespace nvcuda;

// Matrix dimensions must be multiples of 16 for WMMA
// We'll pad our 25x784 and 784x16 to 32x800 and 800x16
constexpr int M = 32;  // Padded from 25
constexpr int K = 800; // Padded from 784
constexpr int N = 16;  // Already aligned

// WMMA operates on 16x16x16 tiles
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

__global__ void wmma_matrix_multiply(const half* a, const half* b, float* c) {
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

int main() {
    // Check if device supports WMMA
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "Device needs to support CUDA compute capability 7.0 or higher\n";
        return 1;
    }

    // Allocate host memory
    std::vector<half> h_a(M * K);
    std::vector<half> h_b(K * N);
    std::vector<float> h_c(M * N);

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = __float2half(dis(gen));
    }
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = __float2half(dis(gen));
    }

    // Allocate device memory
    half *d_a, *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 1);  // One warp per block
    wmma_matrix_multiply<<<gridDim, blockDim>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first few elements of result
    std::cout << "First few elements of result:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
