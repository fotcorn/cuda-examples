#include <cuda_runtime.h>
#include <iostream>

__global__ void map(int *in, int *out, const size_t nx, const size_t ny) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t index = idy * nx + idx;

    if (idx < nx && idy < ny) {
        out[index] = in[index] + 10;
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    // Check if there is a GPU available.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Initialize the input array on the host.
    const size_t nx = 50;
    const size_t ny = 30;
    int hostIn[ny][nx];
    int hostOut[ny][nx];

    for (size_t y = 0; y < ny; y++) {
        for (size_t x = 0; x < nx; x++) {
            hostIn[y][x] = x * y;
        }
    }

    // Allocate memory on the device for both input and output arrays, which have the same size.
    int *deviceIn, *deviceOut;
    CUDA_CHECK(cudaMalloc((void **)&deviceIn, nx * ny * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&deviceOut, nx * ny * sizeof(int)));

    // Copy the input array from the host to the device.
    CUDA_CHECK(cudaMemcpy(deviceIn, hostIn, nx * ny * sizeof(int), cudaMemcpyHostToDevice));

    const int threads = 32; // 32 * 32 = 1024 threads, the maximum per block.
    dim3 threadsPerBlock(threads, threads);

    const int numBlocksX = (nx + threads - 1) / threads;
    const int numBlocksY = (ny + threads - 1) / threads;
    dim3 blocks(numBlocksX, numBlocksY);

    std::cout << "blocks: (" << blocks.x << ", " << blocks.y << ", " << blocks.z << "), threadsPerBlock: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << ")" << std::endl;

    map<<<blocks, threadsPerBlock>>>(deviceIn, deviceOut, nx, ny);
    CUDA_CHECK(cudaGetLastError()); // Check for any errors in the kernel launch.
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all threads to complete.

    // Copy the output array from the device to the host.
    CUDA_CHECK(cudaMemcpy(hostOut, deviceOut, nx * ny * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the output array.
    for (size_t y = 0; y < ny; y++) {
        std::cout << "# " << y << std::endl;
        for (size_t x = 0; x < nx; x++) {
            std::cout << hostOut[y][x] << ' ';
        }
        std::cout << std::endl;
    }

    // Release the allocated memory on the device.
    CUDA_CHECK(cudaFree(deviceIn));
    CUDA_CHECK(cudaFree(deviceOut));

    return 0;
}
