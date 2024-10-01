#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int arraySize = 5;
    const int arrayBytes = arraySize * sizeof(int);

    // Host arrays
    int h_a[arraySize] = {1, 2, 3, 4, 5};
    int h_b[arraySize] = {10, 20, 30, 40, 50};
    int h_c[arraySize];

    // Device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, arrayBytes);
    cudaMalloc((void**)&d_b, arrayBytes);
    cudaMalloc((void**)&d_c, arrayBytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(d_a, d_b, d_c, arraySize);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}