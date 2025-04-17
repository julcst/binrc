#pragma once

#include <cuda_runtime.h>
#include <cudamath.cuh>
#include <cudautil.hpp>

#include <stb_image_write.h>

__global__ void computeSpecularLUT(cudaSurfaceObject_t surfObj, uint width, uint height, uint samples);

struct BRDFLUT {
    cudaArray_t array;
    cudaTextureObject_t texObj;
    cudaSurfaceObject_t surfObj;
    uint width;
    uint height;

    BRDFLUT(uint w = 256, uint h = 256) : width(w), height(h) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>();
        check(cudaMallocArray(&array, &channelDesc, width, height, cudaArraySurfaceLoadStore));

        cudaResourceDesc resDesc = {
            .resType = cudaResourceTypeArray,
            .res = { .array = array },
        };
        cudaTextureDesc texDesc = {
            .addressMode = { cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp },
            .filterMode = cudaFilterModeLinear,
            .readMode = cudaReadModeNormalizedFloat,
            .sRGB = false,
            .borderColor = { 0, 0, 0, 0 },
            .normalizedCoords = true,
            .maxAnisotropy = 0,
            .mipmapFilterMode = cudaFilterModePoint,
            .mipmapLevelBias = 0,
            .minMipmapLevelClamp = 0.0f,
            .maxMipmapLevelClamp = 0.0f,
            .disableTrilinearOptimization = 0,
            .seamlessCubemap = 0,
        };
        check(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

        check(cudaCreateSurfaceObject(&surfObj, &resDesc));
        compute();

        writeToFile("lut.png");
    }

    ~BRDFLUT() {
        check(cudaDestroyTextureObject(texObj));
        check(cudaDestroySurfaceObject(surfObj));
        check(cudaFreeArray(array));
    }

    void compute(uint samples = 1024) {
        // Define the thread block and grid dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        
        // Launch the kernel to compute the directional albedo
        computeSpecularLUT<<<gridSize, blockSize>>>(surfObj, width, height, samples);
        check(cudaDeviceSynchronize());
    }

    void writeToFile(const char* filename) {
        std::vector<unsigned char> data(width * height * 4);
        check(cudaMemcpy2DFromArray(data.data(), width * sizeof(uchar4), array, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToHost));
        stbi_flip_vertically_on_write(true); // Flip the image vertically for correct orientation
        int result = stbi_write_png(filename, width, height, 4, data.data(), width * 4);
    }
};
