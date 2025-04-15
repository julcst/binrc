#pragma once

#include <cuda_runtime.h>
#include <cudamath.cuh>
#include <optix/sampling.cuh>
#include <cudautil.hpp>

#include <stb_image_write.h>

__host__ __device__ __forceinline__ float RadicalInverse_VdC(uint bits)  {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

__host__ __device__ __forceinline__ float2 Hammersley(uint i, uint N) {
    return {float(i) / float(N), RadicalInverse_VdC(i)};
}  

/**
 * Integration of the specular BRDF over a completely white hemisphere.
 * For derivation, see:
 * https://learnopengl.com/PBR/IBL/Specular-IBL
 * @return scale and bias to be applied to F0.
 */
__host__ __device__ __forceinline__ float3 integrateSpecular(float alpha, float cosTheta, uint samples) {
    const float alpha2 = alpha * alpha;

    // Create viewing vector
    const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    const float3 wo = make_float3(sinTheta, 0.0f, cosTheta);
    const float3 n = make_float3(0.0f, 0.0f, 1.0f);
    const float NdotV = wo.z;
    
    float scale = 0.0f;
    float bias = 0.0f;
    float diffuse = 0.0f;
    
    // Integrate over hemisphere
    for (uint i = 0; i < samples; ++i) {
        const float2 rand = Hammersley(i, samples);
        
        // Sample the visible normal distribution function (VNDF)
        const float3 wm = sampleVNDFTrowbridgeReitz(rand, wo, n, alpha);
        
        // Calculate reflected direction
        const float3 wi = reflect(wo, wm);
        
        // Calculate the various dot products needed for BRDF evaluation
        const float NdotL = wi.z;
        const float HdotV = dot(wm, wo);

        if (NdotL <= 0.0f) continue; // Below horizon
        
        // Calculate the various BRDF components
        const float Ffactor = pow5(1.0f - HdotV);
        const float LambdaL = Lambda_TrowbridgeReitz(NdotL, alpha2);
        const float LambdaV = Lambda_TrowbridgeReitz(NdotV, alpha2);
        
        // Specular BRDF weighted by cosine term and importance sampling weight
        const float brdf = (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV);
        
        scale += (1.0f - Ffactor) * brdf;
        bias += Ffactor * brdf;
        diffuse += sampleBrentBurley(rand, wo, NdotV, n, alpha, {}, make_float3(1.0f)).throughput.x;
    }
    scale /= float(samples);
    bias /= float(samples);
    diffuse /= float(samples);
    
    return {scale, bias, diffuse};
}

__global__ void computeSpecularLUT(float4* data, uint width, uint height, uint samples) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Map pixel coordinates to viewing angle (cosTheta) and roughness (alpha)
    const float cosTheta = float(x) / (width - 1);  // [0, 1]
    const float alpha = float(y) / (height - 1);    // [0, 1]
    
    const auto result = integrateSpecular(alpha, cosTheta, samples);
    
    // Store the result. Using float4 for compatibility with texture formats.
    data[y * width + x] = make_float4(result, 1.0f);
}

// Function to allocate a texture, compute the directional albedo, and save it to a file
bool computeAndSaveDirectionalAlbedo(const char* filename, uint width, uint height, uint samples = 1024) {
    // Allocate device memory for the texture
    float4* data = nullptr;
    check(cudaMallocManaged(&data, width * height * sizeof(float4)));
    
    // Define the thread block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch the kernel to compute the directional albedo
    computeSpecularLUT<<<gridSize, blockSize>>>(data, width, height, samples);
    check(cudaDeviceSynchronize());
    
    // Convert to 8-bit per channel for saving (RGBA format)
    std::vector<unsigned char> image_data(width * height * 4);
    
    // Convert floating-point values to 8-bit unsigned char
    for (uint i = 0; i < width * height; ++i) {
        image_data[i * 4 + 0] = (unsigned char)(fminf(fmaxf(data[i].x, 0.0f), 1.0f) * 255.0f); // R
        image_data[i * 4 + 1] = (unsigned char)(fminf(fmaxf(data[i].y, 0.0f), 1.0f) * 255.0f); // G
        image_data[i * 4 + 2] = (unsigned char)(fminf(fmaxf(data[i].z, 0.0f), 1.0f) * 255.0f); // B
        image_data[i * 4 + 3] = (unsigned char)(fminf(fmaxf(data[i].w, 0.0f), 1.0f) * 255.0f); // A
    }

    cudaFree(data); // Free device memory
    
    // Save image to file using stb_image_write
    int result = stbi_write_png(filename, width, height, 4, image_data.data(), width * 4);
    
    // Free image data
    
    // Return true if successful, false otherwise
    return result != 0;
}

void createLUTTexture(uint width, uint height, uint samples) {

    cudaArray_t cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindFloat);
    check(cudaMallocArray(&cuArray, &channelDesc, width, height));

    cudaResourceDesc resDesc = {
        .resType = cudaResourceTypeArray,
        .res = { .array = cuArray },
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
    cudaTextureObject_t texObj;
    check(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
}
