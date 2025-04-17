#include <cuda_runtime.h>
#include <cudamath.cuh>
#include <optix/sampling.cuh>

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
__host__ __device__ __forceinline__ float3 integrateSpecular(float roughness, float cosTheta, uint samples) {
    const float alpha = roughness * roughness;
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
        const float NdotL = max(wi.z, 0.0f); // TODO: Is this correct?
        const float HdotV = abs(dot(wm, wo));

        // Calculate the various BRDF components
        const float Ffactor = pow5(1.0f - HdotV);
        const float LambdaL = Lambda_TrowbridgeReitz(NdotL, alpha2);
        const float LambdaV = Lambda_TrowbridgeReitz(NdotV, alpha2);
        
        // Specular BRDF weighted by cosine term and importance sampling weight
        const float brdf = (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV);
        
        scale += (1.0f - Ffactor) * brdf;
        bias += Ffactor * brdf;
        
        //diffuse += sampleBrentBurley(rand, wo, NdotV, n, alpha, {}, make_float3(1.0f)).throughput.x;
        diffuse += sampleLambertian(rand, {}, make_float3(1.0f)).throughput.x;
    }
    scale /= float(samples);
    bias /= float(samples);
    diffuse /= float(samples);
    
    return {scale, bias, diffuse};
}

__global__ void computeSpecularLUT(cudaSurfaceObject_t surfObj, uint width, uint height, uint samples) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Map pixel coordinates to viewing angle (cosTheta) and roughness (alpha)
    const float cosTheta = float(x) / (width - 1);  // [0, 1]
    const float alpha = float(y) / (height - 1);    // [0, 1]
    
    const auto result = integrateSpecular(alpha, cosTheta, samples);
    
    // Store the result. Using float4 for compatibility with texture formats.
    // surf2Dwrite expects byte offset for x, not pixel index
    surf2Dwrite(make_uchar4(result.x * 255, result.y * 255, result.z * 255, 255), surfObj, x * sizeof(uchar4), y);
}