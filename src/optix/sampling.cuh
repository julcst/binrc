#pragma once

#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "params.cuh"

__device__ constexpr float3 sampleCosineHemisphere(const float2& rand) {
    const auto phi = TWO_PI * rand.x;
    const auto sinTheta = sqrtf(1.0f - rand.y);
    const auto cosTheta = sqrtf(rand.y);
    return {cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta};
}

__device__ constexpr float cosineHemispherePDF(float cosTheta) {
    return cosTheta * INV_PI;
}

struct LightSample {
    float3 emission;
    float3 wi;
    float cosThetaL;
    float dist;
    float pdf;
    float3 position;
    float3 n;
};

// Binary search for the index of the light source
// FIXME: Fix cudaInvalidMemoryAccess
__device__ inline EmissiveTriangle sampleLightTable(float r) {
    uint left = 0;
    uint right = params.lightTableSize - 1;
    while (left < right) {
        const uint mid = (left + right) / 2;
        if (params.lightTable[mid].cdf < r) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return params.lightTable[left];
}

__device__ inline EmissiveTriangle sampleLightTableUniform(float r) {
    const uint index = clamp(r * params.lightTableSize, 0, params.lightTableSize - 1);
    auto light = params.lightTable[index];
    light.weight = 1.0f / params.lightTableSize;
    return light;
}

__device__ __forceinline__ float3 sampleBarycentrics(const float2& rand) {
    const auto s = sqrtf(rand.y);
    const auto u = 1.0f - s;
    const auto v = rand.x * s;
    return {u, v, 1.0f - u - v};
}

__device__ inline LightSample sampleLightSource(const EmissiveTriangle& light, const float2& rand, const float3& x) {
    // Sample a barycentric coordinate on the triangle uniformly
    const auto bary = sampleBarycentrics(rand);

    // Get light point information
    const auto position = bary.x * light.v0 + bary.y * light.v1 + bary.z * light.v2;
    const auto n = normalize(bary.x * light.n0 + bary.y * light.n1 + bary.z * light.n2);
    const auto emission = params.materials[light.materialID].emission;

    const auto dir = position - x;
    const auto dist2 = dot(dir, dir);
    const auto dist = sqrtf(dist2);
    const auto wi = safediv(dir, dist);
    const auto cosThetaL = dot(wi, n);

    // PDF of sampling the triangle and the point on the triangle
    // const auto pdfPoint = light.weight / light.area; // In area measure // TODO: Precalculate
    const auto pdf = safediv(light.weight * dist2, light.area * abs(cosThetaL)); // In solid angle measure

    return {emission, wi, cosThetaL, dist, pdf, position, n};
}

__device__ inline float lightPdfUniform(const float3& wi, const float dist, const float3& lightNormal, const float area) {
    const auto cosThetaL = abs(dot(wi, lightNormal));
    return safediv(dist * dist, area * cosThetaL * params.lightTableSize); // In solid angle measure
}

__device__ inline LightSample sampleLight(const float randSrc, const float2& randSurf, const float3& x) {
    const auto light = sampleLightTableUniform(randSrc);
    return sampleLightSource(light, randSurf, x);
}

struct LightDirSample {
    float3 wo;
    float3 n;
    float3 position;
    float3 emission;
};

__device__ inline LightDirSample samplePhoton(const float randSrc, const float2& randSurf, const float2& randDir) {
    const auto light = sampleLightTableUniform(randSrc);

    // Sample a barycentric coordinate on the triangle uniformly
    const auto bary = sampleBarycentrics(randSurf);

    // Get light point information
    const auto position = bary.x * light.v0 + bary.y * light.v1 + bary.z * light.v2;
    const auto n = normalize(bary.x * light.n0 + bary.y * light.n1 + bary.z * light.n2);
    const auto emission = params.materials[light.materialID].emission;

    const auto tangentToWorld = buildTBN(n);
    const auto wo = tangentToWorld * sampleCosineHemisphere(randDir);

    // pdf = light.weight / light.area;
    // pdf = light.weight * cosThetaL / (light.area * PI); // In solid angle measure
    const auto weight = light.area / light.weight * INV_PI;

    return {wo, n, position, emission * weight};
}

__device__ constexpr float balanceHeuristic(float pdf1, float pdf2) {
    return safediv(pdf1, pdf1 + pdf2);
}

__device__ constexpr float powerHeuristic(float pdf1, float pdf2) {
    const auto f1 = pdf1 * pdf1;
    const auto f2 = pdf2 * pdf2;
    return safediv(f1, f1 + f2);
}

__device__ __forceinline__ Instance sampleInstance(const Instance* instances, const uint instanceCount, const float rand) {
    uint left = 0;
    uint right = instanceCount - 1;
    while (left < right) {
        const uint mid = left + (right - left) / 2;
        if (instances[mid].cdf < rand) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    // Sanity checks
    if (rand > instances[left].cdf) printf("sampleInstance rand %f > cdf[i] %f\n", rand, instances[left].cdf);
    if (left > 0 && rand < instances[left - 1].cdf) printf("sampleInstance rand %f < cdf[i-1] %f\n", rand, instances[left - 1].cdf);
    return instances[left];
}

__device__ __forceinline__ uint sampleMeshTriangleIndex(const HitData& mesh, const float& rand) {
    uint left = 0;
    uint right = mesh.triangleCount - 1;
    while (left < right) {
        const uint mid = left + (right - left) / 2;
        if (mesh.cdfBuffer[mid] < rand) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    // Sanity checks
    if (rand > mesh.cdfBuffer[left]) printf("sampleMeshTriangleIndex rand %f > cdf[i] %f\n", rand, mesh.cdfBuffer[left]);
    if (left > 0 && rand < mesh.cdfBuffer[left - 1]) printf("sampleMeshTriangleIndex rand %f < cdf[i-1] %f\n", rand, mesh.cdfBuffer[left - 1]);
    return left;
}

struct Surface {
    float3 position;
    float3 normal;
    float3 baseColor;
    float3 emission;
    float transmission;
    float metallic;
    float roughness;
};

__device__ __forceinline__ Surface getSurface(const Material* materials, const Instance& inst, const uint triangleIndex, const float3& bary) {
    Surface surf;
    const auto material = materials[inst.geometry.materialID];
    const auto indices = inst.geometry.indexBuffer[triangleIndex];
    const auto v0 = inst.geometry.vertexData[indices.x];
    const auto v1 = inst.geometry.vertexData[indices.y];
    const auto v2 = inst.geometry.vertexData[indices.z];

    const auto localPos = bary.x * v0.position + bary.y * v1.position + bary.z * v2.position;
    const auto localNorm = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;
    const auto localTangentWithOrientation = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;
    const auto texCoord = bary.x * v0.texCoord + bary.y * v1.texCoord + bary.z * v2.texCoord;

    surf.position = make_float3(inst.localToWorld * make_float4(localPos, 1.0f));
    surf.baseColor = material.baseColor;
    surf.emission = material.emission;
    surf.transmission = material.transmission;
    surf.metallic = material.metallic;
    surf.roughness = material.roughness;

    if (material.baseMap) surf.baseColor *= make_float3(tex2D<float4>(material.baseMap, texCoord.x, texCoord.y));

    if (material.mrMap) {
        const auto mr = tex2D<float4>(material.mrMap, texCoord.x, texCoord.y);
        surf.emission *= mr.x;
        surf.roughness *= mr.y;
    }

    if (material.normalMap) { // MikkTSpace normal mapping
        const auto tangentOrientation = localTangentWithOrientation.w;
        const auto localTangent = make_float3(localTangentWithOrientation);
        const auto tangentSpaceNormal = make_float3(tex2D<float4>(material.normalMap, texCoord.x, texCoord.y)) * 2.0f - 1.0f;
        const auto localBitangent = cross(localNorm, localTangent) * tangentOrientation;
        surf.normal = normalize(inst.normalToWorld * (tangentSpaceNormal.x * localTangent + tangentSpaceNormal.y * localBitangent + tangentSpaceNormal.z * localNorm));
    } else {
        surf.normal = normalize(inst.normalToWorld * localNorm);
    }

    return surf;
}

__device__ __forceinline__ Surface sampleScene(const Instance* instances, const uint instanceCount, const Material* materials, const float randSrc, const float2& randSurf) {
    Instance inst = sampleInstance(instances, instanceCount, randSrc);
    const float randTri = (inst.cdf - randSrc) / inst.pdf;
    if (randTri > 1.0f || randTri < 0.0f) printf("randTri (%f - %f) / %f = %f\n", inst.cdf, randSrc, inst.pdf, randTri);
    const auto triangleIndex = sampleMeshTriangleIndex(inst.geometry, randTri);
    const auto bary = sampleBarycentrics(randSurf);
    return getSurface(materials, inst, triangleIndex, bary);
}