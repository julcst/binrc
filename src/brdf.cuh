#pragma once

#include "cudamath.cuh"

/**
 * Schlick's approximation for the Fresnel term (see https://en.wikipedia.org/wiki/Schlick%27s_approximation).
 * The Fresnel term describes how light is reflected at the surface.
 * For conductors the reflection coefficient R0 is chromatic, for dielectrics it is achromatic.
 * R0 = ((n1 - n2) / (n1 + n2))^2 with n1, n2 being the refractive indices of the two materials.
 * We can set n1 = 1.0 (air) and n2 = IoR of the material.
 * Most dielectrics have an IoR near 1.5 => R0 = ((1 - 1.5) / (1 + 1.5))^2 = 0.04.
 */
__device__ constexpr float3 F_SchlickApprox(float HdotV, const float3& R0) {
    return R0 + (1.0f - R0) * pow5(1.0f - HdotV);
}

/**
 * Trowbridge-Reitz normal distribution function
 * See: PBR Book by Matt Pharr and Greg Humphreys
 * See: Physically Based Shading at Disney by Brent Burley
 * See: Microfacet Models for Refraction through Rough Surfaces by Bruce Walter et al.
 */
__device__ constexpr float D_TrowbridgeReitz(float NdotH, float alpha2) {
    const auto cosTheta = NdotH;
    const auto cos2Theta = cosTheta * cosTheta;
    const auto sin2Theta = 1.0f - cos2Theta;
    return alpha2 / (PI * pow2(alpha2 * cos2Theta + sin2Theta));
}

/**
 * Lambda for the Trowbridge-Reitz NDF
 * Measures invisible masked microfacet area per visible microfacet area.
 */
__device__ constexpr float Lambda_TrowbridgeReitz(float NdotV, float alpha2) {
    const auto cosTheta = NdotV;
    const auto cos2Theta = cosTheta * cosTheta;
    const auto sin2Theta = 1.0f - cos2Theta;
    const auto tan2Theta = sin2Theta / cos2Theta;
    return (-1.0f + sqrtf(1.0f + alpha2 * tan2Theta)) * 0.5f;
}

/**
 * Smith's shadowing-masking function for the Trowbridge-Reitz NDF.
 */
__device__ constexpr float G2_TrowbridgeReitz(float NdotL, float NdotV, float alpha2) {
    const auto lambdaL = Lambda_TrowbridgeReitz(NdotL, alpha2);
    const auto lambdaV = Lambda_TrowbridgeReitz(NdotV, alpha2);
    return 1.0f / (1.0f + lambdaL + lambdaV);
}

/**
 * Smith's shadowing-masking function for the Trowbridge-Reitz NDF.
 */
__device__ constexpr float G1_TrowbridgeReitz(float NdotV, float alpha2) {
    const auto lambdaV = Lambda_TrowbridgeReitz(NdotV, alpha2);
    return 1.0f / (1.0f + lambdaV);
}

__device__ constexpr float3 disneyBRDF(const float3& wo, const float3& wi, const float3& n, const float3& albedo, float metallic, float alpha) {
    const auto F0 = mix(make_float3(0.04f), albedo, metallic);
    const auto alpha2 = alpha * alpha;
    const auto H = normalize(wo + wi);
    const auto NdotH = dot(n, H);
    const auto NdotV = dot(n, wo);
    const auto NdotL = dot(n, wi);
    const auto HdotV = dot(H, wo);

    const auto F = F_SchlickApprox(HdotV, F0);
    const auto D = D_TrowbridgeReitz(NdotH, alpha2);
    const auto G = G2_TrowbridgeReitz(NdotL, NdotV, alpha2);
    const auto specular = F * D * G / (4.0f * NdotV * NdotL);

    const auto FD90 = 0.5f + 2.0f * alpha * HdotV * HdotV;
    const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - NdotL)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - NdotV));
    const auto diffuse = albedo * response;

    return specular + diffuse;
}

/**
 * Sample visible normal distribution function using the algorithm
 * from "Sampling Visible GGX Normals with Spherical Caps" by Dupuy et al. 2023.
 * https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf
 * Implementation from https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 * @note Isotropic world space version
 */
__device__ constexpr float3 sampleVNDFTrowbridgeReitz(const float2& u, const float3& wi, float3 n, float alpha) {
    // Dirac function for alpha = 0
    if (alpha == 0.0f) return n;
    // decompose the vector in parallel and perpendicular components
    const auto wi_z = n * dot(wi, n);
    const auto wi_xy = wi - wi_z;
    // warp to the hemisphere configuration
    const auto wiStd = normalize(wi_z - alpha * wi_xy);
    // sample a spherical cap in (-wiStd.z, 1]
    const auto wiStd_z = dot(wiStd, n);
    const auto phi = (2.0f * u.x - 1.0f) * PI;
    const auto z = (1.0f - u.y) * (1.0f + wiStd_z) - wiStd_z;
    const auto sinTheta = sqrtf(1.0f - z * z);
    const auto x = sinTheta * cosf(phi);
    const auto y = sinTheta * sinf(phi);
    const auto cStd = make_float3(x, y, z);
    // reflect sample to align with normal
    const auto up = make_float3(0.0f, 0.0f, 1.0f);
    const auto wr = n + up;
    // prevent division by zero
    const auto wrz_safe = max(wr.z, 1e-10f); // NOTE: Important for the case when wr.z is close to zero
    const auto c = dot(wr, cStd) * wr / wrz_safe - cStd;
    // compute halfway direction as standard normal
    const auto wmStd = c + wiStd;
    const auto wmStd_z = n * dot(n, wmStd);
    const auto wmStd_xy = wmStd_z - wmStd;
    // warp back to the ellipsoid configuration
    const auto wm = normalize(wmStd_z + alpha * wmStd_xy);
    // return final normal
    return wm;
}

/**
 * Sample visible normal distribution function using the algorithm
 * from "Sampling Visible GGX Normals with Spherical Caps" by Dupuy et al. 2023.
 * https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf
 * Implementation from https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 * @note Anisotropic tangent space version
 */
__device__ constexpr float3 sampleVNDFTrowbridgeReitz(const float2& rand, const float3& wi, const float2& alpha) {
    // warp to the hemisphere configuration
    const auto wiStd = normalize(make_float3(make_float2(wi) * alpha, wi.z));
    // sample a spherical cap in (-wi.z, 1]
    const auto phi = (2.0f * rand.x - 1.0f) * PI;
    const auto z = fmaf(1.0f - rand.y, 1.0f + wiStd.z, -wiStd.z);
    const auto sinTheta = sqrtf(clamp(1.0f - z * z, 0.0f, 1.0f));
    const auto x = sinTheta * cosf(phi);
    const auto y = sinTheta * sinf(phi);
    const auto c = make_float3(x, y, z);
    // compute halfway direction as standard normal
    const auto wmStd = c + wiStd;
    // warp back to the ellipsoid configuration
    const auto wm = normalize(make_float3(make_float2(wmStd) * alpha, wmStd.z));
    // return final normal
    return wm;
}

/**
 * PDF for the visible normal distribution function.
 * From: https://auzaiffe.wordpress.com/2024/04/15/vndf-importance-sampling-an-isotropic-distribution/
 */
__device__ float VNDFPDFIsotropic(float3 wo, float3 wi, float alpha2, float3 n) {
    float3 wm = normalize(wo + wi);
    float zm = dot(wm, n);
    float zi = dot(wi, n);
    float nrm = rsqrt((zi * zi) * (1.0f - alpha2) + alpha2);
    float sigmaStd = (zi * nrm) * 0.5f + 0.5f;
    float sigmaI = sigmaStd / nrm;
    float nrmN = (zm * zm) * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PI * 4.0f * nrmN * nrmN * sigmaI);
}

__device__ constexpr float3 sampleCosineHemisphere(const float2& rand) {
    const auto phi = TWO_PI * rand.x;
    const auto sinTheta = sqrtf(1.0f - rand.y);
    const auto cosTheta = sqrtf(rand.y);
    return make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
}

struct SampleResult {
    float3 direction;
    float3 throughput;
};

__device__ constexpr SampleResult sampleTrowbridgeReitz(const float2& rand, const float3& wo, float cosThetaO, const float3& n, float alpha, const float3& F0) {
    const auto wm = sampleVNDFTrowbridgeReitz(rand, wo, n, alpha);
    const auto wi = reflect(wo, wm);
    const auto cosThetaD = dot(wo, wm); // = dot(wi, wm)
    const auto cosThetaI = dot(wi, n);
    const auto F = F_SchlickApprox(cosThetaD, F0);
    const auto alpha2 = alpha * alpha;
    const auto LambdaL = Lambda_TrowbridgeReitz(cosThetaI, alpha2);
    const auto LambdaV = Lambda_TrowbridgeReitz(cosThetaO, alpha2);
    const auto specular = F * (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV); // = F * (G2 / G1)
    return {wi, specular};
}

__device__ constexpr SampleResult sampleTrowbridgeReitzTransmission(const float2& rand, const float3& wo, float cosThetaO, const float3& n, float alpha, const float3& F0, const float3& albedo, bool inside) {
    const auto wm = sampleVNDFTrowbridgeReitz(rand, wo, n, alpha);
    const auto wi = refract(wo, wm, inside ? 1.0f / 1.5f : 1.5f / 1.0f);
    const auto cosThetaD = dot(wo, wm); // = dot(wi, wm)
    const auto cosThetaI = dot(wo, n);
    const auto F = F_SchlickApprox(cosThetaD, F0);
    const auto alpha2 = alpha * alpha;
    const auto LambdaL = Lambda_TrowbridgeReitz(cosThetaI, alpha2);
    const auto LambdaV = Lambda_TrowbridgeReitz(cosThetaO, alpha2);
    const auto transmission = albedo * (1.0f - F) * (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV); // = (1 - F) * (G2 / G1)
    return {wi, transmission};
}

__device__ constexpr SampleResult samplePerfectTransmission(const float3& wo, const float3& n, const float3& albedo, bool inside) {
    const auto wm = n;
    const auto wi = refract(wo, wm, inside ? 1.0f / 1.5f : 1.5f / 1.0f);
    const auto transmission = albedo;
    return {wi, transmission};
}

__device__ constexpr SampleResult sampleBrentBurley(const float2& rand, const float3& wo, float cosThetaO, const float3& n, float alpha, const float3x3& tangentToWorld, const float3& albedo) {
    const auto wi = tangentToWorld * sampleCosineHemisphere(rand);
    const auto wm = normalize(wi + wo);
    const auto cosThetaD = dot(wo, wm); // = dot(wi, wm)
    const auto cosThetaI = dot(wi, n);
    const auto FD90 = 0.5f + 2.0f * alpha * cosThetaD * cosThetaD;
    const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - cosThetaI)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - cosThetaO));
    // NOTE: We drop the 1.0 / PI prefactor
    const auto diffuse = albedo * response;
    return {wi, diffuse};
}

struct LightSample {
    float3 emission;
    float3 wi;
    float cosThetaL;
    float dist;
    float pdf;
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
    const uint index = r * params.lightTableSize;
    auto light = params.lightTable[index];
    light.weight = 1.0f / params.lightTableSize;
    return light;
}

__device__ inline LightSample sampleLightSource(const EmissiveTriangle& light, const float2& rand, const float3& x) {
    // Sample a barycentric coordinate on the triangle uniformly
    const auto s = sqrtf(rand.y);
    const auto u = 1.0f - s;
    const auto v = rand.x * s;
    const auto w = 1.0f - u - v;

    // Get light point information
    const auto position = u * light.v0 + v * light.v1 + w * light.v2;
    const auto n = normalize(u * light.n0 + v * light.n1 + w * light.n2);
    const auto emission = params.materials[light.materialID].emission;

    const auto dir = position - x;
    const auto dist2 = dot(dir, dir);
    const auto dist = sqrtf(dist2);
    const auto wi = dir / dist;
    const auto cosThetaL = dot(-wi, n);

    // PDF of sampling the triangle and the point on the triangle
    // const auto pdfPoint = light.weight / light.area; // In area measure
    // TODO: Optimize away one division
    const auto pdf = (light.weight * dist2) / (light.area * cosThetaL); // In solid angle measure

    return {emission, wi, cosThetaL, dist, pdf};
}

__device__ inline LightSample sampleLight(const float3& rand, const float3& x) {
    const auto light = sampleLightTable(rand.x);
    return sampleLightSource(light, make_float2(rand.y, rand.z), x);
}

__device__ inline float lightPDF(const LightSample& light) {
}