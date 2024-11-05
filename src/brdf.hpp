#pragma once

#include "cudamath.cuh"

/**
 * Sample visible normal distribution function using the algorithm
 * from "Sampling Visible GGX Normals with Spherical Caps" by Dupuy et al. 2023.
 * https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf
 * Implementation from https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
__device__ constexpr float3 sampleVNDFTrowbridgeReitz(float2 u, float3 wi, float alpha, float3 n) {
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
    const auto sinTheta = sqrtf(max(1.0f - z * z, 0.0f));
    const auto x = sinTheta * cosf(phi);
    const auto y = sinTheta * sinf(phi);
    const auto cStd = make_float3(x, y, z);
    // reflect sample to align with normal
    const auto up = make_float3(0.0f, 0.0f, 1.0f);
    const auto wr = n + up;
    // prevent division by zero
    const auto wrz_safe = max(wr.z, 1e-6f);
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

__device__ constexpr float3 sampleCosineHemisphere(float2 rand) {
    const auto phi = TWO_PI * rand.x;
    const auto sinTheta = sqrtf(1.0f - rand.y);
    const auto cosTheta = sqrtf(rand.y);
    return make_float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

/**
 * Schlick's approximation for the Fresnel term (see https://en.wikipedia.org/wiki/Schlick%27s_approximation).
 * The Fresnel term describes how light is reflected at the surface.
 * For conductors the reflection coefficient R0 is chromatic, for dielectrics it is achromatic.
 * R0 = ((n1 - n2) / (n1 + n2))^2 with n1, n2 being the refractive indices of the two materials.
 * We can set n1 = 1.0 (air) and n2 = IoR of the material.
 * Most dielectrics have an IoR near 1.5 => R0 = ((1 - 1.5) / (1 + 1.5))^2 = 0.04.
 */
__device__ constexpr float3 F_SchlickApprox(float HdotV, float3 R0) {
    return R0 + (1.0f - R0) * pow5(1.0f - HdotV);
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
