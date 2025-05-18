#pragma once

#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "params.cuh"

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
    return safediv(alpha2, PI * pow2(alpha2 * cos2Theta + sin2Theta));
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
    if (!isfinite(tan2Theta)) return 0.0f; // Avoid NaN
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

__device__ constexpr float VNDF_TrowbridgeReitz(float NdotV, float NdotH, float HdotV, float alpha2) {
    const auto G1 = G1_TrowbridgeReitz(NdotV, alpha2);
    const auto cosTheta = NdotV;
    const auto D = D_TrowbridgeReitz(NdotH, alpha2);
    return safediv(G1 * D * HdotV, cosTheta);
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
    if (alpha <= 1e-6f) return n;
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
    const auto c = safediv(dot(wr, cStd) * wr, wr.z) - cStd;
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
    const float3 c = {x, y, z};
    // compute halfway direction as standard normal
    const auto wmStd = c + wiStd;
    // warp back to the ellipsoid configuration
    const auto wm = normalize(make_float3(make_float2(wmStd) * alpha, wmStd.z));
    // return final normal
    return wm;
}

__device__ constexpr float3 sampleCosineHemisphere(const float2& rand) {
    const auto phi = TWO_PI * rand.x;
    const auto sinTheta = sqrtf(1.0f - rand.y);
    const auto cosTheta = sqrtf(rand.y);
    return {cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta};
}

__device__ constexpr float cosineHemispherePDF(float cosTheta) {
    return cosTheta * INV_PI;
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
    // NOTE: cosThetaI is included in the VNDF sampling pdf
    const auto specular = F * (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV); // = F * (G2 / G1)
    return {wi, specular};
}

__device__ constexpr SampleResult sampleTrowbridgeReitzTransmission(const float2& rand, const float3& wo, float cosThetaO, const float3& n, float alpha, const float3& F0, const float3& albedo, float eta) {
    const auto wm = sampleVNDFTrowbridgeReitz(rand, wo, n, alpha); // TODO: Sample wm for transmission and reflection together
    const auto wi = refract(wo, wm, eta);
    const auto cosThetaD = dot(wo, wm); // = dot(wi, wm)
    const auto cosThetaI = dot(wo, n); // TODO: Check if this is correct
    const auto F = F_SchlickApprox(cosThetaD, F0);
    const auto alpha2 = alpha * alpha;
    const auto LambdaL = Lambda_TrowbridgeReitz(cosThetaI, alpha2);
    const auto LambdaV = Lambda_TrowbridgeReitz(cosThetaO, alpha2);
    const auto transmission = albedo * max(1.0f - F, 0.0f) * (1.0f + LambdaV) / (1.0f + LambdaL + LambdaV); // = (1 - F) * (G2 / G1)
    if (isnegative(transmission)) printf("Transmission is negative: %f %f %f\n", transmission.x, transmission.y, transmission.z);
    return {wi, transmission};
}

__device__ constexpr SampleResult sampleLambertian(const float2& rand, const float3x3& tangentToWorld, const float3& albedo) {
    const auto wi = tangentToWorld * sampleCosineHemisphere(rand);
    const auto diffuse = albedo;
    return {wi, diffuse};
}

// FIXME: This fails the white furnace test
__device__ constexpr SampleResult sampleBrentBurley(const float2& rand, const float3& wo, float cosThetaO, const float3& n, float alpha, const float3x3& tangentToWorld, const float3& albedo) {
    const auto wi = tangentToWorld * sampleCosineHemisphere(rand);
    const auto wm = normalize(wi + wo);
    const auto cosThetaD = dot(wo, wm); // = dot(wi, wm)
    const auto cosThetaI = dot(wi, n);
    const auto FD90 = 0.5f + 2.0f * alpha * cosThetaD * cosThetaD;
    const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - cosThetaI)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - cosThetaO));
    // NOTE: We drop the 1.0 / PI  and the NdotL prefactor because we are sampling the hemisphere
    const auto diffuse = albedo * response;
    return {wi, diffuse};
}

struct LightContext {
    float alpha2;
    float NdotH;
    float NdotV;
    float NdotL;
    float HdotL;
    float HdotV;
    float eta;
    float pSpecular;
    float pDiffuse;
    float pTransmission;
};

__device__ constexpr float disneyPdf(const LightContext& ctx) {
    if (ctx.NdotL < 0.0f) return 0.0f; // TODO: Why is this necessary?
    const auto D = D_TrowbridgeReitz(ctx.NdotH, ctx.alpha2);
    const auto lambdaV = Lambda_TrowbridgeReitz(max(ctx.NdotV, 0.0f), ctx.alpha2);
    const auto G1 = 1.0f / (1.0f + lambdaV);
    const auto VNDF = G1 * D;
    const auto pdfSpecular = safediv(VNDF, 4.0f * max(ctx.NdotV, 0.0f));
    const auto pdfTransmission = safediv(VNDF * abs(ctx.HdotL), pow2(ctx.HdotL + ctx.HdotV / ctx.eta));

    const auto pdfDiffuse = max(ctx.NdotL, 0.0f) * INV_PI;

    const auto pdf = ctx.pSpecular * pdfSpecular + ctx.pDiffuse * pdfDiffuse + ctx.pTransmission * pdfTransmission;

    return pdf;
}

struct MISSampleResult {
    float3 direction;
    float3 throughput;
    float pdf;
    bool isDirac;
    bool isSpecular;
};

__device__ constexpr MISSampleResult sampleDisney(const float rType, const float2& rMicrofacet, const float2& rDiffuse, const float3& wo, const float3& n, const bool inside, const float3& baseColor, const float metallic, const float alpha, const float transmission) {
    const auto F0 = mix(make_float3(0.04f), baseColor, metallic); // Specular base
    const auto albedo = (1.0f - metallic) * baseColor; // Diffuse base
    const auto eta = inside ? 1.0f / 1.5f : 1.5f / 1.0f;

    const auto NdotV = dot(n, wo);

    // Importance sampling weights
    const auto wSpecular = luminance(F_SchlickApprox(NdotV, F0));
    const auto wDiffuse = luminance(albedo);
    const auto pSpecular = safediv(wSpecular, wSpecular + wDiffuse);
    const auto pDiffuse = (1.0f - pSpecular) * (1.0f - transmission);
    const auto pTransmission = (1.0f - pSpecular) * transmission;

    auto throughput = make_float3(0.0f);
    auto wi = make_float3(0.0f);
    auto isSpecular = true;

    if (rType < pSpecular) { 
        // Sample Trowbridge-Reitz specular
        const auto sample = sampleTrowbridgeReitz(rMicrofacet, wo, NdotV, n, alpha, F0);
        throughput = sample.throughput / pSpecular;
        wi = sample.direction;
    } else {
        // TODO: Proper weighting
        if (transmission < 0.5f) { // Sample Brent-Burley diffuse
            const auto tangentToWorld = buildTBN(n);
            // const auto sample = sampleBrentBurley(rDiffuse, wo, NdotV, n, alpha, tangentToWorld, albedo);
            const auto sample = sampleLambertian(rDiffuse, tangentToWorld, albedo);
            throughput = sample.throughput / pDiffuse;
            wi = sample.direction;
            isSpecular = false;
        } else {
            const auto sample = sampleTrowbridgeReitzTransmission(rMicrofacet, wo, NdotV, n, alpha, F0, baseColor, eta); // TODO: Handle no refraction case
            throughput = sample.throughput / pTransmission;
            wi = sample.direction;
        }
    }

    const auto H = normalize(wo + wi);
    const auto pdf = disneyPdf({alpha * alpha, abs(dot(n, H)), NdotV, dot(n, wi), dot(H, wi), dot(H, wo), eta, pSpecular, pDiffuse, pTransmission});

    const auto isDirac = alpha == 0.0f && pDiffuse == 0.0f;

    return {wi, throughput, pdf, isDirac, isSpecular};
}

struct BRDFResult {
    float3 throughput;
    float pdf;
    bool isDirac;
};

__device__ constexpr BRDFResult evalDisney(const float3& wo, const float3& wi, const float3& n, const float3& baseColor, float metallic, float alpha, float transmissiveness, bool inside) {
    const auto F0 = mix(make_float3(0.04f), baseColor, metallic);
    const auto albedo = (1.0f - metallic) * baseColor;
    const auto alpha2 = alpha * alpha;
    const auto H = normalize(wo + wi);
    const auto NdotH = abs(dot(n, H));
    const auto NdotV = dot(n, wo);
    const auto NdotL = dot(n, wi);
    const auto cNdotL = maxf(NdotL, 0.0f); // TODO: Return if NdotL < 0 and reflection
    const auto aNdotL = abs(NdotL);
    const auto HdotV = abs(dot(H, wo));
    const auto HdotL = abs(dot(H, wi));
    const auto eta = inside ? 1.0f / 1.5f : 1.5f / 1.0f;

    //if (NdotL <= 0.0f) return {{1.0f}, 1.0f};

    const auto F = F_SchlickApprox(HdotV, F0); // TODO: Different F0 inside and outside
    const auto D = D_TrowbridgeReitz(NdotH, alpha2);
    const auto lambdaL = Lambda_TrowbridgeReitz(cNdotL, alpha2);
    const auto lambdaV = Lambda_TrowbridgeReitz(NdotV, alpha2);
    const auto G1 = 1.0f / (1.0f + lambdaV);
    const auto G = 1.0f / (1.0f + lambdaL + lambdaV);
    const auto specular = safediv(F * D * G, 4.0f * NdotV);
    const auto transmission = safediv(albedo * max(1.0f - F, 0.0f) * D * G * HdotL * HdotV, NdotV * aNdotL * pow2(HdotL + HdotV / eta));
    const auto VNDF = G1 * D;
    const auto pdfSpecular = safediv(VNDF, 4.0f * NdotV);
    const auto pdfTransmission = safediv(VNDF * abs(HdotL), pow2(HdotL + HdotV / eta));

    // const auto FD90 = 0.5f + 2.0f * alpha * HdotV * HdotV;
    // const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - cNdotL)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - NdotV));
    const auto diffuse = albedo * cNdotL * INV_PI;
    const auto pdfDiffuse = cNdotL * INV_PI;

    const auto wSpecular = luminance(F_SchlickApprox(NdotV, F0));
    const auto wDiffuse = luminance(albedo);
    const auto pSpecular = safediv(wSpecular, wSpecular + wDiffuse);
    const auto pDiffuse = (1.0f - pSpecular) * (1.0f - transmissiveness);
    const auto pTransmission = (1.0f - pSpecular) * transmissiveness;
    const auto pdf = pSpecular * pdfSpecular + pDiffuse * pdfDiffuse + pTransmission * pdfTransmission;

    const auto isDirac = alpha == 0.0f && wDiffuse == 0.0f;

    if (isnegative(transmission)) printf("Transmission is negative: F %f %f %f D %f G %f HdotL %f HdotV %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, HdotL, HdotV, NdotL, NdotV, NdotH);
    if (isnegative(specular)) printf("Specular is negative: F %f %f %f D %f G %f HdotL %f HdotV %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, HdotL, HdotV, NdotL, NdotV, NdotH);
    if (isnegative(diffuse)) printf("Diffuse is negative: F %f %f %f D %f G %f HdotL %f HdotV %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, HdotL, HdotV, NdotL, NdotV, NdotH);
    if (isnegative(pdfSpecular)) printf("pdfSpecular is negative: %f %f %f %f %f %f %f\n", pdfSpecular, NdotV, NdotL, HdotL, HdotV, lambdaL, lambdaV);
    if (isnegative(pdfDiffuse)) printf("pdfDiffuse is negative: %f %f %f %f %f %f %f\n", pdfDiffuse, NdotV, NdotL, HdotL, HdotV, lambdaL, lambdaV);
    if (isnegative(pdfTransmission)) printf("pdfTransmission is negative: %f %f %f %f %f %f %f\n", pdfTransmission, NdotV, NdotL, HdotL, HdotV, lambdaL, lambdaV);
    if (HdotV - 1.0f > 1e-6f) printf("HdotV is greater than 1: %f H %f %f %f V %f %f %f\n", HdotV, H.x, H.y, H.z, wo.x, wo.y, wo.z);

    return {specular + mix(diffuse, transmission, transmissiveness), pdf, isDirac};
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

__device__ inline LightDirSample sampleLight(const float randSrc, const float2& randSurf, const float2& randDir) {
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
    const auto weight = light.area / light.weight;

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