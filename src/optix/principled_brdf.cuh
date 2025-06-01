#pragma once

#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "sampling.cuh"

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

__device__ constexpr float G1_Smith(float NdotV, float alpha2) {
    return 2.0f * NdotV / (sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV);
}

__device__ constexpr float G2_Smith(float NdotL, float NdotV, float alpha2) {
    const auto denomA = NdotV * sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    const auto denomB = NdotL * sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
    return 2.0f * NdotL * NdotV / (denomA + denomB);
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
    // calc halfway direction as standard normal
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
    // calc halfway direction as standard normal
    const auto wmStd = c + wiStd;
    // warp back to the ellipsoid configuration
    const auto wm = normalize(make_float3(make_float2(wmStd) * alpha, wmStd.z));
    // return final normal
    return wm;
}

// From: https://auzaiffe.wordpress.com/2024/04/15/vndf-importance-sampling-an-isotropic-distribution/
__device__ constexpr float pdfVNDF(float NdotL, float HdotN, float alpha2, float3 n) {
    float nrm = 1.0f / sqrt((NdotL * NdotL) * (1.0f - alpha2) + alpha2);
    float sigmaStd = (NdotL * nrm) * 0.5f + 0.5f;
    float sigmaI = sigmaStd / nrm;
    float nrmN = (HdotN * HdotN) * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * 4.0f * nrmN * nrmN * sigmaI);
}

__device__ constexpr float2 sampleUniformDiskPolar(const float2& rand) {
    const auto r = sqrt(rand.x);
    const auto theta = 2.0f * PI * rand.y;
    return {r * cos(theta), r * sin(theta)};
}

// From: https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
__device__ __forceinline__ float3 sampleWm(const float2& rand, const float3& w, const float alpha) {
    auto wh = normalize(make_float3(alpha * w.x, alpha * w.y, w.z)); // Transform w to hemisphere
    if (wh.z < 0.0f) wh = -wh; // Flip to face forward
    const auto basis = buildTBN(wh); // Find orthonormal basis
    auto p = sampleUniformDiskPolar(rand); // Sample disk
    // Warp hemispherical projection for visible normal sampling
    float h = sqrt(1.0f - pow2(p.x));
    p.y = mix((1.0f + wh.z) * 0.5f, h, p.y);
    // Reproject to hemisphere and transform normal to ellipsoid configuration>> 
    float pz = sqrt(max(0.0f, 1.0f - dot(p, p)));
    float3 nh = basis * make_float3(p.x, p.y, pz);
    return normalize(make_float3(alpha * nh.x, alpha * nh.y, max(1e-6f, nh.z)));
}

__device__ __forceinline__ float3 sampleWm(const float2& rand, const float3& w, const float3& n, const float alpha) {
    const auto TBN = buildTBN(n);
    return TBN * sampleWm(rand, w * TBN, alpha);
}

struct Sample {
    float3 direction;
    float3 throughput;
};

__device__ constexpr Sample sampleLambertian(const float2& rand, const float3x3& tangentToWorld, const float3& albedo) {
    const auto wi = tangentToWorld * sampleCosineHemisphere(rand);
    const auto diffuse = albedo;
    return {wi, diffuse};
}

// FIXME: This fails the white furnace test
__device__ constexpr Sample sampleBrentBurley(const float2& rand, const float3& wo, float cosThetaO, const float3& n, float alpha, const float3x3& tangentToWorld, const float3& albedo) {
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

struct BRDFResult {
    float3 throughput;
    float pdf;
    bool isDirac;
};

__device__ constexpr BRDFResult evalDisney(const float3& wo, const float3& wi, const float3& n, const float3& baseColor, float metallic, float alpha, float transmissiveness, bool inside) {
    const auto F0 = mix(make_float3(0.04f), baseColor, metallic); // TODO: Is F0 different inside and outside?
    const auto albedo = (1.0f - metallic) * baseColor;
    const auto alpha2 = alpha * alpha;
    auto NdotV = dot(n, wo);
    auto NdotL = dot(n, wi);
    const auto sameHemisphere = NdotV * NdotL >= 0.0f;
    //const auto inside = dot(n, wo) < 0.0f;
    const auto eta = inside ? 1.5f : 1.0f / 1.5f;

    float3 wm = make_float3(0.0f);
    if (sameHemisphere) {
        wm = normalize(wo + wi); // Microfacet normal in reflection case
    } else {
        wm = normalize(wo + wi / eta); // Microfacet normal in transmission case
        if (dot(wm, n) < 0.0f) {
            wm = -wm; // TODO: Check if this is correct
        }
    }
    const auto NdotH = abs(dot(n, wm));
    const auto HdotV = dot(wm, wo);
    const auto HdotL = dot(wm, wi);

    //if (NdotL <= 0.0f) return {{1.0f}, 1.0f};

    const auto wSpecular = luminance(F_SchlickApprox(abs(NdotV), F0));
    const auto wDiffuse = luminance(albedo);
    const auto pSpecular = safediv(wSpecular, wSpecular + wDiffuse);
    const auto pDiffuse = (1.0f - pSpecular) * (1.0f - transmissiveness);
    const auto pTransmission = (1.0f - pSpecular) * transmissiveness;

    const auto isDirac = alpha == 0.0f && wDiffuse == 0.0f;

    if (NdotL == 0.0f || NdotV == 0.0f || dot(wm, wm) == 0.0f) return {{make_float3(0.0f)}, 0.0f, isDirac};
    // Discard backfacing samples
    if (HdotL * NdotL < 0.0f || HdotV * NdotV < 0.0f) return {{make_float3(0.0f)}, 0.0f, isDirac}; // TODO: Check if this is correct

    NdotV = abs(NdotV);
    NdotL = abs(NdotL); // Always abs?

    const auto F = F_SchlickApprox(abs(HdotV), F0);
    const auto D = D_TrowbridgeReitz(NdotH, alpha2);
    const auto lambdaL = Lambda_TrowbridgeReitz(HdotL, alpha2); // abs?
    const auto lambdaV = Lambda_TrowbridgeReitz(HdotV, alpha2); // abs?
    // const auto G1 = 1.0f / (1.0f + lambdaV);
    // const auto G = 1.0f / (1.0f + lambdaL + lambdaV);
    const auto invG1 = 1.0f + lambdaV;
    const auto invG = 1.0f + lambdaL + lambdaV;

    auto pdf = 0.0f;
    auto throughput = make_float3(0.0f);

    if (sameHemisphere) {
        // Outgoing and incoming on same hemisphere => Diffuse + Specular
        const auto specular = F * safediv(D, invG * 4.0f * NdotV); // TODO: Handle dirac
        throughput += specular;
        // NOTE: D_VNDF = D * G1 * HdotV / NdotV
        const auto pdfSpecular = safediv(D, invG1 * 4.0f * NdotV); // = D_VNDF / (4 * HdotV)
        pdf += pSpecular * pdfSpecular;

        // This would be Burley's diffuse but it is not energy conserving
        // const auto FD90 = 0.5f + 2.0f * alpha * HdotV * HdotV;
        // const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - cNdotL)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - NdotV));
        const auto diffuse = albedo * ((1.0f - transmissiveness) * NdotL * INV_PI);
        throughput += diffuse;
        const auto pdfDiffuse = NdotL * INV_PI;
        pdf += pDiffuse * pdfDiffuse;

        // if (isnegative(specular)) printf("Specular is negative: F %f %f %f D %f G %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, NdotL, NdotV, NdotH);
        // if (isnegative(diffuse)) printf("Diffuse is negative: F %f %f %f D %f G %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, NdotL, NdotV, NdotH);
        // if (isnegative(pdfSpecular)) printf("pdfSpecular is negative: %f %f %f %f %f\n", pdfSpecular, NdotV, NdotL, lambdaL, lambdaV);
        // if (isnegative(pdfDiffuse)) printf("pdfDiffuse is negative: %f %f %f %f %f\n", pdfDiffuse, NdotV, NdotL, lambdaL, lambdaV);
    } else if (pTransmission > 0.0f) {
        // Transmission
        const auto transmission = albedo * (1.0f - F) * safediv(transmissiveness * D * abs(HdotL * HdotV), invG * NdotV * NdotL * pow2(HdotL + HdotV / eta));
        throughput += transmission;
        const auto pdfTransmission = safediv(D, invG1 * pow2(HdotL + HdotV / eta)); // Missing * abs(HdotL) ?
        pdf += pTransmission * pdfTransmission;

        // if (luminance(transmission) > 100.0f) printf("Transmission is larger than 100: T %f %f %f pdf %f D %f invG %f NV %f NL %f HV %f HL %f\n", transmission.x, transmission.y, transmission.z, pdfTransmission, D, invG, NdotV, NdotL, HdotV, HdotL);
        // if (isnegative(transmission)) printf("Transmission is negative: F %f %f %f D %f G %f HdotL %f HdotV %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, HdotL, HdotV, NdotL, NdotV, NdotH);
        // if (isnegative(pdfTransmission)) printf("pdfTransmission is negative: %f %f %f %f %f %f %f\n", pdfTransmission, NdotV, NdotL, HdotL, HdotV, lambdaL, lambdaV);
    }
    
    // if (HdotV - 1.0f > 1e-6f) printf("HdotV is greater than 1: %f wm %f %f %f V %f %f %f\n", HdotV, wm.x, wm.y, wm.z, wo.x, wo.y, wo.z);

    return {throughput, pdf, isDirac};
}

struct MaterialProperties {
    float3 F0;
    float3 albedo;
    float alpha2;
    float transmission;
};

__device__ constexpr MaterialProperties calcMaterialProperties(const float3& baseColor, float metallic, float alpha, float transmission) {
    const auto F0 = mix(make_float3(0.04f), baseColor, metallic); // TODO: Is F0 different inside and outside?
    const auto albedo = (1.0f - metallic) * baseColor;
    const auto alpha2 = alpha * alpha;

    return {F0, albedo, alpha2, transmission};
}

struct DisneyWeights {
    float specular;
    float diffuse;
    float transmission;
};

__device__ constexpr DisneyWeights calcDisneyWeights(const MaterialProperties& mat, float cosThetaO) {
    const auto approximateF = F_SchlickApprox(cosThetaO, mat.F0);

    // Importance sampling weights
    const auto wSpecular = luminance(approximateF); // TODO: Directional Albedo specular + transmission
    const auto wDiffuse = luminance(mat.albedo);

    const auto pSpecular = safediv(wSpecular, wSpecular + wDiffuse);
    const auto pDiffuse = (1.0f - pSpecular) * (1.0f - mat.transmission);
    const auto pTransmission = (1.0f - pSpecular) * mat.transmission;
    return {pSpecular, pDiffuse, pTransmission};
}

struct LightParams {
    float NdotV;
    float NdotL;
    bool sameHemisphere;
    float eta;
    float3 n;
    float3 wi;
    float3 wo;
};

__device__ constexpr LightParams calcLightParams(const float3& wo, const float3& wi, const float3& n) {
    const auto NdotV = dot(n, wo);
    const auto NdotL = dot(n, wi);
    const auto sameHemisphere = NdotV * NdotL >= 0.0f;
    const auto woIsInside = dot(n, wo) < 0.0f;
    const auto eta = woIsInside ? 1.5f : 1.0f / 1.5f;
    const auto surfaceN = woIsInside ? -n : n; // Flip normal if inside
    return {abs(NdotV), abs(NdotL), sameHemisphere, eta, surfaceN, wi, wo};
}

__device__ constexpr float3 calcMicrofacetNormal(const float3& wo, const float3& wi, const float3& n, float eta) {
    const auto sameHemisphere = dot(n, wo) * dot(n, wi) >= 0.0f;
    if (sameHemisphere) {
        return normalize(wo + wi); // Microfacet normal in reflection case
    } else {
        auto wm = normalize(wo + wi / eta); // Microfacet normal in transmission case
        if (dot(wm, n) < 0.0f) wm = -wm; // Flip to face forward
        return wm;
    }
}

__device__ constexpr BRDFResult evalDisneyWeighted(const float3& wo, const float3& wi, const float3& wm, const float3& n, const MaterialProperties& mat, const DisneyWeights& weights, bool inside) {
    const auto eta = inside ? 1.5f : 1.0f / 1.5f;
    const auto sameHemisphere = dot(n, wo) * dot(n, wi) >= 0.0f;
    const auto NdotH = abs(dot(n, wm));
    const auto HdotV = abs(dot(wm, wo));
    const auto HdotL = abs(dot(wm, wi));
    auto NdotV = abs(dot(n, wo));
    auto NdotL = abs(dot(n, wi));

    const auto isDirac = mat.alpha2 == 0.0f && weights.diffuse == 0.0f;

    // Discard samples with no contribution
    if (NdotL == 0.0f || NdotV == 0.0f || dot(wm, wm) == 0.0f) {
        return {{make_float3(0.0f)}, 0.0f, isDirac};
    }

    // Discard backfacing samples as they won't have any contribution
    if (dot(wm, wi) * dot(n, wi) < 0.0f || dot(wm, wo) * dot(n, wo) < 0.0f) {
        return {{make_float3(0.0f)}, 0.0f, isDirac}; // TODO: Check if this is correct
    }

    const auto F = F_SchlickApprox(HdotV, mat.F0);
    const auto D = mat.alpha2 == 0 ? 1.0f: D_TrowbridgeReitz(NdotH, mat.alpha2);
    const auto lambdaL = Lambda_TrowbridgeReitz(NdotL, mat.alpha2); // abs?
    const auto lambdaV = Lambda_TrowbridgeReitz(NdotV, mat.alpha2); // abs?
    const auto invG1 = 1.0f + lambdaV;
    const auto invG = 1.0f + lambdaL + lambdaV;

    auto pdf = 0.0f;
    auto throughput = make_float3(0.0f);

    if (sameHemisphere) {
        // Outgoing and incoming on same hemisphere => Diffuse + Specular
        // NOTE: We include NdotL in the BRDF weight
        // BRDF = F * D * G / (4 * NdotV * NdotL)
        // const auto specular = F * safediv(D, invG * 4.0f * NdotV); // = BRDF * NdotL
        // throughput += specular;
        // NOTE: D_VNDF = D * G1 * HdotV / NdotV
        const auto pdfSpecular = safediv(D, invG1 * 4.0f * NdotV); // = D_VNDF / (4 * HdotV)
        pdf += weights.specular * pdfSpecular;

        // This would be Burley's diffuse but it is not energy conserving
        // const auto FD90 = 0.5f + 2.0f * alpha * HdotV * HdotV;
        // const auto response = (1.0f + (FD90 - 1.0f) * pow5(1.0f - cNdotL)) * (1.0f + (FD90 - 1.0f) * pow5(1.0f - NdotV));
        // const auto diffuse = albedo * (1.0f - transmissiveness);
        // const auto diffuse = mat.albedo * ((1.0f - mat.transmission) * NdotL * INV_PI);
        // throughput += diffuse;
        const auto pdfDiffuse = NdotL * INV_PI;
        pdf += weights.diffuse * pdfDiffuse;

        throughput /= pdf;

        // NOTE: By smartly dividing the throughput by the pdf we can reduce the number of divisions
        const auto cDiffuse = (1.0f - mat.transmission) * mat.albedo;
        const auto k = 4.0f * NdotV * NdotL;
        const auto denom = 1.0f / (invG * (weights.diffuse * k * invG1 + weights.specular * PI * D));
        throughput = (invG1 * ((k * invG) * cDiffuse + (PI * D) * F)) * denom; // Single division throughput

        // if (isnegative(specular)) printf("Specular is negative: F %f %f %f D %f G %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, NdotL, NdotV, NdotH);
        // if (isnegative(diffuse)) printf("Diffuse is negative: F %f %f %f D %f G %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, G, NdotL, NdotV, NdotH);
        // if (isnegative(pdfSpecular)) printf("pdfSpecular is negative: %f %f %f %f %f\n", pdfSpecular, NdotV, NdotL, lambdaL, lambdaV);
        // if (isnegative(pdfDiffuse)) printf("pdfDiffuse is negative: %f %f %f %f %f\n", pdfDiffuse, NdotV, NdotL, lambdaL, lambdaV);
    } else if (weights.transmission > 0.0f) {
        // Transmission
        // const auto transmission = albedo * (1.0f - F) * safediv(transmissiveness * D * G * abs(HdotL * HdotV), NdotV * NdotL * pow2(HdotL + HdotV / eta));
        const auto transmission = mat.albedo * (1.0f - F) * mat.transmission * invG1 / (invG * weights.transmission); // TODO: Handle dirac
        throughput = transmission;
        // NOTE: D_VNDF = D * G1 * HdotV / NdotV
        const auto pdfTransmission = safediv(D, invG1 * pow2(HdotL + HdotV / eta)); // Missing * abs(HdotL) ?
        pdf = weights.transmission * pdfTransmission;

        // if (luminance(transmission) > 2.0f) printf("Transmission is larger than 2: %f %f %f %f %f %f %f %f %f %f %f %f\n", transmission.x, transmission.y, transmission.z, pdfTransmission, dot(n, wi), dot(n, wo), HdotL, HdotV, NdotH, eta, 1.0f / invG1, D);
        // if (isnegative(transmission)) printf("Transmission is negative: F %f %f %f D %f G %f HdotL %f HdotV %f NdotL %f NdotV %f NdotH %f\n", F.x, F.y, F.z, D, 1.0f / invG, HdotL, HdotV, NdotL, NdotV, NdotH);
        // if (!isnegative(transmission - 1.0f)) printf("Transmission is larger than 1: %f %f %f %f %f %f %f\n", transmission.x, transmission.y, transmission.z, NdotV, NdotL, HdotL, HdotV);
        // if (isnegative(pdfTransmission)) printf("pdfTransmission is negative: %f %f %f %f %f %f %f\n", pdfTransmission, NdotV, NdotL, HdotL, HdotV, lambdaL, lambdaV);
    }
    
    // if (HdotV - 1.0f > 1e-6f) printf("HdotV is greater than 1: %f wm %f %f %f V %f %f %f\n", HdotV, wm.x, wm.y, wm.z, wo.x, wo.y, wo.z);

    return {throughput, pdf, isDirac};
}

struct SampleResult {
    float3 direction;
    float3 throughput;
    float pdf;
    bool isDirac;
    bool isSpecular;
};

__device__ constexpr SampleResult sampleDisney(const float rType, const float2& rMicrofacet, const float2& rDiffuse, const float3& wo, const float3& n, const bool inside, const float3& baseColor, const float metallic, const float alpha, const float transmission) {
    const auto mat = calcMaterialProperties(baseColor, metallic, alpha, transmission);
    const auto eta = inside ? 1.5f : 1.0f / 1.5f; // Index of refraction
    const auto weights = calcDisneyWeights(mat, dot(n, wo)); // abs?

    auto wi = make_float3(0.0f);
    auto wm = make_float3(0.0f);
    auto isSpecular = true;

    if (rType < weights.diffuse) {
        // Lambertian
        const auto TBN = buildTBN(n);
        wi = TBN * sampleCosineHemisphere(rDiffuse);
        wm = normalize(wi + wo);
        isSpecular = false;
    } else {
        // Trowbridge-Reitz Microfacet sampling
        wm = sampleVNDFTrowbridgeReitz(rMicrofacet, wo, n, alpha);

        if (rType < weights.diffuse + weights.transmission && isnonzero(wi = refract(wo, wm, eta))) {
            // Transmission
        } else {
            // Reflection
            wi = reflect(wo, wm);
        }
    }

    const auto eval = evalDisneyWeighted(wo, wi, wm, n, mat, weights, inside);

    return {wi, eval.throughput, eval.pdf, eval.isDirac, isSpecular};
}
