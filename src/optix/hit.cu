#include <optix_device.h>
#include <cuda_runtime.h>

#include "cudamath.cuh"
#include "payload.cuh"
#include "params.cuh"

extern "C" __global__ void __closesthit__ch() {
    // Get optix built-in variables
    const auto bary2 = optixGetTriangleBarycentrics();
    const auto bary = make_float3(1.0f - bary2.x - bary2.y, bary2);
    const auto data = reinterpret_cast<HitData*>(optixGetSbtDataPointer());

    // Get triangle vertices
    const auto idx = data->indexBuffer[optixGetPrimitiveIndex()];
    const auto v0 = data->vertexData[idx.x];
    const auto v1 = data->vertexData[idx.y];
    const auto v2 = data->vertexData[idx.z];

    const auto e0 = v1.position - v0.position;
    const auto e1 = v2.position - v0.position;
    const auto area = 0.5f * length(optixTransformVectorFromObjectToWorldSpace(cross(e0, e1)));

    // Interpolate normal
    const auto objectSpaceNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;

    // Interpolate tangent
    const auto objectSpaceTangentWithOrientation = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;
    const auto objectSpaceTangent = make_float3(objectSpaceTangentWithOrientation);
    const auto worldSpaceTangent = optixTransformVectorFromObjectToWorldSpace(objectSpaceTangent);

    const auto texCoord = bary.x * v0.texCoord + bary.y * v1.texCoord + bary.z * v2.texCoord;

    // Get material
    const auto material = params.materials[data->materialID];
    
    auto baseColor = material.baseColor;
    if (material.baseMap) baseColor *= make_float3(tex2D<float4>(material.baseMap, texCoord.x, texCoord.y));

    auto mr = make_float2(material.metallic, material.roughness);
    if (material.mrMap) {
        auto t = tex2D<float4>(material.mrMap, texCoord.x, texCoord.y);
        mr *= make_float2(t.z, t.y);
    }

    // NOTE: Normal mapping produces artifacts with pathtracing: See Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing by Schüssler et al. 2017 for a solution
    if (material.normalMap) { // MikkTSpace normal mapping
        const auto tangentOrientation = objectSpaceTangentWithOrientation.w;
        const auto tangentSpaceNormal = make_float3(tex2D<float4>(material.normalMap, texCoord.x, texCoord.y)) * 2.0f - 1.0f;
        const auto objectSpaceBitangent = cross(objectSpaceNormal, objectSpaceTangent) * tangentOrientation;
        setNormal(normalize(optixTransformNormalFromObjectToWorldSpace(tangentSpaceNormal.x * objectSpaceTangent + tangentSpaceNormal.y * objectSpaceBitangent + tangentSpaceNormal.z * objectSpaceNormal)));
    } else {
        setNormal(normalize(optixTransformNormalFromObjectToWorldSpace(objectSpaceNormal)));
    }
    
    setBaseColor(baseColor);
    setTangent(worldSpaceTangent);
    setMetallic(mr.x);
    setEmission(material.emission);
    setRoughness(mr.y);
    setTransmission(params.flags & TRANSMISSION_FLAG ? material.transmission : 0.0f);
    setArea(area);
    setT(optixGetRayTmax());
}

extern "C" __global__ void __miss__ms() {
    const auto dir = optixGetWorldRayDirection();
    auto sky = make_float3(0.0f);

    setEmission(sky);
    setT(INFINITY);
}