#pragma once

#include <cuda_runtime.h>

#include "cudamathtypes.cuh"

using uint = unsigned int;

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 6.28318530717958647692f;

// Operators on float

__host__ __device__ constexpr float fract(float x) {
    return x - truncf(x);
}

__host__ __device__ constexpr float pow2(float x) {
    return x * x;
}

__host__ __device__ constexpr float pow3(float x) {
    return pow2(x) * x;
}

__host__ __device__ constexpr float pow4(float x) {
    return pow2(pow2(x));
}

__host__ __device__ constexpr float pow5(float x) {
    return pow4(x) * x;
}

__host__ __device__ constexpr float mix(float a, float b, float t) {
    return a + (b - a) * t;
}

__host__ __device__ constexpr float clamp(float x, float a, float b) {
    return max(a, min(b, x));
}

__host__ __device__ constexpr float safesqrt(float x) {
    return sqrtf(max(x, 0.0f));
}

// Operators on CUDA float2

__host__ __device__ constexpr float2 make_float2(float x) {
    return {x, x};
}

__host__ __device__ constexpr float2 make_float2(const float3& a) {
    return {a.x, a.y};
}

__host__ __device__ constexpr float2 make_float2(const float4& a) {
    return {a.x, a.y};
}

__host__ __device__ constexpr float2 operator+(const float2& a, const float2& b) {
    return {a.x + b.x, a.y + b.y};
}

__host__ __device__ constexpr float2 operator-(const float2& a, const float2& b) {
    return {a.x - b.x, a.y - b.y};
}

__host__ __device__ constexpr float2 operator*(const float2& a, const float2& b) {
    return {a.x * b.x, a.y * b.y};
}

__host__ __device__ constexpr float2 operator/(const float2& a, const float2& b) {
    return {a.x / b.x, a.y / b.y};
}

__host__ __device__ constexpr float2 operator+(const float2& a, float b) {
    return {a.x + b, a.y + b};
}

__host__ __device__ constexpr float2 operator-(const float2& a, float b) {
    return {a.x - b, a.y - b};
}

__host__ __device__ constexpr float2 operator*(const float2& a, float b) {
    return {a.x * b, a.y * b};
}

__host__ __device__ constexpr float2 operator/(const float2& a, float b) {
    return {a.x / b, a.y / b};
}

__host__ __device__ constexpr float2 operator+(float a, const float2& b) {
    return {a + b.x, a + b.y};
}

__host__ __device__ constexpr float2 operator-(float a, const float2& b) {
    return {a - b.x, a - b.y};
}

__host__ __device__ constexpr float2 operator*(float a, const float2& b) {
    return {a * b.x, a * b.y};
}

__host__ __device__ constexpr float2 operator/(float a, const float2& b) {
    return {a / b.x, a / b.y};
}

__host__ __device__ constexpr void operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
}

__host__ __device__ constexpr void operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
}

__host__ __device__ constexpr void operator*=(float2& a, const float2& b) {
    a.x *= b.x;
    a.y *= b.y;
}

__host__ __device__ constexpr void operator/=(float2& a, const float2& b) {
    a.x /= b.x;
    a.y /= b.y;
}

__host__ __device__ constexpr void operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
}

__host__ __device__ constexpr void operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
}

__host__ __device__ constexpr void operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
}

__host__ __device__ constexpr void operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
}

__host__ __device__ constexpr float2 operator-(const float2& v) {
    return {-v.x, -v.y};
}

__host__ __device__ constexpr float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ constexpr float pow2(const float2& a) {
    return dot(a, a);
}

__host__ __device__ constexpr float length(const float2& v) {
    return sqrtf(pow2(v));
}

__host__ __device__ constexpr float2 normalize(const float2& v) {
    return v * rsqrtf(pow2(v));
}

__host__ __device__ constexpr float2 fract(const float2& v) {
    return {fract(v.x), fract(v.y)};
}

__host__ __device__ constexpr float2 mix(const float2& a, const float2& b, float t) {
    return a + (b - a) * t;
}

// Operators on CUDA float3

__host__ __device__ constexpr float3 make_float3(float x) {
    return {x, x, x};
}

__host__ __device__ constexpr float3 make_float3(const float2& a, float z) {
    return {a.x, a.y, z};
}

__host__ __device__ constexpr float3 make_float3(float x, const float2& b) {
    return {x, b.x, b.y};
}

__host__ __device__ constexpr float3 make_float3(const float4& a) {
    return {a.x, a.y, a.z};
}

__host__ __device__ constexpr float3 operator+(const float3& a, const float3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ constexpr float3 operator-(const float3& a, const float3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ constexpr float3 operator*(const float3& a, const float3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ constexpr float3 operator/(const float3& a, const float3& b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ constexpr float3 operator+(const float3& a, float b) {
    return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ constexpr float3 operator-(const float3& a, float b) {
    return {a.x - b, a.y - b, a.z - b};
}

__host__ __device__ constexpr float3 operator*(const float3& a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ constexpr float3 operator/(const float3& a, float b) {
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ constexpr float3 operator+(float a, const float3& b) {
    return {a + b.x, a + b.y, a + b.z};
}

__host__ __device__ constexpr float3 operator-(float a, const float3& b) {
    return {a - b.x, a - b.y, a - b.z};
}

__host__ __device__ constexpr float3 operator*(float a, const float3& b) {
    return {a * b.x, a * b.y, a * b.z};
}

__host__ __device__ constexpr float3 operator/(float a, const float3& b) {
    return {a / b.x, a / b.y, a / b.z};
}

__host__ __device__ constexpr void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ constexpr void operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__host__ __device__ constexpr void operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__host__ __device__ constexpr void operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__host__ __device__ constexpr void operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

__host__ __device__ constexpr void operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__host__ __device__ constexpr void operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__host__ __device__ constexpr void operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

__host__ __device__ constexpr float3 operator-(const float3& v) {
    return {-v.x, -v.y, -v.z};
}

__host__ __device__ constexpr float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ constexpr float pow2(const float3& a) {
    return dot(a, a);
}

__host__ __device__ constexpr float3 cross(const float3& a, const float3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ constexpr float length(const float3& v) {
    return norm3df(v.x, v.y, v.z);
}

__host__ __device__ constexpr float3 normalize(const float3& v) {
    return v * rnorm3df(v.x, v.y, v.z);
}

__host__ __device__ constexpr float3 fract(const float3& v) {
    return {fract(v.x), fract(v.y), fract(v.z)};
}

__host__ __device__ constexpr float3 mix(const float3& a, const float3& b, float t) {
    return a + (b - a) * t;
}

__host__ __device__ constexpr float3 min(const float3& a, const float3& b) {
    return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

__host__ __device__ constexpr float3 reflect(const float3& i, const float3& n) {
    return 2.0f * dot(n, i) * n - i;
}

__host__ __device__ constexpr float3 refract(const float3& i, const float3& n, const float eta) {
    const auto cosTheta_i  = dot(n, i);
    const auto sin2Theta_i = 1.0f - safesqrt(cosTheta_i);
    const auto sin2Theta_t = sin2Theta_i * rsqrtf(eta);
    const auto cosTheta_t = safesqrt(1.0f - sin2Theta_t); // NOTE: Important to prevent NaNs
    return (cosTheta_i / eta - cosTheta_t) * n - i / eta;
}

__host__ __device__ constexpr bool isfinite(const float3& v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

// Operators on CUDA float4

__host__ __device__ constexpr float4 make_float4(float x) {
    return {x, x, x, x};
}

__host__ __device__ constexpr float4 make_float4(const float2& a, float z, float w) {
    return {a.x, a.y, z, w};
}

__host__ __device__ constexpr float4 make_float4(const float2& a, const float2& b) {
    return {a.x, a.y, b.x, b.y};
}

__host__ __device__ constexpr float4 make_float4(const float3& a, float w) {
    return {a.x, a.y, a.z, w};
}

__host__ __device__ constexpr float4 operator+(const float4& a, const float4& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__host__ __device__ constexpr float4 operator-(const float4& a, const float4& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__host__ __device__ constexpr float4 operator*(const float4& a, const float4& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

__host__ __device__ constexpr float4 operator/(const float4& a, const float4& b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

__host__ __device__ constexpr float4 operator+(const float4& a, float b) {
    return {a.x + b, a.y + b, a.z + b, a.w + b};
}

__host__ __device__ constexpr float4 operator-(const float4& a, float b) {
    return {a.x - b, a.y - b, a.z - b, a.w - b};
}

__host__ __device__ constexpr float4 operator*(const float4& a, float b) {
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}

__host__ __device__ constexpr float4 operator/(const float4& a, float b) {
    return {a.x / b, a.y / b, a.z / b, a.w / b};
}

__host__ __device__ constexpr float4 operator+(float a, const float4& b) {
    return {a + b.x, a + b.y, a + b.z, a + b.w};
}

__host__ __device__ constexpr float4 operator-(float a, const float4& b) {
    return {a - b.x, a - b.y, a - b.z, a - b.w};
}

__host__ __device__ constexpr float4 operator*(float a, const float4& b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

__host__ __device__ constexpr float4 operator/(float a, const float4& b) {
    return {a / b.x, a / b.y, a / b.z, a / b.w};
}

__host__ __device__ constexpr void operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__host__ __device__ constexpr void operator-=(float4& a, const float4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

__host__ __device__ constexpr void operator*=(float4& a, const float4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

__host__ __device__ constexpr void operator/=(float4& a, const float4& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

__host__ __device__ constexpr void operator+=(float4& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__host__ __device__ constexpr void operator-=(float4& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__host__ __device__ constexpr void operator*=(float4& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__host__ __device__ constexpr void operator/=(float4& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

__host__ __device__ constexpr float4 operator-(const float4& v) {
    return {-v.x, -v.y, -v.z, -v.w};
}

__host__ __device__ constexpr float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ constexpr float pow2(const float4& a) {
    return dot(a, a);
}

__host__ __device__ constexpr float length(const float4& v) {
    return norm4df(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr float4 normalize(const float4& v) {
    return v * rnorm4df(v.x, v.y, v.z, v.w);
}

__host__ __device__ constexpr float4 fract(const float4& v) {
    return {fract(v.x), fract(v.y), fract(v.z), fract(v.w)};
}

__host__ __device__ constexpr float4 mix(const float4& a, const float4& b, float t) {
    return a + (b - a) * t;
}

// Operators on CUDA int2

__host__ __device__ constexpr int2 make_int2(int x) {
    return {x, x};
}

__host__ __device__ constexpr int2 make_int2(const int3& a) {
    return {a.x, a.y};
}

__host__ __device__ constexpr int2 make_int2(const int4& a) {
    return {a.x, a.y};
}

__host__ __device__ constexpr int2 operator+(const int2& a, const int2& b) {
    return {a.x + b.x, a.y + b.y};
}

__host__ __device__ constexpr int2 operator-(const int2& a, const int2& b) {
    return {a.x - b.x, a.y - b.y};
}

__host__ __device__ constexpr int2 operator*(const int2& a, const int2& b) {
    return {a.x * b.x, a.y * b.y};
}

__host__ __device__ constexpr int2 operator/(const int2& a, const int2& b) {
    return {a.x / b.x, a.y / b.y};
}

__host__ __device__ constexpr int2 operator+(const int2& a, int b) {
    return {a.x + b, a.y + b};
}

__host__ __device__ constexpr int2 operator-(const int2& a, int b) {
    return {a.x - b, a.y - b};
}

__host__ __device__ constexpr int2 operator*(const int2& a, int b) {
    return {a.x * b, a.y * b};
}

__host__ __device__ constexpr int2 operator/(const int2& a, int b) {
    return {a.x / b, a.y / b};
}

__host__ __device__ constexpr int2 operator+(int a, const int2& b) {
    return {a + b.x, a + b.y};
}

__host__ __device__ constexpr int2 operator-(int a, const int2& b) {
    return {a - b.x, a - b.y};
}

__host__ __device__ constexpr int2 operator*(int a, const int2& b) {
    return {a * b.x, a * b.y};
}

__host__ __device__ constexpr int2 operator/(int a, const int2& b) {
    return {a / b.x, a / b.y};
}

__host__ __device__ constexpr void operator+=(int2& a, const int2& b) {
    a.x += b.x;
    a.y += b.y;
}

__host__ __device__ constexpr void operator-=(int2& a, const int2& b) {
    a.x -= b.x;
    a.y -= b.y;
}

__host__ __device__ constexpr void operator*=(int2& a, const int2& b) {
    a.x *= b.x;
    a.y *= b.y;
}

__host__ __device__ constexpr void operator/=(int2& a, const int2& b) {
    a.x /= b.x;
    a.y /= b.y;
}

__host__ __device__ constexpr void operator+=(int2& a, int b) {
    a.x += b;
    a.y += b;
}

__host__ __device__ constexpr void operator-=(int2& a, int b) {
    a.x -= b;
    a.y -= b;
}

__host__ __device__ constexpr void operator*=(int2& a, int b) {
    a.x *= b;
    a.y *= b;
}

__host__ __device__ constexpr void operator/=(int2& a, int b) {
    a.x /= b;
    a.y /= b;
}

__host__ __device__ constexpr int2 operator-(const int2& v) {
    return {-v.x, -v.y};
}

__host__ __device__ constexpr int dot(const int2& a, const int2& b) {
    return a.x * b.x + a.y * b.y;
}

// Operators on CUDA int3

__host__ __device__ constexpr int3 make_int3(int x) {
    return {x, x, x};
}

__host__ __device__ constexpr int3 make_int3(const int2& a, int z) {
    return {a.x, a.y, z};
}

__host__ __device__ constexpr int3 make_int3(int x, const int2& b) {
    return {x, b.x, b.y};
}

__host__ __device__ constexpr int3 make_int3(const int4& a) {
    return {a.x, a.y, a.z};
}

__host__ __device__ constexpr int3 operator+(const int3& a, const int3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ constexpr int3 operator-(const int3& a, const int3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ constexpr int3 operator*(const int3& a, const int3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ constexpr int3 operator/(const int3& a, const int3& b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ constexpr int3 operator+(const int3& a, int b) {
    return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ constexpr int3 operator-(const int3& a, int b) {
    return {a.x - b, a.y - b, a.z - b};
}

__host__ __device__ constexpr int3 operator*(const int3& a, int b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ constexpr int3 operator/(const int3& a, int b) {
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ constexpr int3 operator+(int a, const int3& b) {
    return {a + b.x, a + b.y, a + b.z};
}

__host__ __device__ constexpr int3 operator-(int a, const int3& b) {
    return {a - b.x, a - b.y, a - b.z};
}

__host__ __device__ constexpr int3 operator*(int a, const int3& b) {
    return {a * b.x, a * b.y, a * b.z};
}

__host__ __device__ constexpr int3 operator/(int a, const int3& b) {
    return {a / b.x, a / b.y, a / b.z};
}

__host__ __device__ constexpr void operator+=(int3& a, const int3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ constexpr void operator-=(int3& a, const int3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__host__ __device__ constexpr void operator*=(int3& a, const int3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__host__ __device__ constexpr void operator/=(int3& a, const int3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__host__ __device__ constexpr int3 operator-(const int3& v) {
    return {-v.x, -v.y, -v.z};
}

__host__ __device__ constexpr int dot(const int3& a, const int3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ constexpr int3 cross(const int3& a, const int3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Operators on CUDA int4

__host__ __device__ constexpr int4 make_int4(int x) {
    return {x, x, x, x};
}

__host__ __device__ constexpr int4 make_int4(const int2& a, int z, int w) {
    return {a.x, a.y, z, w};
}

__host__ __device__ constexpr int4 make_int4(const int2& a, const int2& b) {
    return {a.x, a.y, b.x, b.y};
}

__host__ __device__ constexpr int4 make_int4(const int3& a, int w) {
    return {a.x, a.y, a.z, w};
}

__host__ __device__ constexpr int4 operator+(const int4& a, const int4& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__host__ __device__ constexpr int4 operator-(const int4& a, const int4& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__host__ __device__ constexpr int4 operator*(const int4& a, const int4& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

__host__ __device__ constexpr int4 operator/(const int4& a, const int4& b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

__host__ __device__ constexpr int4 operator+(const int4& a, int b) {
    return {a.x + b, a.y + b, a.z + b, a.w + b};
}

__host__ __device__ constexpr int4 operator-(const int4& a, int b) {
    return {a.x - b, a.y - b, a.z - b, a.w - b};
}

__host__ __device__ constexpr int4 operator*(const int4& a, int b) {
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}

__host__ __device__ constexpr int4 operator/(const int4& a, int b) {
    return {a.x / b, a.y / b, a.z / b, a.w / b};
}

__host__ __device__ constexpr int4 operator+(int a, const int4& b) {
    return {a + b.x, a + b.y, a + b.z, a + b.w};
}

__host__ __device__ constexpr int4 operator-(int a, const int4& b) {
    return {a - b.x, a - b.y, a - b.z, a - b.w};
}

__host__ __device__ constexpr int4 operator*(int a, const int4& b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

__host__ __device__ constexpr int4 operator/(int a, const int4& b) {
    return {a / b.x, a / b.y, a / b.z, a / b.w};
}

__host__ __device__ constexpr int dot(const int4& a, const int4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Operators on CUDA uint2

__host__ __device__ constexpr uint2 make_uint2(uint x) {
    return {x, x};
}

__host__ __device__ constexpr uint2 make_uint2(const uint3& a) {
    return {a.x, a.y};
}

__host__ __device__ constexpr uint2 make_uint2(const uint4& a) {
    return {a.x, a.y};
}

__host__ __device__ constexpr uint2 operator+(const uint2& a, const uint2& b) {
    return {a.x + b.x, a.y + b.y};
}

__host__ __device__ constexpr uint2 operator-(const uint2& a, const uint2& b) {
    return {a.x - b.x, a.y - b.y};
}

__host__ __device__ constexpr uint2 operator*(const uint2& a, const uint2& b) {
    return {a.x * b.x, a.y * b.y};
}

__host__ __device__ constexpr uint2 operator/(const uint2& a, const uint2& b) {
    return {a.x / b.x, a.y / b.y};
}

__host__ __device__ constexpr uint2 operator+(const uint2& a, uint b) {
    return {a.x + b, a.y + b};
}

__host__ __device__ constexpr uint2 operator-(const uint2& a, uint b) {
    return {a.x - b, a.y - b};
}

__host__ __device__ constexpr uint2 operator*(const uint2& a, uint b) {
    return {a.x * b, a.y * b};
}

__host__ __device__ constexpr uint2 operator/(const uint2& a, uint b) {
    return {a.x / b, a.y / b};
}

__host__ __device__ constexpr uint2 operator+(uint a, const uint2& b) {
    return {a + b.x, a + b.y};
}

__host__ __device__ constexpr uint2 operator-(uint a, const uint2& b) {
    return {a - b.x, a - b.y};
}

__host__ __device__ constexpr uint2 operator*(uint a, const uint2& b) {
    return {a * b.x, a * b.y};
}

__host__ __device__ constexpr uint2 operator/(uint a, const uint2& b) {
    return {a / b.x, a / b.y};
}

__host__ __device__ constexpr void operator+=(uint2& a, const uint2& b) {
    a.x += b.x;
    a.y += b.y;
}

__host__ __device__ constexpr void operator-=(uint2& a, const uint2& b) {
    a.x -= b.x;
    a.y -= b.y;
}

__host__ __device__ constexpr void operator*=(uint2& a, const uint2& b) {
    a.x *= b.x;
    a.y *= b.y;
}

__host__ __device__ constexpr void operator/=(uint2& a, const uint2& b) {
    a.x /= b.x;
    a.y /= b.y;
}

__host__ __device__ constexpr void operator+=(uint2& a, uint b) {
    a.x += b;
    a.y += b;
}

__host__ __device__ constexpr void operator-=(uint2& a, uint b) {
    a.x -= b;
    a.y -= b;
}

__host__ __device__ constexpr void operator*=(uint2& a, uint b) {
    a.x *= b;
    a.y *= b;
}

__host__ __device__ constexpr void operator/=(uint2& a, uint b) {
    a.x /= b;
    a.y /= b;
}

__host__ __device__ constexpr uint dot(const uint2& a, const uint2& b) {
    return a.x * b.x + a.y * b.y;
}

// Operators on CUDA uint3

__host__ __device__ constexpr uint3 make_uint3(uint x) {
    return {x, x, x};
}

__host__ __device__ constexpr uint3 make_uint3(const uint2& a, uint z) {
    return {a.x, a.y, z};
}

__host__ __device__ constexpr uint3 make_uint3(uint x, const uint2& b) {
    return {x, b.x, b.y};
}

__host__ __device__ constexpr uint3 make_uint3(const uint4& a) {
    return {a.x, a.y, a.z};
}

__host__ __device__ constexpr uint3 operator+(const uint3& a, const uint3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ constexpr uint3 operator-(const uint3& a, const uint3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ constexpr uint3 operator*(const uint3& a, const uint3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ constexpr uint3 operator/(const uint3& a, const uint3& b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ constexpr uint3 operator+(const uint3& a, uint b) {
    return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ constexpr uint3 operator-(const uint3& a, uint b) {
    return {a.x - b, a.y - b, a.z - b};
}

__host__ __device__ constexpr uint3 operator*(const uint3& a, uint b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ constexpr uint3 operator/(const uint3& a, uint b) {
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ constexpr uint3 operator+(uint a, const uint3& b) {
    return {a + b.x, a + b.y, a + b.z};
}

__host__ __device__ constexpr uint3 operator-(uint a, const uint3& b) {
    return {a - b.x, a - b.y, a - b.z};
}

__host__ __device__ constexpr uint3 operator*(uint a, const uint3& b) {
    return {a * b.x, a * b.y, a * b.z};
}

__host__ __device__ constexpr uint3 operator/(uint a, const uint3& b) {
    return {a / b.x, a / b.y, a / b.z};
}

__host__ __device__ constexpr void operator+=(uint3& a, const uint3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ constexpr void operator-=(uint3& a, const uint3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__host__ __device__ constexpr void operator*=(uint3& a, const uint3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__host__ __device__ constexpr void operator/=(uint3& a, const uint3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__host__ __device__ constexpr void operator+=(uint3& a, uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

__host__ __device__ constexpr void operator-=(uint3& a, uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__host__ __device__ constexpr void operator*=(uint3& a, uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__host__ __device__ constexpr void operator/=(uint3& a, uint b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

__host__ __device__ constexpr uint dot(const uint3& a, const uint3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Operators on CUDA uint4

__host__ __device__ constexpr uint4 make_uint4(uint x) {
    return {x, x, x, x};
}

__host__ __device__ constexpr uint4 make_uint4(const uint2& a, uint z, uint w) {
    return {a.x, a.y, z, w};
}

__host__ __device__ constexpr uint4 make_uint4(const uint2& a, const uint2& b) {
    return {a.x, a.y, b.x, b.y};
}

__host__ __device__ constexpr uint4 make_uint4(const uint3& a, uint w) {
    return {a.x, a.y, a.z, w};
}

__host__ __device__ constexpr uint4 operator+(const uint4& a, const uint4& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__host__ __device__ constexpr uint4 operator-(const uint4& a, const uint4& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__host__ __device__ constexpr uint4 operator*(const uint4& a, const uint4& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

__host__ __device__ constexpr uint4 operator/(const uint4& a, const uint4& b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

__host__ __device__ constexpr uint4 operator+(const uint4& a, uint b) {
    return {a.x + b, a.y + b, a.z + b, a.w + b};
}

__host__ __device__ constexpr uint4 operator-(const uint4& a, uint b) {
    return {a.x - b, a.y - b, a.z - b, a.w - b};
}

__host__ __device__ constexpr uint4 operator*(const uint4& a, uint b) {
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}

__host__ __device__ constexpr uint4 operator/(const uint4& a, uint b) {
    return {a.x / b, a.y / b, a.z / b, a.w / b};
}

__host__ __device__ constexpr uint4 operator+(uint a, const uint4& b) {
    return {a + b.x, a + b.y, a + b.z, a + b.w};
}

__host__ __device__ constexpr uint4 operator-(uint a, const uint4& b) {
    return {a - b.x, a - b.y, a - b.z, a - b.w};
}

__host__ __device__ constexpr uint4 operator*(uint a, const uint4& b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

__host__ __device__ constexpr uint4 operator/(uint a, const uint4& b) {
    return {a / b.x, a / b.y, a / b.z, a / b.w};
}

__host__ __device__ constexpr void operator+=(uint4& a, const uint4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__host__ __device__ constexpr void operator-=(uint4& a, const uint4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

__host__ __device__ constexpr void operator*=(uint4& a, const uint4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

__host__ __device__ constexpr void operator/=(uint4& a, const uint4& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

__host__ __device__ constexpr void operator+=(uint4& a, uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__host__ __device__ constexpr void operator-=(uint4& a, uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__host__ __device__ constexpr void operator*=(uint4& a, uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__host__ __device__ constexpr void operator/=(uint4& a, uint b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

__host__ __device__ constexpr uint dot(const uint4& a, const uint4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// float2x2

__host__ __device__ constexpr float2 operator*(const float2x2& m, const float2& v) {
    return make_float2(
        m[0].x * v.x + m[1].x * v.y,
        m[0].y * v.x + m[1].y * v.y
    );
}

// float3x3

__host__ __device__ constexpr float3 operator*(const float3x3& m, const float3& v) {
    return make_float3(
        m[0].x * v.x + m[1].x * v.y + m[2].x * v.z,
        m[0].y * v.x + m[1].y * v.y + m[2].y * v.z,
        m[0].z * v.x + m[1].z * v.y + m[2].z * v.z
    );
}

// float4x4

__host__ __device__ constexpr float4 operator*(const float4x4& m, const float4& v) {
    return make_float4(
        m[0].x * v.x + m[1].x * v.y + m[2].x * v.z + m[3].x * v.w,
        m[0].y * v.x + m[1].y * v.y + m[2].y * v.z + m[3].y * v.w,
        m[0].z * v.x + m[1].z * v.y + m[2].z * v.z + m[3].z * v.w,
        m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w * v.w
    );
}

/**
 * Reorthogonalizes a tangent space using the Gram-Schmidt process and returns an orthonormal tangent space matrix
 * Note: n needs to be normalized and t must be linearly independent from n
 */
__host__ __device__ constexpr float3x3 buildTBN(const float3& n, const float3& t) {
    const auto nt = normalize(t - dot(t, n) * n);
    const auto b = cross(n, nt);
    return make_float3x3(nt, b, n);
}

/**
 * Builds an orthogonal tangent space to world space matrix from a normalized normal
 */
__host__ __device__ constexpr float3x3 buildTBN(const float3& n) {
    if (n.x == 0.0f) {
        return buildTBN(n, {1.0f, 0.0f, 0.0f});
    } else {
        return buildTBN(n, {0.0f, 1.0f, 0.0f});
    }
}