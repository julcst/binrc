#pragma once

#include <cuda_runtime.h>

using uint = unsigned int;

constexpr float PI = 3.14159265358979323846f;
constexpr float INV_PI = 0.3183098861837906715377f;
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

__host__ __device__ constexpr float minf(float a, float b) {
    return a < b ? a : b;
}

__host__ __device__ constexpr float maxf(float a, float b) {
    return a > b ? a : b;
}

__host__ __device__ constexpr float clamp(float x, float a, float b) {
    return maxf(a, minf(b, x));
}

__host__ __device__ constexpr float safesqrt(float x) {
    return sqrtf(maxf(x, 0.0f));
}

__host__ __device__ constexpr float safediv(float a, float b, float fallback = 0.0f) {
    const auto res = a / b;
    return std::isfinite(res) ? res : fallback;
}

__host__ __device__ constexpr float sign(float x) {
    return x < 0.0f ? -1.0f : 1.0f;
}

__host__ __device__ constexpr float step(float x) {
    return maxf(0.0f, x);
}

// Matrices

struct float2x2 {
    float2 m[2] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
    };
    __host__ __device__ constexpr float2& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float2& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float2x2 make_float2x2(const float2& a, const float2& b) {
    return {a, b};
}

struct float3x3 {
    float3 m[3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    };
    __host__ __device__ constexpr float3& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float3& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float3x3 make_float3x3(const float3& a, const float3& b, const float3& c) {
    return {a, b, c};
}

struct float4x4 {
    float4 m[4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f},
    };
    __host__ __device__ constexpr float4& operator[](uint i) { return m[i]; }
    __host__ __device__ constexpr const float4& operator[](uint i) const { return m[i]; }
};

__host__ __device__ constexpr float4x4 make_float4x4(const float4& a, const float4& b, const float4& c, const float4& d) {
    return {a, b, c, d};
}

// float2x2

__host__ __device__ constexpr float2 operator*(const float2x2& m, const float2& v) {
    return {
        m[0].x * v.x + m[1].x * v.y,
        m[0].y * v.x + m[1].y * v.y
    };
}

// float3x3

__host__ __device__ constexpr float3 operator*(const float3x3& m, const float3& v) {
    return {
        m[0].x * v.x + m[1].x * v.y + m[2].x * v.z,
        m[0].y * v.x + m[1].y * v.y + m[2].y * v.z,
        m[0].z * v.x + m[1].z * v.y + m[2].z * v.z
    };
}

// float4x4

__host__ __device__ constexpr float4 operator*(const float4x4& m, const float4& v) {
    return {
        m[0].x * v.x + m[1].x * v.y + m[2].x * v.z + m[3].x * v.w,
        m[0].y * v.x + m[1].y * v.y + m[2].y * v.z + m[3].y * v.w,
        m[0].z * v.x + m[1].z * v.y + m[2].z * v.z + m[3].z * v.w,
        m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w * v.w
    };
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
    //return v * rsqrtf(pow2(v));
    return v / length(v);
}

__host__ __device__ constexpr float2 fract(const float2& v) {
    return {fract(v.x), fract(v.y)};
}

__host__ __device__ constexpr float2 mix(const float2& a, const float2& b, float t) {
    return a + (b - a) * t;
}

__host__ __device__ constexpr bool isfinite(const float2& v) {
    return std::isfinite(v.x) && std::isfinite(v.y);
}

__host__ __device__ constexpr float2 sign(const float2& v) {
    return {sign(v.x), sign(v.y)};
}

__host__ __device__ constexpr float2 abs(const float2& v) {
    return {abs(v.x), abs(v.y)};
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
    //return norm3df(v.x, v.y, v.z);
    return sqrtf(dot(v, v));
}

__host__ __device__ constexpr float3 normalize(const float3& v) {
    //return v * rnorm3df(v.x, v.y, v.z);
    return v / length(v);
}

__host__ __device__ constexpr float3 fract(const float3& v) {
    return {fract(v.x), fract(v.y), fract(v.z)};
}

__host__ __device__ constexpr float3 mix(const float3& a, const float3& b, float t) {
    return a + (b - a) * t;
}

__host__ __device__ constexpr float3 min(const float3& a, const float3& b) {
    return {minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z)};
}

__host__ __device__ constexpr float3 min(const float3& a, float b) {
    return {minf(a.x, b), minf(a.y, b), minf(a.z, b)};
}

__host__ __device__ constexpr float3 min(float a, const float3& b) {
    return {minf(a, b.x), minf(a, b.y), minf(a, b.z)};
}

__host__ __device__ constexpr float3 max(const float3& a, const float3& b) {
    return {maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z)};
}

__host__ __device__ constexpr float3 max(const float3& a, float b) {
    return {maxf(a.x, b), maxf(a.y, b), maxf(a.z, b)};
}

__host__ __device__ constexpr float3 max(float a, const float3& b) {
    return {maxf(a, b.x), maxf(a, b.y), maxf(a, b.z)};
}

__host__ __device__ constexpr float3 reflect(const float3& i, const float3& n) {
    return 2.0f * dot(n, i) * n - i;
}

__host__ __device__ constexpr float3 refract(const float3& i, const float3& n, const float eta) {
    const auto cosTheta_i  = dot(n, i);
    const auto sin2Theta_i = 1.0f - safesqrt(cosTheta_i);
    //const auto sin2Theta_t = sin2Theta_i * rsqrtf(eta);
    const auto sin2Theta_t = sin2Theta_i / eta;
    const auto cosTheta_t = safesqrt(1.0f - sin2Theta_t); // NOTE: Important to prevent NaNs
    return (cosTheta_i / eta - cosTheta_t) * n - i / eta;
}

__host__ __device__ constexpr bool isfinite(const float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

__host__ __device__ constexpr bool isnegative(const float3& v) {
    return v.x < 0.0f && v.y < 0.0f && v.z < 0.0f;
}

__host__ __device__ constexpr float luminance(const float3& linearRGB) {
    return dot({0.2126f, 0.7152f, 0.0722f}, linearRGB);
}

__host__ __device__ constexpr float3 safediv(float3 a, float3 b, float fallback = 0.0f) {
    const auto res = a / b;
    return {
        std::isfinite(res.x) ? res.x : fallback,
        std::isfinite(res.y) ? res.y : fallback,
        std::isfinite(res.z) ? res.z : fallback
    };
}

__host__ __device__ constexpr float3 safediv(float3 a, float b, float fallback = 0.0f) {
    const auto res = a / b;
    return {
        std::isfinite(res.x) ? res.x : fallback,
        std::isfinite(res.y) ? res.y : fallback,
        std::isfinite(res.z) ? res.z : fallback
    };
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
    //return norm4df(v.x, v.y, v.z, v.w);
    return sqrtf(dot(v, v));
}

__host__ __device__ constexpr float4 normalize(const float4& v) {
    //return v * rnorm4df(v.x, v.y, v.z, v.w);
    return v / length(v);
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
 * Builds an orthonormal basis from a normalized normal
 * See: https://graphics.pixar.com/library/OrthonormalB/paper.pdf
 * "Building an Orthonormal Basis, Revisited"
 */
__host__ __device__ constexpr float3x3 buildTBN(const float3& n) {
    float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    const float3 b1 = {1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
    const float3 b2 = {b, sign + n.y * n.y * a, -n.y};
    return make_float3x3(b1, b2, n);
}

__host__ __device__ constexpr float2 toNormSpherical(const float3& n) {
    return {atan2f(n.y, n.x) * INV_PI * 0.5f + 0.5f, acosf(n.z) * INV_PI};
}

__host__ __device__ constexpr float2 toNormCylindrical(const float3& n) {
    return {atan2f(n.y, n.x) * INV_PI * 0.5f + 0.5f, n.z * 0.5f + 0.5f};
}

/**
 * Fast float3 to octahedral mapping from "A Survey of Efficient Representations for Independent Unit Vectors"
 * @param n The normalized unit vector to be mapped
 * @return The octahedral mapping of the input vector in the range [-1, 1]
 */
__host__ __device__ constexpr float2 toOct(const float3& n) {
    // Project the sphere onto the octahedron, and then onto the xy-plane
    const auto p = make_float2(n) * (1.0f / (abs(n.x) + abs(n.y) + abs(n.z)));

    // Reflect the folds of the lower hemisphere over the diagonals
    return (n.z <= 0.0f) ? ((1.0f - abs({p.y, p.x})) * sign(p)) : p;
}

__host__ __device__ constexpr float2 toNormOct(const float3& n) {
    return toOct(n) * 0.5f + 0.5f;
}