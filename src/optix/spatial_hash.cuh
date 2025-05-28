#pragma once

#include <cuco/static_multimap.cuh>
#include <thrust/device_vector.h>

template <typename T>
struct SpatialHash {
    // Key: Cell index
    using Key = unsigned int;
    constexpr static Key EMPTY_KEY = 0xFFFFFFFF;
    constexpr static unsigned int MAX_GRID_SIZE = std::cbrt(1 << 32 - 1);

    // Value: Index of the object
    using Value = unsigned int;
    constexpr static Value EMPTY_VALUE = 0xFFFFFFFF;

    float3 min;
    float3 max;
    float max_radius;
    cuco::static_multimap<Key, Value> map;
    thrust::device_vector<T> values;

    struct DeviceView {
        float3 min;
        float3 max;
        float cellSize; // = 1 / cellRes;
        float cellRes; // = 1 / cellSize;
        cuco::static_multimap<Key, Value> map;
        thrust::device_vector<T> values;

        __device__ __forceinline__ void insert(const Key& key, const Value& value) {
            map.insert(key, value);
        }

        __device__ __forceinline__ Key toKey(const char3& cell) const {
            return cell.x + cell.y * MAX_GRID_SIZE + cell.z * MAX_GRID_SIZE * MAX_GRID_SIZE;
        }

        __device__ __forceinline__ char3 toCell(const float3& position) const {
            const auto cell = (position - min) * cellRes; // = (position - min) / cellSize;
            return {static_cast<char>(cell.x), static_cast<char>(cell.y), static_cast<char>(cell.z)};
        }

        __device__ __forceinline__ constexpr float clampToSegment(const float value, const float start, const float length) const {
            return clamp(value, start, start + length);
        }

        __device__ __forceinline__ void search(const float3& position, const float radius) const {
            const auto minCell = toCell(position - radius);
            const auto maxCell = toCell(position + radius);
            char3 currentCell;
            float3 closestPointInCell; // TODO: Compute pow2(closestPoint - position) directly to avoid extra subtraction and multiplication
            for (currentCell.x = minCell.x; currentCell.x <= maxCell.x; currentCell.x++) {
                closestPointInCell.x = clampToSegment(position.x, min.x + currentCell.x * cellSize, cellSize);
                for (currentCell.y = minCell.y; currentCell.y <= maxCell.y; currentCell.y++) {
                    closestPointInCell.y = clampToSegment(position.y, min.y + currentCell.y * cellSize, cellSize);
                    for (currentCell.z = minCell.z; currentCell.z <= maxCell.z; currentCell.z++) {
                        closestPointInCell.z = clampToSegment(position.z, min.z + currentCell.z * cellSize, cellSize);
                        
                        float dist2 = length2(closestPointInCell - position);
                        if (dist2 > radius * radius)
                            continue; // Discard cells outside the radius
                        
                        Key k = toKey(currentCell);
                        // TODO: Search
                    }
                }
            }
        }
    };

};