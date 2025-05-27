#pragma once

#include <cuco/static_multimap.cuh>

template <typename T>
struct SpatialHash {
    // Key: Cell index, 0xFFFFFFFF for empty cells
    // Value: Pointer to the object, 0 for empty objects
    cuco::static_multimap<unsigned int, unsigned int> map;
};