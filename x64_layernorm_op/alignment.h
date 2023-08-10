#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

constexpr size_t gAlignment = 64;

// Get aligment size for bytes.
inline size_t GetAlignSize(size_t n) { return (n + gAlignment - 1) / gAlignment * gAlignment; }

template <typename T>
inline size_t GetAlignSize(size_t n) {
  constexpr int kDataAlignment = gAlignment / sizeof(T);

  return (n + kDataAlignment - 1) / kDataAlignment * kDataAlignment;
}

inline void* AlignedMalloc(size_t alignment, size_t nbytes) {
  void* data = nullptr;
#if defined(_MSC_VER)
  data = _aligned_malloc(nbytes, alignment);
#else
  int ret = posix_memalign(&data, alignment, nbytes);
  if (ret != 0) {
    // throw error!
  }
#endif
  return data;
}

inline void AlignedFree(void* data) {
#ifdef _MSC_VER
  _aligned_free(data);
#else
  free(data);
#endif
  data = nullptr;
}
