# General Wheel C++

A friendly modern C++ toolkit that makes your life easier with thread-safe collections and useful utilities. 很方便！

## Features

### Thread-safe Collections

- **`concurrent_unordered_map`**: Like a regular map but worry-free in multi-threaded code
  ```cpp
  concurrent_unordered_map<int, std::string> map;
  map.insert({1, "one"});  // Thread-safe insertion
  auto value = map.at(1);  // Thread-safe lookup
  ```

- **`concurrent_vector`**: A vector you can safely use across threads
  ```cpp
  concurrent_vector<int> vec;
  vec.push_back(42);  // Multiple threads can push_back safely
  ```

- **`concurrent_hashset`**: A set that handles concurrent access gracefully

### Linear Algebra Boost

- **SIMD-optimized vector operations for performance-critical code**
- **Platform-specific optimizations (NEON for ARM64)**
- **Fast vector operations**

> Note: Enable the optional assembly/SIMD path by configuring CMake with `-DLINALG_USE_ASM=ON` to use the current highest-performance implementation.

#### Performance Benchmarks (LINALG_USE_ASM)

The following micro-benchmarks were run on a macOS machine (single process, warm cache) using 5 (or 3 where noted) epochs per configuration. Times are in microseconds (µs) and represent the reported average for the optimized SIMD / batch implementation vs. a straightforward scalar baseline. Speedup = (scalar avg / optimized avg). Real-world gains depend on compiler flags, CPU, and memory layout; these numbers illustrate typical ratios.

##### Dot Product

| Vector Size | Optimized Avg (µs) | Scalar Avg (µs) | Speedup |
| ----------- | ------------------ | --------------- | ------- |
| 1,000       | 0.28               | 4.83            | 17.24x  |
| 10,000      | 3.12               | 35.82           | 11.47x  |
| 100,000     | 33.04              | 362.07          | 10.96x  |
| 1,000,000   | 343.24             | 3704.71         | 10.79x  |

##### Cosine Similarity

| Vector Size | Optimized Avg (µs) | Scalar Avg (µs) | Speedup |
| ----------- | ------------------ | --------------- | ------- |
| 1,000       | 0.32               | 3.56            | 11.12x  |
| 10,000      | 3.22               | 37.09           | 11.51x  |
| 100,000     | 33.55              | 366.77          | 10.93x  |
| 1,000,000   | 348.52             | 3651.85         | 10.48x  |

##### Batch Cosine Similarity

| Vector Size | Batch Size | Batch Avg (µs) | Multiple Single Calls Avg (µs) | Speedup |
| ----------- | ---------- | -------------- | ------------------------------ | ------- |
| 1,000       | 10         | 2.06           | 3.42                           | 1.66x   |
| 1,000       | 50         | 9.14           | 16.81                          | 1.84x   |
| 1,000       | 100        | 16.81          | 32.96                          | 1.96x   |
| 10,000      | 10         | 20.28          | 32.98                          | 1.63x   |
| 10,000      | 50         | 85.36          | 168.40                         | 1.97x   |
| 10,000      | 100        | 175.69         | 341.24                         | 1.94x   |

##### Mean Pooling (Averaging N Vectors)

| Vector Size | #Vectors | Optimized Avg (µs) | Scalar Avg (µs) | Speedup |
| ----------- | -------: | ------------------ | --------------- | ------- |
| 1,000       |        2 | 1.30               | 4.02            | 3.10x   |
| 1,000       |       10 | 4.24               | 12.20           | 2.88x   |
| 1,000       |       50 | 19.57              | 54.30           | 2.77x   |
| 10,000      |        2 | 12.52              | 40.72           | 3.25x   |
| 10,000      |       10 | 43.00              | 125.20          | 2.91x   |
| 10,000      |       50 | 189.09             | 544.04          | 2.88x   |
| 100,000     |        2 | 133.68             | 415.26          | 3.11x   |
| 100,000     |       10 | 456.12             | 1280.60         | 2.81x   |
| 100,000     |       50 | 2196.11            | 5701.20         | 2.60x   |

##### Batch Channel Mean Pooling

Computes channel-wise means over (feature_dim × channel_dim) slices per batch element.

| Feature Dim | Channel Dim | Batch Size | Optimized Avg (µs) | Scalar Avg (µs) | Speedup |
| ----------- | ----------- | ---------- | ------------------ | --------------- | ------- |
| 640         | 30          | 20         | 151.30             | 526.10          | 3.48x   |
| 640         | 30          | 80         | 620.33             | 2133.75         | 3.44x   |
| 640         | 80          | 20         | 401.82             | 1359.92         | 3.38x   |
| 640         | 80          | 80         | 1758.10            | 5422.72         | 3.08x   |
| 2560        | 30          | 20         | 632.08             | 2356.10         | 3.73x   |
| 2560        | 30          | 80         | 2533.43            | 8354.40         | 3.30x   |
| 2560        | 80          | 20         | 1615.53            | 5366.45         | 3.32x   |
| 2560        | 80          | 80         | 6540.08            | 21727.98        | 3.32x   |

##### Batch Feature Mean Pooling

Computes feature-wise mean across batch dimension (very cache-friendly in optimized path).

| Feature Dim | Batch Size | Optimized Avg (µs) | Scalar Avg (µs) | Speedup |
| ----------- | ---------- | ------------------ | --------------- | ------- |
| 640         | 20         | 1.55               | 45.42           | 29.30x  |
| 640         | 80         | 6.28               | 182.00          | 28.97x  |
| 640         | 320        | 31.83              | 722.00          | 22.68x  |
| 2560        | 20         | 6.38               | 182.03          | 28.52x  |
| 2560        | 80         | 27.13              | 737.27          | 27.17x  |
| 2560        | 320        | 105.18             | 2946.87         | 28.02x  |
| 10240       | 20         | 26.93              | 739.88          | 27.47x  |
| 10240       | 80         | 103.93             | 2968.17         | 28.56x  |
| 10240       | 320        | 778.93             | 11947.42        | 15.34x  |

> Note: Extremely large feature + batch combinations may become memory-bandwidth bound, reducing relative speedup, as seen in the largest configuration.

##### Summary

- Dot product & cosine similarity: ~10–17x faster than scalar.
- Batch cosine similarity: ~1.6–2.0x over repeated single calls (loop fusion + reduced memory traffic).
- Mean pooling: ~2.6–3.3x.
- Batch channel mean pooling: ~3.0–3.7x.
- Batch feature mean pooling: up to ~29x (excellent SIMD & cache utilization), still >15x in largest case.

Re-run benchmarks locally after changing compiler flags (e.g. `-O3 -march=native`) to see platform-specific gains.

#### Example Usage

```cpp
#include <linalg_boost/linalg_boost.hpp>

// Create test vectors
std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
std::vector<float> vec2 = {5.0f, 6.0f, 7.0f, 8.0f};

float dot1 = wheel::linalg_boost::dot_product(vec1.data(), vec2.data(), vec1.size());
float similarity1 = wheel::linalg_boost::cosine_similarity(vec1.data(), vec2.data(), vec1.size());

float dot2 = wheel::linalg_boost::dot_product(vec1, vec2);
float similarity2 = wheel::linalg_boost::cosine_similarity(vec1, vec2);

// Mean pooling operations
std::vector<std::vector<float>> token_embeddings = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
std::vector<float> mean_embedding = wheel::linalg_boost::mean_pooling(token_embeddings);
// mean_embedding => [3.0f, 4.0f]

// Batch mean pooling
std::vector<std::vector<std::vector<float>>> batch_embeddings = {
  {{1.0f, 2.0f}, {3.0f, 4.0f}},              // First sequence
  {{5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}} // Second sequence
};
std::vector<std::vector<float>> batch_means = wheel::linalg_boost::batch_mean_pooling(batch_embeddings);
// batch_means => [[2.0f, 3.0f], [7.0f, 8.0f]]

// Batch compute optimized version
const size_t batch_size = 100;
const size_t vec_size = 128;

std::vector<const float*> batch_vectors(batch_size);
std::vector<float> results(batch_size);

wheel::linalg_boost::batch_dot_product(batch_vectors.data(), vec2.data(),
                     vec_size, batch_size, results.data());

wheel::linalg_boost::batch_cosine_similarity(batch_vectors.data(), vec2.data(),
                       vec_size, batch_size, results.data());

// Find top-k most similar vectors
std::vector<std::vector<float>> vectors(batch_size, std::vector<float>(vec_size));
auto dot_results = wheel::linalg_boost::batch_dot_product(vectors, vec2);
auto similarity_results = wheel::linalg_boost::batch_cosine_similarity(vectors, vec2);

const size_t k = 5;
std::vector<size_t> indices(k);
std::vector<float> scores(k);
wheel::linalg_boost::top_k_similar(batch_vectors.data(), vec2.data(), vec_size,
                   batch_size, k, indices.data(), scores.data());

auto [top_indices, top_scores] = wheel::linalg_boost::top_k_similar(vectors, vec2, k);
```


### String Utils

- **String cleaning (trim spaces, remove unwanted parts)**
  ```cpp
  std::string_view text = "  1145141919  ";
  auto trimmed = ltrim(rtrim(text));  // "1145141919"
  ```

- **UTF-8 text handling that works with Chinese, Japanese, etc. (支持中文!)**
  ```cpp
  std::string str = "你要玩原神还是玩c加加";
  auto replaced = replace_str(str, "c加加", "rust");  // "你要玩原神还是玩rust"
  ```

- **Split strings**
  ```cpp
  std::string csv = "apple,banana,orange";
  for (auto part : SplitString(csv, ',')) {
      // Iterates through: "apple", "banana", "orange"
  }
  ```

- **UTF-8 string splitting by character count**
  ```cpp
  std::string utf8_text = "刻晴甘雨神里雷电水神仆人";
  for (auto chunk : Utf8Splitter(utf8_text, 2)) {
      // Splits into chunks of 2 characters each, preserving UTF-8
  }
  ```

### Thread Safety Tool `MutexData`

- **Wrap your data to make it thread-safe automatically**
- **Safe locking patterns that prevent deadlocks**
- **Controlled access to shared data without race conditions**
  ```cpp
  struct UserData {
      std::string name;
      int age;
  };

  MutexData<UserData> protected_data;

  // Thread-safe data access
  protected_data.modify([](UserData& data) {
      data.name = "丁真";
      data.age = 233;
  });

  // Read data safely
  auto name = protected_data.read([](const UserData& data) {
      return data.name;  // Returns "丁真"
  });
  ```

## Requirements

- C++17 or later
- CMake 3.14 or later

## Building & Installation

### Building

```bash
git clone https://github.com/FredYakumo/general-wheel-cpp.git
cd general-wheel-cpp
mkdir build && cd build
cmake -G "Ninja" ..
cmake --build . --parallel
```

### Installation

```bash
sudo cmake --install .
```

This will install:
- Library files to `lib/general-wheel-cpp/`
- Header files to `include/general-wheel-cpp/`
- CMake configuration to `lib/cmake/general-wheel-cpp/`

### Integration with Your Project

Add to your CMake project:

```cmake
find_package(general-wheel-cpp REQUIRED)
target_link_libraries(your_target general-wheel-cpp::general-wheel-cpp)
```

## License

[MIT License](LICENSE)
