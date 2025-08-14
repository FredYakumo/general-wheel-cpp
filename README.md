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

### Linear Algebra Boost

- **SIMD-optimized vector operations for performance-critical code**
- **Platform-specific optimizations (NEON for ARM64)**
- **Fast vector operations**
  ```cpp
  #include <linalg_boost/linalg_boost.hpp>

  // Create test vectors
  std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> vec2 = {5.0f, 6.0f, 7.0f, 8.0f};

  // Calculate dot product with SIMD optimization
  float dot = wheel::linalg_boost::dot_product(vec1.data(), vec2.data(), vec1.size());

  // Calculate cosine similarity with platform-specific optimizations 
  float similarity = wheel::linalg_boost::cosine_similarity(vec1.data(), vec2.data(), vec1.size());
  
  // Batch processing for multiple vectors against one reference
  const size_t batch_size = 100;
  const size_t vec_size = 128;
  std::vector<const float*> batch_vectors(batch_size);
  std::vector<float> results(batch_size);
  
  // Calculate dot products for all vectors in batch against reference vector
  wheel::linalg_boost::batch_dot_product(batch_vectors.data(), vec2.data(), 
                                         vec_size, batch_size, results.data());
  
  // Calculate similarities for all vectors in batch against reference vector
  wheel::linalg_boost::batch_cosine_similarity(batch_vectors.data(), vec2.data(), 
                                              vec_size, batch_size, results.data());
                                              
  // Find top-k most similar vectors
  const size_t k = 5;
  std::vector<size_t> indices(k);
  std::vector<float> scores(k);
  
  wheel::linalg_boost::top_k_similar(batch_vectors.data(), vec2.data(), vec_size,
                                     batch_size, k, indices.data(), scores.data());
  
  // Now indices contains indices of top-k similar vectors and scores contains their similarity scores
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
