# General Wheel C++

A friendly modern C++ toolkit that makes your life easier with thread-safe collections and useful utilities. 很方便！

## Features

- **Thread-safe Collections**:
  - `concurrent_unordered_map`: Like a regular map but worry-free in multi-threaded code
  - `concurrent_vector`: A vector you can safely use across threads
  - `concurrent_hashset`: A set that handles concurrent access gracefully

- **String Utils**:
  - String cleaning (trim spaces, remove unwanted parts)
  - UTF-8 text handling that works Chinese, Japanese, etc. (支持中文!)
  - Split strings
  - Join strings with sep char

- **Thread Safety Too`MutexData`: Wrap your data to make it thread-safe automatically
  - Safe locking patterns that prevent deadlocks
  - Controlled access to shared data without race conditions

## Requirements

- C++17 or later
- CMake 3.14 or later

# How to use

## Building

```bash
git clone https://github.com/FredYakumo/general-wheel-cpp.git
cd general-wheel-cpp
mkdir build && cd build
cmake -G "Ninja" ..
cmake --build . --parallel
```

## Installation

```bash
sudo cmake --install .
```

This will install:
- Library files to `lib/general-wheel-cpp/`
- Header files to `include/general-wheel-cpp/`
- CMake configuration to `lib/cmake/general-wheel-cpp/`

## Usage

Add to your CMake project:

```cmake
find_package(general-wheel-cpp REQUIRED)
target_link_libraries(your_target general-wheel-cpp::general-wheel-cpp)
```

### Code Examples:

```cpp
using namespace wheel;

#include <collection/concurrent_unordered_map.hpp>
#include <collection/concurrent_vector.hpp>
#include <string_utils.hpp>
#include <mutex_data.hpp>

concurrent_unordered_map<int, std::string> map;
map.insert({1, "one"});  // Thread-safe insertion
auto value = map.at(1);  // Thread-safe lookup

concurrent_vector<int> vec;
vec.push_back(42);  // Multiple threads can push_back safely

// String utilities

// UTF-8 string manipulation
std::string_view text = "  1145141919  ";
auto trimmed = ltrim(rtrim(text));  // "1145141919"

// 替换文本 (Replace text)
std::string str = "你要玩原神还是玩c加加";
auto replaced = replace_str(str, "c加加", "rust");  // "你要玩原神还是玩rust"

// String splitting with iterators
std::string csv = "apple,banana,orange";
for (auto part : SplitString(csv, ',')) {
    // Iterates through: "apple", "banana", "orange"
}

// UTF-8 string splitting by character count
std::string utf8_text = "刻晴甘雨神里雷电水神仆人";
for (auto chunk : Utf8Splitter(utf8_text, 2)) {
    // Splits into chunks of 2 characters each, preserving UTF-8
}

// Thread-safe data wrapper
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

## License

[MIT License](LICENSE)
