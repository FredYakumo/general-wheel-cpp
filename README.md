# General Wheel C++

A modern C++ utility library providing thread-safe collections and other general-purpose components.

## Features

- Thread-safe collections:
  - Concurrent unordered map
  - Concurrent vector 

- String utilities:
  - UTF-8 string manipulation
  - String trimming (ltrim, rtrim)
  - String replacement with UTF-8 support
  - String joining with custom delimiters
  - String splitting with iterators
  - Text removal between markers
  - UTF-8 string splitting by character count

- Thread safety utilities:
  - Mutex-protected data wrapper
  - RAII-style mutex locking
  - Thread-safe data access patterns

## Requirements

- C++17 or later
- CMake 3.14 or later

## Building

```bash
mkdir build
cd build
cmake -G "Ninja" ..
cmake --build . --parallel
```

## Installation

```bash
sudo cmake --install .
```

This will install:
- Library files to `lib/`
- Header files to `include/`
- CMake configuration to `lib/cmake/general-wheel-cpp/`

## Usage

After installation, you can use this library in your CMake project:

```cmake
find_package(general-wheel-cpp REQUIRED)
target_link_libraries(your_target general-wheel-cpp::general-wheel-cpp)
```

Example usage:

```cpp
using namespace wheel;

#include <collection/concurrent_unordered_map.hpp>
#include <collection/concurrent_vector.hpp>
#include <string_utils.hpp>
#include <mutex_data.hpp>

// Thread-safe collections
concurrent_unordered_map<int, std::string> map;
map.insert({1, "one"});

concurrent_vector<int> vec;
vec.push_back(42);

// String utilities

// UTF-8 string manipulation
std::string_view text = "  Hello World  ";
auto trimmed = ltrim(rtrim(text));  // "Hello World"

// String replacement with UTF-8 support
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
    data.age = 30;
});

// Read data safely
auto name = protected_data.read([](const UserData& data) {
    return data.name;
});
```

## License

[MIT License](LICENSE)