# General Wheel C++

A modern C++ utility library providing thread-safe collections and other general-purpose components.

## Features

- Thread-safe collections:
  - Concurrent unordered map
  - Concurrent vector 

## Requirements

- C++17 or later
- CMake 3.14 or later

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Installation

```bash
make install
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
#include <collection/concurrent_unordered_map.hpp>
#include <collection/concurrent_vector.hpp>

// Create a thread-safe map
general_wheel::concurrent_unordered_map<int, std::string> map;
map.insert({1, "one"});

// Create a thread-safe vector
general_wheel::concurrent_vector<int> vec;
vec.push_back(42);
```



## License

[MIT License](LICENSE)