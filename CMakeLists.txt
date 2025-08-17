cmake_minimum_required(VERSION 3.14)
project(general-wheel-cpp VERSION 0.1.0 LANGUAGES CXX)



# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



# LINALG_USE_NEON as a configurable option, default ON for ARM64 platforms
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    option(LINALG_USE_NEON "Enable NEON optimizations for linear algebra operations" ON)
else()
    option(LINALG_USE_NEON "Enable NEON optimizations for linear algebra operations" OFF)
endif()

option(LINALG_USE_ASM "Enable assembly optimizations for linear algebra operations" OFF)


# Output platform information
message(STATUS "Build platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Output the status of the LINALG_USE_NEON option
message(STATUS "Linear algebra NEON optimizations (LINALG_USE_NEON): ${LINALG_USE_NEON}")

# Output the status of the LINALG_USE_ASM option
message(STATUS "Linear algebra assembly optimizations (LINALG_USE_ASM): ${LINALG_USE_ASM}")


# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "include/*.h"
)

# Create library target
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# Add include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

# Enable testing
enable_testing()

add_executable(${PROJECT_NAME}_test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/markdown_utils.cpp)
target_include_directories(${PROJECT_NAME}_test
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

if(LINALG_USE_NEON)
    target_compile_definitions(${PROJECT_NAME} PUBLIC LINALG_USE_NEON)
    target_compile_definitions(${PROJECT_NAME}_test PRIVATE LINALG_USE_NEON)
endif()

if(LINALG_USE_ASM)
    target_compile_definitions(${PROJECT_NAME} PUBLIC LINALG_USE_ASM)
    target_compile_definitions(${PROJECT_NAME}_test PRIVATE LINALG_USE_ASM)
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
)

# Export targets
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Generate the config file
include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR include
    CACHE PATH "Installation directory for header files")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
    PATH_VARS INCLUDE_INSTALL_DIR
)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
)