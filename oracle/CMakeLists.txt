# artemis-oracle: llama.cpp as Artemis's oracle. Builds llama.cpp (CPU and Vulkan backends) from LLAMA_DIR as a
# subproject, plus our dump tool. Everything lands in one bin directory so the tools find libllama and the backends.
cmake_minimum_required(VERSION 3.21)
project(artemis_oracle C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout at the pinned tag")
if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "set LLAMA_DIR to a llama.cpp checkout")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp)

add_executable(artemis-oracle dump.cpp)
target_link_libraries(artemis-oracle PRIVATE llama ggml ggml-cpu)
target_compile_features(artemis-oracle PRIVATE cxx_std_17)
# mtp-taps uses llama.cpp's staging header (llama_set_embeddings_nextn and friends), which lives in src/.
target_include_directories(artemis-oracle PRIVATE ${LLAMA_DIR}/src)
