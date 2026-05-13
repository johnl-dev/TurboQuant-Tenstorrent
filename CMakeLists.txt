cmake_minimum_required(VERSION 3.18)
project(turboquant_accelerator)

set(CMAKE_CXX_STANDARD 20)

# Locate the tt-metal installation via $TT_METAL_HOME, matching the pattern
# used by rref_tenstorrent and other Tenstorrent community projects.
list(APPEND CMAKE_PREFIX_PATH $ENV{TT_METAL_HOME})
find_package(TTMetal REQUIRED)

# ---------------------------------------------------------------------------
# Main host binary
# ---------------------------------------------------------------------------
add_executable(turboquant_host host/turboquant_host.cpp)

target_compile_definitions(turboquant_host PRIVATE
    TQ_DIM=128
    TQ_BITS=4
    TQ_QJL_DIM=128
)

target_include_directories(turboquant_host PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(turboquant_host PRIVATE
    TTMetal::tt_metal
    TTMetal::tt_metal_distributed
)

# ---------------------------------------------------------------------------
# Launcher binary (interactive wrapper, no TT-Metal dependency)
# ---------------------------------------------------------------------------
add_executable(turboquant_launcher host/turboquant_launcher.cpp)

# ---------------------------------------------------------------------------
# Kernel files are discovered at runtime via path relative to the working
# directory — no special CMake step needed, just ensure you run the binary
# from the project root so kernels/ is reachable.
# ---------------------------------------------------------------------------
