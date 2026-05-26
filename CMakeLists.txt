cmake_minimum_required(VERSION 3.18)
project(turboquant_accelerator)

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# TT-Metal source-build layout.
# Set TT_METAL_HOME to your tt-metal clone root before running cmake:
#   export TT_METAL_HOME=/mnt/nvme10/joseph_ufl/tt-metal
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
# ---------------------------------------------------------------------------

set(TT_METAL_HOME "$ENV{TT_METAL_HOME}" CACHE PATH
    "Root of the tt-metal source checkout (must be built already)")

if(NOT TT_METAL_HOME)
    message(FATAL_ERROR
        "TT_METAL_HOME is not set. Run:\n"
        "  export TT_METAL_HOME=/mnt/nvme10/joseph_ufl/tt-metal\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release")
endif()

message(STATUS "TT_METAL_HOME = ${TT_METAL_HOME}")

# ---------------------------------------------------------------------------
# Include paths
# Confirmed layout from your build:
#   Headers : $TT_METAL_HOME/tt_metal/api/tt-metalium/
#             $TT_METAL_HOME/build_Release/include/tt-metalium/
#   Libs    : $TT_METAL_HOME/build/lib/
# ---------------------------------------------------------------------------

set(TT_METAL_INCLUDES
    ${TT_METAL_HOME}
    ${TT_METAL_HOME}/tt_metal
    ${TT_METAL_HOME}/tt_metal/api
    ${TT_METAL_HOME}/tt_metal/api/tt-metalium
    ${TT_METAL_HOME}/build_Release/include
    ${TT_METAL_HOME}/build_Release/include/tt-metalium
    ${TT_METAL_HOME}/tt_metal/hw/inc
    ${TT_METAL_HOME}/tt_metal/third_party/fmt/include
    ${TT_METAL_HOME}/tt_metal/third_party/umd
    ${TT_METAL_HOME}/tt_metal/third_party/umd/device
    ${TT_METAL_HOME}/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include
)

# ---------------------------------------------------------------------------
# Libraries
# Confirmed present: libtt_metal.so  libdevice.so  libfmt.so  libtt_stl.so
# ---------------------------------------------------------------------------

set(TT_LIB_DIR ${TT_METAL_HOME}/build/lib)

find_library(TT_METAL_LIB  tt_metal  PATHS ${TT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(TT_DEVICE_LIB device    PATHS ${TT_LIB_DIR} NO_DEFAULT_PATH)
find_library(TT_FMT_LIB    fmt       PATHS ${TT_LIB_DIR} NO_DEFAULT_PATH)
find_library(TT_STL_LIB    tt_stl    PATHS ${TT_LIB_DIR} NO_DEFAULT_PATH)

set(TT_OPTIONAL_LIBS)
foreach(lib TT_DEVICE_LIB TT_FMT_LIB TT_STL_LIB)
    if(${lib})
        list(APPEND TT_OPTIONAL_LIBS ${${lib}})
    endif()
endforeach()

message(STATUS "tt_metal lib  = ${TT_METAL_LIB}")
message(STATUS "optional libs = ${TT_OPTIONAL_LIBS}")

# ---------------------------------------------------------------------------
# Main host binary
# ---------------------------------------------------------------------------

add_executable(turboquant_host host/turboquant_host.cpp)

target_compile_definitions(turboquant_host PRIVATE
    TQ_DIM=128
    TQ_BITS=4
    TQ_QJL_DIM=128
    HOST_BUILD
    FMT_HEADER_ONLY  # avoid fmt ODR issues with header-only mode
)

target_include_directories(turboquant_host PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TT_METAL_INCLUDES}
)

target_link_libraries(turboquant_host PRIVATE
    ${TT_METAL_LIB}
    ${TT_OPTIONAL_LIBS}
    pthread dl
)

# Embed RPATH so the binary finds the .so files at runtime without
# needing LD_LIBRARY_PATH
set_target_properties(turboquant_host PROPERTIES
    BUILD_RPATH   ${TT_LIB_DIR}
    INSTALL_RPATH ${TT_LIB_DIR}
)

# ---------------------------------------------------------------------------
# Multi-tile prototype host (Stage 0 first)
# ---------------------------------------------------------------------------

add_executable(turboquant_multi_tile multi_tile/host/turboquant_multi_tile.cpp)

target_compile_definitions(turboquant_multi_tile PRIVATE
    TQ_DIM=128
    TQ_BITS=4
    TQ_QJL_DIM=128
    HOST_BUILD
    FMT_HEADER_ONLY
)

target_include_directories(turboquant_multi_tile PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/multi_tile/include
    ${TT_METAL_INCLUDES}
)

target_link_libraries(turboquant_multi_tile PRIVATE
    ${TT_METAL_LIB}
    ${TT_OPTIONAL_LIBS}
    pthread dl
)

set_target_properties(turboquant_multi_tile PROPERTIES
    BUILD_RPATH   ${TT_LIB_DIR}
    INSTALL_RPATH ${TT_LIB_DIR}
)

# ---------------------------------------------------------------------------
# Multi-tile Stage 1 (PolarQuant) host
# ---------------------------------------------------------------------------

add_executable(turboquant_multi_tile_pq multi_tile/host/turboquant_multi_tile_pq.cpp)

target_compile_definitions(turboquant_multi_tile_pq PRIVATE
    TQ_DIM=128
    TQ_BITS=4
    TQ_QJL_DIM=128
    HOST_BUILD
    FMT_HEADER_ONLY
)

target_include_directories(turboquant_multi_tile_pq PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/multi_tile/include
    ${TT_METAL_INCLUDES}
)

target_link_libraries(turboquant_multi_tile_pq PRIVATE
    ${TT_METAL_LIB}
    ${TT_OPTIONAL_LIBS}
    pthread dl
)

set_target_properties(turboquant_multi_tile_pq PROPERTIES
    BUILD_RPATH   ${TT_LIB_DIR}
    INSTALL_RPATH ${TT_LIB_DIR}
)

# ---------------------------------------------------------------------------
# Multi-tile Stage 2a (inverse rotation) host
# ---------------------------------------------------------------------------

add_executable(turboquant_multi_tile_ir multi_tile/host/turboquant_multi_tile_ir.cpp)

target_compile_definitions(turboquant_multi_tile_ir PRIVATE
    TQ_DIM=128
    TQ_BITS=4
    TQ_QJL_DIM=128
    HOST_BUILD
    FMT_HEADER_ONLY
)

target_include_directories(turboquant_multi_tile_ir PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/multi_tile/include
    ${TT_METAL_INCLUDES}
)

target_link_libraries(turboquant_multi_tile_ir PRIVATE
    ${TT_METAL_LIB}
    ${TT_OPTIONAL_LIBS}
    pthread dl
)

set_target_properties(turboquant_multi_tile_ir PROPERTIES
    BUILD_RPATH   ${TT_LIB_DIR}
    INSTALL_RPATH ${TT_LIB_DIR}
)

# ---------------------------------------------------------------------------
# Multi-tile Stage 2b1 (residual) host
# ---------------------------------------------------------------------------

add_executable(turboquant_multi_tile_resid multi_tile/host/turboquant_multi_tile_resid.cpp)
target_compile_definitions(turboquant_multi_tile_resid PRIVATE
    TQ_DIM=128 TQ_BITS=4 TQ_QJL_DIM=128 HOST_BUILD FMT_HEADER_ONLY)
target_include_directories(turboquant_multi_tile_resid PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/multi_tile/include
    ${TT_METAL_INCLUDES})
target_link_libraries(turboquant_multi_tile_resid PRIVATE
    ${TT_METAL_LIB} ${TT_OPTIONAL_LIBS} pthread dl)
set_target_properties(turboquant_multi_tile_resid PROPERTIES
    BUILD_RPATH ${TT_LIB_DIR} INSTALL_RPATH ${TT_LIB_DIR})

# ---------------------------------------------------------------------------
# Multi-tile Stage 2b2 (QJL sign projection) host
# ---------------------------------------------------------------------------

add_executable(turboquant_multi_tile_proj multi_tile/host/turboquant_multi_tile_proj.cpp)
target_compile_definitions(turboquant_multi_tile_proj PRIVATE
    TQ_DIM=128 TQ_BITS=4 TQ_QJL_DIM=128 TQ_QJL_SEED=0xCAFEBABEu
    HOST_BUILD FMT_HEADER_ONLY)
target_include_directories(turboquant_multi_tile_proj PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/multi_tile/include
    ${TT_METAL_INCLUDES})
target_link_libraries(turboquant_multi_tile_proj PRIVATE
    ${TT_METAL_LIB} ${TT_OPTIONAL_LIBS} pthread dl)
set_target_properties(turboquant_multi_tile_proj PROPERTIES
    BUILD_RPATH ${TT_LIB_DIR} INSTALL_RPATH ${TT_LIB_DIR})

# ---------------------------------------------------------------------------
# Multi-tile Stage 2c (r_norm) host
# ---------------------------------------------------------------------------

add_executable(turboquant_multi_tile_rnorm multi_tile/host/turboquant_multi_tile_rnorm.cpp)
target_compile_definitions(turboquant_multi_tile_rnorm PRIVATE
    TQ_DIM=128 TQ_BITS=4 TQ_QJL_DIM=128 HOST_BUILD FMT_HEADER_ONLY)
target_include_directories(turboquant_multi_tile_rnorm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/multi_tile/include
    ${TT_METAL_INCLUDES})
target_link_libraries(turboquant_multi_tile_rnorm PRIVATE
    ${TT_METAL_LIB} ${TT_OPTIONAL_LIBS} pthread dl)
set_target_properties(turboquant_multi_tile_rnorm PROPERTIES
    BUILD_RPATH ${TT_LIB_DIR} INSTALL_RPATH ${TT_LIB_DIR})

# ---------------------------------------------------------------------------
# Launcher binary (no TT-Metal dependency)
# ---------------------------------------------------------------------------

add_executable(turboquant_launcher host/turboquant_launcher.cpp)

# ---------------------------------------------------------------------------
# Kernel files are loaded at runtime relative to the working directory.
# Always run turboquant_host from the project root so kernels/ is reachable.
# ---------------------------------------------------------------------------
