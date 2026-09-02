cmake_minimum_required(VERSION 3.22.1)
project(yakuyomi_ncnn)
set(CMAKE_CXX_STANDARD 17)

# ncnn is vendored as prebuilt static libs, one directory per Android ABI.
# Both ABIs are custom 20260526 builds with NCNN_SIMPLEOMP=ON (pthread-based
# OpenMP, no libomp __kmp_abort_process crash paths) + Vulkan + OpenMP:
# arm64-v8a was built 20260718, armeabi-v7a 20260901 (NDK r28, same flags).
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI arm64-v8a)
endif()
set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
add_library(yakuyomi_ncnn SHARED ncnn_jni.cpp)
target_link_libraries(yakuyomi_ncnn ncnn log)