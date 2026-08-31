cmake_minimum_required(VERSION 3.22.1)
project(yakuyomi_ncnn)
set(CMAKE_CXX_STANDARD 17)

# ncnn is vendored as prebuilt static libs, one directory per Android ABI:
# arm64-v8a is the custom 20260718 build, armeabi-v7a is the official
# 20260526 android-vulkan prebuilt (same options: Vulkan, OpenMP, threads).
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI arm64-v8a)
endif()
set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
add_library(yakuyomi_ncnn SHARED ncnn_jni.cpp)
target_link_libraries(yakuyomi_ncnn ncnn log)