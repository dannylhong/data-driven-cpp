project(databook_cpp)
cmake_minimum_required(VERSION 3.2)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui
                    HINTS "/home/daniel/Libraries")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/daniel/Libraries/eigen3)
include_directories(/home/daniel/Libraries/matplotlib-cpp/)
include_directories(/usr/include/python3.10)
include_directories(/usr/lib/python3/dist-packages/numpy/core/include)
find_package(Matplot++ REQUIRED HINTS "/home/daniel/Libraries/vcpkg/installed/x64-linux")

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_subdirectory(CH01)