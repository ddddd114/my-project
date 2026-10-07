cmake_minimum_required(VERSION 3.11)
project(camera_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(MVS_ROOT "/opt/MVS")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(MVS_LIB "${MVS_ROOT}/lib/aarch64/libMvCameraControl.so")
else()
    set(MVS_LIB "${MVS_ROOT}/lib/64/libMvCameraControl.so")
endif()

add_executable(video_demo video_demo.cpp)
target_include_directories(video_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(video_demo ${OpenCV_LIBS})

add_executable(daheng_demo daheng_demo.cpp)
target_include_directories(daheng_demo PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(daheng_demo ${OpenCV_LIBS} gxiapi)
