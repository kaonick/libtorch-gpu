cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

include_directories(${CMAKE_SOURCE_DIR})

message( ${CMAKE_SOURCE_DIR} )
message( ${CMAKE_CURRENT_SOURCE_DIR} )

set(CMAKE_PREFIX_PATH E:/software/libtorch1.2.0_GPU C:/opencv/opencv0703/opencv/build)


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(CRAFT-app CRAFT_test.cpp)
target_link_libraries(CRAFT-app ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET CRAFT-app PROPERTY CXX_STANDARD 11)





