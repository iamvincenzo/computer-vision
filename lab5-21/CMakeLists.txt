cmake_minimum_required(VERSION 2.8)
project(disparity)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS})
install(TARGETS disparity RUNTIME DESTINATION bin)
