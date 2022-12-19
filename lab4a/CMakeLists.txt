cmake_minimum_required(VERSION 2.8)
project(canny)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(canny CannyEdgeDetector.cpp)
target_link_libraries(canny ${OpenCV_LIBS})
install(TARGETS canny RUNTIME DESTINATION bin)

