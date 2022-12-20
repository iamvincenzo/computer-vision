cmake_minimum_required(VERSION 2.8)
project(hough)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(sol sol.cpp)
target_link_libraries(sol ${OpenCV_LIBS})
install(TARGETS sol RUNTIME DESTINATION bin)
