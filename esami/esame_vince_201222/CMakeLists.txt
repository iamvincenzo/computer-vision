cmake_minimum_required(VERSION 2.8)
project(hough)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(skel 339641.cpp)
target_link_libraries(skel ${OpenCV_LIBS})
install(TARGETS skel RUNTIME DESTINATION bin)
