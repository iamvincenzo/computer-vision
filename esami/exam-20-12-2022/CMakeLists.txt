cmake_minimum_required(VERSION 2.8)
project(simple)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(vince vince.cpp)
target_link_libraries(vince ${OpenCV_LIBS})
install(TARGETS vince RUNTIME DESTINATION bin)

