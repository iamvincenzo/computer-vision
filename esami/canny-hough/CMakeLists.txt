cmake_minimum_required(VERSION 2.8)
project(simple)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
install(TARGETS main RUNTIME DESTINATION bin)
