cmake_minimum_required(VERSION 2.8)
project(binarization)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(binarization binarization.cpp)
target_link_libraries(binarization ${OpenCV_LIBS})
install(TARGETS binarization RUNTIME DESTINATION bin)

add_executable(binarization1 binarization1.cpp)
target_link_libraries(binarization1 ${OpenCV_LIBS})
install(TARGETS binarization1 RUNTIME DESTINATION bin)

