cmake_minimum_required(VERSION 2.8)
project(esercitazione1)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(es1_10 es1_10.cpp)
target_link_libraries(es1_10 ${OpenCV_LIBS})
install(TARGETS es1_10 RUNTIME DESTINATION bin)

add_executable(downsample 11downsample.cpp)
target_link_libraries(downsample ${OpenCV_LIBS})
install(TARGETS downsample RUNTIME DESTINATION bin)

add_executable(luminance 12luminance.cpp)
target_link_libraries(luminance ${OpenCV_LIBS})
install(TARGETS luminance RUNTIME DESTINATION bin)

add_executable(simple 13simple.cpp)
target_link_libraries(simple ${OpenCV_LIBS})
install(TARGETS simple RUNTIME DESTINATION bin)

