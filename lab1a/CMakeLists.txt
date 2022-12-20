cmake_minimum_required(VERSION 2.8)
project(convolution)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g")

add_executable(es1 es1.cpp)
target_link_libraries(es1 ${OpenCV_LIBS})
install(TARGETS es1 RUNTIME DESTINATION bin)

add_executable(conv convolution.cpp)
target_link_libraries(conv ${OpenCV_LIBS})
install(TARGETS conv RUNTIME DESTINATION bin)

add_executable(conv3D conv3D.cpp)
target_link_libraries(conv3D ${OpenCV_LIBS})
install(TARGETS conv3D RUNTIME DESTINATION bin)

