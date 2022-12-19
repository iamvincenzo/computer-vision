cmake_minimum_required(VERSION 2.8)
project(background-subraction)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(es1 es1.cpp)
target_link_libraries(es1 ${OpenCV_LIBS})
install(TARGETS es1 RUNTIME DESTINATION bin)

add_executable(es2 es2.cpp)
target_link_libraries(es2 ${OpenCV_LIBS})
install(TARGETS es2 RUNTIME DESTINATION bin)

add_executable(es3 es3.cpp)
target_link_libraries(es3 ${OpenCV_LIBS})
install(TARGETS es3 RUNTIME DESTINATION bin)

