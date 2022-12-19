cmake_minimum_required(VERSION 2.8)
project(eroding_dilating)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(eroding_dilating eroding_dilating.cpp)
target_link_libraries(eroding_dilating ${OpenCV_LIBS})
install(TARGETS eroding_dilating RUNTIME DESTINATION bin)

add_executable(eroding_dilating_modified eroding_dilating_modified.cpp)
target_link_libraries(eroding_dilating_modified ${OpenCV_LIBS})
install(TARGETS eroding_dilating_modified RUNTIME DESTINATION bin)

