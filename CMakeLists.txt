cmake_minimum_required(VERSION 3.30)
project(LicensePlateRecognition)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LicensePlateRecognition
        main.cpp
        src/functions.cpp
        src/functions.h
        testing/testing.cpp
        testing/testing.h
)

target_link_libraries(LicensePlateRecognition ${OpenCV_LIBS})