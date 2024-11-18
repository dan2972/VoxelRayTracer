cmake_minimum_required(VERSION 3.22)
project(Raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenGL REQUIRED)

add_subdirectory(external/glfw)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/external/include)

file(GLOB SOURCES src/*.cpp src/*.cu)

add_executable(${PROJECT_NAME} ${SOURCES} external/glad.c)
target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL)