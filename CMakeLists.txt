cmake_minimum_required(VERSION 3.22)
project(Raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenGL REQUIRED)

add_subdirectory(external/glfw)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/external/include)

file(GLOB SOURCES src/*.cpp src/*.cu)

# Copy shaders to build directory
set(EXECUTABLE_OUTPUT_DIR $<TARGET_FILE_DIR:${PROJECT_NAME}>)
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/res/shaders")
set(SHADER_OUTPUT_DIR "${EXECUTABLE_OUTPUT_DIR}/res/shaders")

file(GLOB SHADER_FILES "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag")

add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_FILES} "${SHADER_OUTPUT_DIR}"
    COMMENT "Copying shader files to ${SHADER_OUTPUT_DIR}"
)

add_executable(${PROJECT_NAME} ${SOURCES} external/glad.c)
target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL)

add_dependencies(${PROJECT_NAME} copy_shaders)