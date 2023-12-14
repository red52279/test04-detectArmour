CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(detectArmour)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(src/detect)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
        detect
        )

















#include_directories(${PROJECT_SOURCE_DIR}/src/detect/include)
#aux_source_directory(${PROJECT_SOURCE_DIR}/src/detect/src SRC)
#aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC2)
#set(SRC ${SRC} ${SRC2})






