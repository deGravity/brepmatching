cmake_minimum_required(VERSION 3.14)
project(coincidence_matching)

set (CMAKE_CXX_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  ps_frustrum
  GIT_REPOSITORY   https://github.com/deGravity/parasolid_frustrum.git
  GIT_TAG   v1.0
)

FetchContent_MakeAvailable(ps_frustrum)

set(SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

add_executable(coincidence_matching 
  ${SOURCES}
)

target_include_directories(coincidence_matching PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(coincidence_matching PUBLIC 
  parasolid_frustrum 
  Eigen3::Eigen 
 )

add_definitions(-DPART1="${CMAKE_CURRENT_SOURCE_DIR}/output_baseline1.x_t")
add_definitions(-DPART2="${CMAKE_CURRENT_SOURCE_DIR}/output_baseline2.x_t")
add_definitions(-DMATCHING="${CMAKE_CURRENT_SOURCE_DIR}/match.json")