cmake_minimum_required(VERSION 3.14)
project(coincidence_matching)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
  ps_frustrum
  GIT_REPOSITORY   https://github.com/deGravity/parasolid_frustrum.git
  GIT_TAG   v1.0
)

FetchContent_MakeAvailable(ps_frustrum)

add_library(cm_lib 
	${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/src/cm_lib.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/include/cm_lib.h)

add_executable(cm_debug ${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/src/debug.cpp)

pybind11_add_module(coincidence_matching   
	${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/src/bindings.cpp
)

target_include_directories(cm_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/coincidence_matching/include)

target_link_libraries(cm_lib PUBLIC parasolid_frustrum)
target_link_libraries(cm_debug PUBLIC cm_lib)
target_link_libraries(coincidence_matching PUBLIC cm_lib)

add_definitions(-DPART1="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/output_baseline1.x_t")
add_definitions(-DPART2="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/output_baseline2.x_t")
add_definitions(-DMATCHING="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/match.json")

add_definitions(-DSMALLCUBE="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/small_cube.x_t")
add_definitions(-DMEDCUBE="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/med_cube.x_t")
add_definitions(-DLARGECUBE="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/large_cube.x_t")
add_definitions(-DSHIFTCUBE="${CMAKE_CURRENT_SOURCE_DIR}/coincidence_matching/shifted_cube.x_t")