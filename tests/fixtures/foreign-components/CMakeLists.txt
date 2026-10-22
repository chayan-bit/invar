cmake_minimum_required(VERSION 3.22)
project(invar_foreign_fixture_probe LANGUAGES C CXX)

find_package(pinocchio 4.1.0 EXACT REQUIRED)

add_executable(pinocchio_fk_cli
  pinocchio_fk_cli.cpp
  ../../../examples/cells/reach_place/components/pinocchio_fk/core.cpp)
target_compile_features(pinocchio_fk_cli PRIVATE cxx_std_17)
target_include_directories(pinocchio_fk_cli PRIVATE
  ../../../examples/cells/reach_place/components/pinocchio_fk)
target_link_libraries(pinocchio_fk_cli PRIVATE pinocchio::pinocchio)

add_library(pinocchio_fk_reference SHARED
  ../../../examples/cells/reach_place/components/pinocchio_fk/abi_reference.cpp
  ../../../examples/cells/reach_place/components/pinocchio_fk/core.cpp)
target_compile_features(pinocchio_fk_reference PRIVATE cxx_std_17)
target_link_libraries(pinocchio_fk_reference PRIVATE pinocchio::pinocchio)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(pinocchio_fk_reference PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/cells/reach_place/components/pinocchio_fk/abi_exports.map")
endif()

if(INVAR_GENERATED_INCLUDE_DIR)
  add_library(pinocchio_fk_generated SHARED
    ../../../examples/cells/reach_place/components/pinocchio_fk/bridge.c
    ../../../examples/cells/reach_place/components/pinocchio_fk/abi_reference.cpp
    ../../../examples/cells/reach_place/components/pinocchio_fk/core.cpp)
  target_compile_features(pinocchio_fk_generated PRIVATE c_std_11 cxx_std_17)
  target_include_directories(pinocchio_fk_generated PRIVATE
    "${INVAR_GENERATED_INCLUDE_DIR}"
    ../../../examples/cells/reach_place/components/pinocchio_fk)
  target_link_libraries(pinocchio_fk_generated PRIVATE pinocchio::pinocchio)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pinocchio_fk_generated PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/cells/reach_place/components/pinocchio_fk/bridge_exports.map")
  endif()
endif()

if(INVAR_GENERATED_UR5E_INCLUDE_DIR)
  add_library(pinocchio_fk_ur5e_generated SHARED
    ../../../examples/cells/reach_place/components/pinocchio_fk/bridge.c
    ../../../examples/cells/reach_place/components/pinocchio_fk/abi_reference.cpp
    ../../../examples/cells/reach_place/components/pinocchio_fk/core.cpp)
  target_compile_features(pinocchio_fk_ur5e_generated PRIVATE c_std_11 cxx_std_17)
  target_compile_definitions(pinocchio_fk_ur5e_generated PRIVATE INVAR_FK_UR5E)
  target_include_directories(pinocchio_fk_ur5e_generated PRIVATE
    "${INVAR_GENERATED_UR5E_INCLUDE_DIR}"
    ../../../examples/cells/reach_place/components/pinocchio_fk)
  target_link_libraries(pinocchio_fk_ur5e_generated PRIVATE pinocchio::pinocchio)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(pinocchio_fk_ur5e_generated PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/cells/reach_place/components/pinocchio_fk/bridge_exports.map")
  endif()
endif()

add_executable(pinocchio_abi_layout pinocchio_abi_layout.c)
target_compile_features(pinocchio_abi_layout PRIVATE c_std_11)
target_link_libraries(pinocchio_abi_layout PRIVATE pinocchio_fk_reference)
