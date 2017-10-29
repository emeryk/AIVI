#
#  CMakeLists.txt - create video alphablending saliency
#
#   Cree le: 26 oct. 2011
#    Auteur: AIV LaBRI - Université Bordeaux 1
#		
#

cmake_minimum_required(VERSION 2.6)

PROJECT(DETECTION-VISAGES)

#SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "CMAKE_BUILD_TYPE: Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(TARGET1 "create_frames_salMix")
SET(SRC1 create_frames_salMix.cpp)

ADD_DEFINITIONS(-Wall -pedantic)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

ADD_EXECUTABLE(${TARGET1} ${SRC1})
TARGET_LINK_LIBRARIES(${TARGET1} ${OpenCV_LIBS})
