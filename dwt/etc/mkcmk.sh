#!/bin/sh

CMK_FILE="CMakeLists.txt"
EXE_NAME="${1:-draw-walkthrough}"

echo "cmake_minimum_required(VERSION 3.4)" > $CMK_FILE
echo "set (CMAKE_CXX_STANDARD 11)" >> $CMK_FILE

echo "project($EXE_NAME)" >> $CMK_FILE
echo "include_directories(inc core)" >> $CMK_FILE
echo "find_library(lib-pthread pthread)" >> $CMK_FILE
echo "find_library(lib-dl dl)" >> $CMK_FILE
echo "find_library(lib-X11 X11)" >> $CMK_FILE
echo "add_definitions(-DX11)" >> $CMK_FILE

echo "add_executable(" >> $CMK_FILE
echo "$EXE_NAME" >> $CMK_FILE
./build.sh srcs $CMK_FILE append
echo ")" >> $CMK_FILE

echo "target_link_libraries(" >> $CMK_FILE
echo "$EXE_NAME \${lib-pthread} \${lib-dl} \${lib-X11}" >> $CMK_FILE
echo ")" >> $CMK_FILE

SHDR_DST="\${CMAKE_CURRENT_SOURCE_DIR}/data/simple_ogl"
SHDR_SRC="\${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
echo "add_custom_target(" >> $CMK_FILE
echo "shaders ALL" >> $CMK_FILE
echo "\${CMAKE_COMMAND} -E make_directory $SHDR_DST" >> $CMK_FILE
echo "COMMAND \${CMAKE_COMMAND} -E copy $SHDR_SRC/* $SHDR_DST" >> $CMK_FILE
echo ")" >> $CMK_FILE
