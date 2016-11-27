cmake_minimum_required(VERSION 3.6)
project(CI_Homework_2)

#Flags for the specified for the homework
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -DNDEBUG -g -static -std=c99 -pipe -lm")

set(SOURCE_FILES
    LICENSE
    README.md Problem_1/Problem_1.c)

add_executable(CI_Homework_2 ${SOURCE_FILES})