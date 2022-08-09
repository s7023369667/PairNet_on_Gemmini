cmake_minimum_required(VERSION 3.21)
project(gemmini_projects C)
set(CMAKE_C_STANDARD 99)

add_executable(PairNet_ALLQ
        include/func.h
        pbPlots/pbPlots.h
        pbPlots/pbPlots.c
        pbPlots/supportLib.c
        pbPlots/supportLib.h
        src/pairNet_ALLQ_main.c
        include/gemmini_custom.h
        ../../chipyard/generators/gemmini/software/gemmini-rocc-tests/include/top_hfile.h)

add_executable(conv1d
        include/func.h
        src/conv1d_main.c
        )

target_link_libraries(PairNet_ALLQ m)
target_include_directories(PairNet_ALLQ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pbPlots)
target_include_directories(PairNet_ALLQ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(conv1d m)
target_include_directories(conv1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories("/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests")