cmake_minimum_required(VERSION 3.21)
project(gemmini_projects C)
set(CMAKE_C_STANDARD 99)

add_executable(PairNet_QDEQ
        src/post_processing_plot.c
        src/pairNet_QDEQ_main.c
        include/func.h
        include/gesture_signals.h
        include/pairnet_params.h
        pbPlots/pbPlots.h
        pbPlots/pbPlots.c
        pbPlots/supportLib.c
        pbPlots/supportLib.h
)
add_executable(PairNet_ALLQ
        src/post_processing_plot.c
        include/func.h
        include/Qgesture_signals.h
        pbPlots/pbPlots.h
        pbPlots/pbPlots.c
        pbPlots/supportLib.c
        pbPlots/supportLib.h
        src/pairNet_ALLQ_main.c include/gemmini_custom.h)


add_executable(PairNet_mc2conv1d
        src/post_processing_plot.c
        src/mc2_conv1d_main.c
        include/Qgesture_signals.h
        pbPlots/pbPlots.h
        pbPlots/pbPlots.c
        pbPlots/supportLib.c
        pbPlots/supportLib.h include/gemmini_custom.h)
add_executable(
        pbPlots_example
        pbPlots/example4.c
        pbPlots/pbPlots.h
        pbPlots/pbPlots.c
        pbPlots/supportLib.c
        pbPlots/supportLib.h
        )

target_link_libraries(PairNet_QDEQ m)
target_include_directories(PairNet_QDEQ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pbPlots)
target_include_directories(PairNet_QDEQ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(PairNet_ALLQ m)
target_include_directories(PairNet_ALLQ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pbPlots)
target_include_directories(PairNet_ALLQ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(PairNet_mc2conv1d m)
target_include_directories(PairNet_mc2conv1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pbPlots)
target_include_directories(PairNet_mc2conv1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(pbPlots_example m)
include_directories("/home/sam/chipyard/generators/gemmini/software/gemmini-rocc-tests")