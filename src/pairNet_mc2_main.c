// pairNet_mc2_main.c
// Created by Awhu on 2022/8/15.
// testing pairnet with our Conv1d run by Gemmini

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "include/gemmini_custom.h"
#include "include/func.h"
#include "include/top_hfile.h"
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

static void preprocessing_signals(int batch_size, int in_dim, int in_channels, const elem_t input_signals[batch_size][in_dim][in_channels],
                                  elem_t out_signals[batch_size][in_dim][in_channels], elem_t input_z1){
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < in_dim; ++j) {
            for (int k = 0; k < in_channels; ++k) {
                out_signals[i][j][k] = clip(input_signals[i][j][k] - input_z1);
            }
        }
    }
}

static void Relu_Clip(int batch_size, int out_dim, int out_channels, elem_t C[batch_size][out_dim][out_channels],
                      elem_t z4, elem_t next_z1){
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            for (int k = 0; k < out_channels; ++k) {
                C[i][j][k] = QRelu_Clip(C[i][j][k], z4, true);
                ////dealing with quantization issue
                C[i][j][k] = clip(C[i][j][k] - next_z1);
            }
        }
    }
}

int main () {
    /*****PairNet Quantized*****/
    printf("PairNet conv1d with Gemmini\n");
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    int gesN = GESN;
    gemmini_flush(0);
    uint64_t start, end, cost;
    enum tiled_matmul_type_t tiled_matmul_type = WS;
    elem_t processed_signals[BATCH_SIZE][QConv_BN1_params.input_width][QConv_BN1_params.in_channels] ;
    elem_t QConv_BN_1_out[BATCH_SIZE][QConv_BN1_params.output_width][QConv_BN1_params.out_channels] ;
    elem_t QConv_BN_2_out[BATCH_SIZE][QConv_BN2_params.output_width][QConv_BN2_params.out_channels] ;
    elem_t QConv_BN_3_out[BATCH_SIZE][QConv_BN3_params.output_width][QConv_BN3_params.out_channels] ;
    elem_t QConv_BN_4_out[BATCH_SIZE][QConv_BN4_params.output_width][QConv_BN4_params.out_channels] ;
    elem_t QConv_BN_5_out[BATCH_SIZE][QConv_BN5_params.output_width][QConv_BN5_params.out_channels] ;
    elem_t QGap_out[BATCH_SIZE][QConv_BN5_params.out_channels];
    elem_t QDense_out[BATCH_SIZE][gesN];
    preprocessing_signals(BATCH_SIZE, QConv_BN1_params.input_width, QConv_BN1_params.in_channels, gesture_signals,
                          processed_signals, (elem_t)QConv_BN1_params.z1);
    ////1st layer
    start = read_cycles();
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN1_params.out_scale,
                  (elem_t *)processed_signals, (elem_t *)QConv_BN_mc2_1,(acc_t *)QConv_BN_bias1, (elem_t *)QConv_BN_1_out, PE,
                  QConv_BN1_params.input_width, QConv_BN1_params.stride_size,
                  QConv_BN1_params.kernel_size, QConv_BN1_params.in_channels, BATCH_SIZE,
                  QConv_BN1_params.out_channels, QConv_BN1_params.output_width);
    Relu_Clip(BATCH_SIZE, QConv_BN1_params.output_width, QConv_BN1_params.out_channels, QConv_BN_1_out,
              (elem_t)QConv_BN1_params.z4, (elem_t)QConv_BN2_params.z1);
//    block_print(1, QConv_BN1_params.output_width, QConv_BN1_params.out_channels, QConv_BN_1_out);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d1 = %lu\n", end - start);
    ////2nd layer
    start = read_cycles();
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN2_params.out_scale,
                  (elem_t *)QConv_BN_1_out, (elem_t *)QConv_BN_mc2_2,(acc_t *)QConv_BN_bias2, (elem_t *)QConv_BN_2_out, PE,
                  QConv_BN2_params.input_width, QConv_BN2_params.stride_size,
                  QConv_BN2_params.kernel_size, QConv_BN2_params.in_channels, BATCH_SIZE,
                  QConv_BN2_params.out_channels, QConv_BN2_params.output_width);
    Relu_Clip(BATCH_SIZE, QConv_BN2_params.output_width, QConv_BN2_params.out_channels, QConv_BN_2_out,
              (elem_t)QConv_BN2_params.z4, (elem_t)QConv_BN3_params.z1);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d2 = %lu\n", end - start);
    ////3rd layer
    start = read_cycles();
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN3_params.out_scale,
                  (elem_t *)QConv_BN_2_out,(elem_t *)QConv_BN_mc2_3,(acc_t *)QConv_BN_bias3, (elem_t *)QConv_BN_3_out, PE,
                  QConv_BN3_params.input_width, QConv_BN3_params.stride_size,
                  QConv_BN3_params.kernel_size, QConv_BN3_params.in_channels, BATCH_SIZE,
                  QConv_BN3_params.out_channels, QConv_BN3_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN3_params.output_width, QConv_BN3_params.out_channels, QConv_BN_3_out,
              (elem_t)QConv_BN3_params.z4, (elem_t)QConv_BN4_params.z1);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d3 = %lu\n", end - start);
    ////4th layer
    start = read_cycles();
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN4_params.out_scale,
                  (elem_t *)QConv_BN_3_out, (elem_t *)QConv_BN_mc2_4,(acc_t *)QConv_BN_bias4, (elem_t *)QConv_BN_4_out, PE,
                  QConv_BN4_params.input_width, QConv_BN4_params.stride_size,
                  QConv_BN4_params.kernel_size, QConv_BN4_params.in_channels, BATCH_SIZE,
                  QConv_BN4_params.out_channels, QConv_BN4_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN4_params.output_width, QConv_BN4_params.out_channels, QConv_BN_4_out,
              (elem_t)QConv_BN4_params.z4, (elem_t)QConv_BN5_params.z1);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d4 = %lu\n", end - start);
    ////5th layer
    start = read_cycles();
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN5_params.out_scale,
                  (elem_t *)QConv_BN_4_out, (elem_t *)QConv_BN_mc2_5,(acc_t *)QConv_BN_bias5, (elem_t *)QConv_BN_5_out, PE,
                  QConv_BN5_params.input_width, QConv_BN5_params.stride_size,
                  QConv_BN5_params.kernel_size, QConv_BN5_params.in_channels, BATCH_SIZE,
                  QConv_BN5_params.out_channels, QConv_BN5_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN5_params.output_width, QConv_BN5_params.out_channels, QConv_BN_5_out,
              (elem_t)QConv_BN5_params.z4, 0);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d5 = %lu\n", end - start);
    ////global average pooling
    start = read_cycles();
    mc2_1dconv_global_avg(BATCH_SIZE, QConv_BN5_params.output_width,QConv_BN5_params.out_channels,PE,
                          (elem_t*)QConv_BN_5_out, (elem_t*)QGap_out);
//    Qglobal_avg_pooling(BATCH_SIZE, QConv_BN5_params.output_width,QConv_BN5_params.out_channels,QConv_BN_5_out, QGap_out);
    end = read_cycles();
    printf("Cost(clock cycles) GAP = %lu\n", end - start);
    ////Dense
    start = read_cycles();
    QDense_gemmini(BATCH_SIZE, Dense1_params.K, Dense1_params.J, QGap_out, QDense_params, QDense_bias, QDense_out,Dense1_params.s1,
                   (elem_t)Dense1_params.z1, Dense1_params.s2, (elem_t)Dense1_params.z2, Dense1_params.sb, (elem_t)Dense1_params.zb,
                   Dense1_params.s3, (elem_t)Dense1_params.z3);
    end = read_cycles();
    printf("Cost(clock cycles) Dense = %lu\n", end - start);

    post_processing(BATCH_SIZE, gesN, QDense_out,3);
    cost = end - start;
    printf("spent time: %lu\n", cost);
    printf("\n");
    printf("SUCCESS\n");
    exit(0);
}