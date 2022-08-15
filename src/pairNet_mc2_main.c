// pairNet_mc2_main.c
// Created by Awhu on 2022/8/15.
// testing pairnet with our Conv1d run by Gemmini

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "gemmini_custom.h"
#include "func.h"
#include "include/top_hfile.h"
#ifndef BAREMETAL
#include <sys/mman.h>
#endif


static void Relu_Clip(int batch_size, int out_dim, int out_channels, elem_t C[batch_size][out_dim][out_channels],
                      elem_t z4){
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            for (int k = 0; k < out_channels; ++k) {
                C[i][j][k] = QRelu_Clip(round_near_even(C[i][j][k]), z4, true);
            }
        }
    }
}
static void GAP(int batch_size, int input_width, int in_channels, elem_t input_feature[batch_size][input_width][in_channels],
                elem_t output_feature[batch_size][in_channels]){
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int c = 0; c < in_channels; c++) {
            double sum = 0;
            for (int i = 0; i < input_width; i++) {
                sum += input_feature[batch_idx][i][c];
            }
//            printf("%d\t", (elem_t)rounding(sum / (1 * input_width)));
            output_feature[batch_idx][c] = (elem_t)rounding(sum / (1 * input_width));
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

    elem_t QConv_BN_1_out[BATCH_SIZE][QConv_BN1_params.output_width][QConv_BN1_params.out_channels] ;
    elem_t QConv_BN_2_out[BATCH_SIZE][QConv_BN2_params.output_width][QConv_BN2_params.out_channels] ;
    elem_t QConv_BN_3_out[BATCH_SIZE][QConv_BN3_params.output_width][QConv_BN3_params.out_channels] ;
    elem_t QConv_BN_4_out[BATCH_SIZE][QConv_BN4_params.output_width][QConv_BN4_params.out_channels] ;
    elem_t QConv_BN_5_out[BATCH_SIZE][QConv_BN5_params.output_width][QConv_BN5_params.out_channels] ;
    elem_t QDense_out[BATCH_SIZE][gesN];

    start = read_cycles();
    ////1st layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN1_params.out_scale, (elem_t)QConv_BN1_params.z4, (elem_t *)gesture_signals, (elem_t *)QConv_BN_mc2_1,
                  (acc_t *)QConv_BN_bias1, QConv_BN_1_out, PE, QConv_BN1_params.input_width, QConv_BN1_params.stride_size,
                  QConv_BN1_params.kernel_size, QConv_BN1_params.in_channels, BATCH_SIZE,
                  QConv_BN1_params.out_channels, QConv_BN1_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN1_params.output_width, QConv_BN1_params.out_channels, QConv_BN_1_out,
              (elem_t)QConv_BN1_params.z4);


    //2nd layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN2_params.out_scale, (elem_t)QConv_BN2_params.z4, (elem_t *)gesture_signals, (elem_t *)QConv_BN_mc2_2,
                  (acc_t *)QConv_BN_bias2, QConv_BN_2_out, PE, QConv_BN2_params.input_width, QConv_BN2_params.stride_size,
                  QConv_BN2_params.kernel_size, QConv_BN2_params.in_channels, BATCH_SIZE,
                  QConv_BN2_params.out_channels, QConv_BN2_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN2_params.output_width, QConv_BN2_params.out_channels, QConv_BN_2_out,
              (elem_t)QConv_BN2_params.z4);

    //3rd layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN3_params.out_scale, (elem_t)QConv_BN3_params.z4, (elem_t *)gesture_signals, (elem_t *)QConv_BN_mc2_3,
                  (acc_t *)QConv_BN_bias3, QConv_BN_3_out, PE, QConv_BN3_params.input_width, QConv_BN3_params.stride_size,
                  QConv_BN3_params.kernel_size, QConv_BN3_params.in_channels, BATCH_SIZE,
                  QConv_BN3_params.out_channels, QConv_BN3_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN3_params.output_width, QConv_BN3_params.out_channels, QConv_BN_3_out,
              (elem_t)QConv_BN3_params.z4);

    //4th layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN4_params.out_scale, (elem_t)QConv_BN4_params.z4, (elem_t *)gesture_signals, (elem_t *)QConv_BN_mc2_4,
                  (acc_t *)QConv_BN_bias4, QConv_BN_4_out, PE, QConv_BN4_params.input_width, QConv_BN4_params.stride_size,
                  QConv_BN4_params.kernel_size, QConv_BN3_params.in_channels, BATCH_SIZE,
                  QConv_BN3_params.out_channels, QConv_BN3_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN3_params.output_width, QConv_BN3_params.out_channels, QConv_BN_3_out,
              (elem_t)QConv_BN4_params.z4);

    //5th layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float)QConv_BN5_params.out_scale, (elem_t)QConv_BN5_params.z4, (elem_t *)gesture_signals, (elem_t *)QConv_BN_mc2_5,
                  (acc_t *)QConv_BN_bias5, QConv_BN_5_out, PE, QConv_BN5_params.input_width, QConv_BN5_params.stride_size,
                  QConv_BN5_params.kernel_size, QConv_BN5_params.in_channels, BATCH_SIZE,
                  QConv_BN5_params.out_channels, QConv_BN5_params.output_width);

    Relu_Clip(BATCH_SIZE, QConv_BN5_params.output_width, QConv_BN5_params.out_channels, QConv_BN_5_out,
              (elem_t)QConv_BN5_params.z4);


    end = read_cycles();

    post_processing(BATCH_SIZE, gesN, QDense_out,4);
    cost = end - start;
    printf("spent time: %lu\n", cost);
    printf("\n");
    printf("SUCCESS\n");
    exit(0);
}