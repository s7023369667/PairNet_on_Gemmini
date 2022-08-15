// pairNet_ALLQ_main.c
// Created by sam on 2022/1/20.
//

#include <stdio.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/func.h"
#include "include/top_hfile.h"

int main(){
    /*****PairNet Quantized Inference*****/
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t start, end;
    int gesN = GESN;
//    gemmini_flush(0);
    ////conv1
    start = read_cycles();
    elem_t QConv_BN1_out[BATCH_SIZE][QConv_BN1_params.output_width][QConv_BN1_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN1_params.input_width,QConv_BN1_params.in_channels,gesture_signals,
                          QConv_BN1_params.kernel_size, QConv_BN1_params.out_channels, QConv_BN1_params.stride_size, QConv_BN1,
                          QConv_BN1_params.output_width,QConv_BN_bias1,QConv_BN1_params.s1,(elem_t)QConv_BN1_params.z1,
                          QConv_BN1_params.s2,(elem_t)QConv_BN1_params.z2,QConv_BN1_params.sb,(elem_t)QConv_BN1_params.zb,
                          QConv_BN1_params.s4,(elem_t)QConv_BN1_params.z4,QConv_BN1_out);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d1 = %lu\n", end - start);
//    block_print(1,QConv_BN1_params.output_width, QConv_BN1_params.out_channels,QConv_BN1_out);

    ////conv2
    start = read_cycles();
    elem_t QConv_BN2_out[BATCH_SIZE][QConv_BN2_params.output_width][QConv_BN2_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN2_params.input_width,QConv_BN2_params.in_channels,QConv_BN1_out,
                          QConv_BN2_params.kernel_size, QConv_BN2_params.out_channels, QConv_BN2_params.stride_size, QConv_BN2,
                          QConv_BN2_params.output_width,QConv_BN_bias2,QConv_BN2_params.s1,(elem_t)QConv_BN2_params.z1,
                          QConv_BN2_params.s2,(elem_t)QConv_BN2_params.z2,QConv_BN2_params.sb,(elem_t)QConv_BN2_params.zb,
                          QConv_BN2_params.s4,(elem_t)QConv_BN2_params.z4,QConv_BN2_out);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d2 = %lu\n", end - start);
//    block_print(1,QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels,QConv_BN_2_out);

    ////conv3
    start = read_cycles();
    elem_t QConv_BN3_out[BATCH_SIZE][QConv_BN3_params.output_width][QConv_BN3_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN3_params.input_width,QConv_BN3_params.in_channels,QConv_BN2_out,
                          QConv_BN3_params.kernel_size, QConv_BN3_params.out_channels, QConv_BN3_params.stride_size, QConv_BN3,
                          QConv_BN3_params.output_width,QConv_BN_bias3,QConv_BN3_params.s1,(elem_t)QConv_BN3_params.z1,
                          QConv_BN3_params.s2,(elem_t)QConv_BN3_params.z2,QConv_BN3_params.sb,(elem_t)QConv_BN3_params.zb,
                          QConv_BN3_params.s4,(elem_t)QConv_BN3_params.z4,QConv_BN3_out);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d3 = %lu\n", end - start);
    //    block_print(1,QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels,QConv_BN_3_out);
    ////conv4
    start = read_cycles();
    elem_t QConv_BN4_out[BATCH_SIZE][QConv_BN4_params.output_width][QConv_BN4_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN4_params.input_width,QConv_BN4_params.in_channels,QConv_BN3_out,
                          QConv_BN4_params.kernel_size, QConv_BN4_params.out_channels, QConv_BN4_params.stride_size, QConv_BN4,
                          QConv_BN4_params.output_width,QConv_BN_bias4,QConv_BN4_params.s1,(elem_t)QConv_BN4_params.z1,
                          QConv_BN4_params.s2,(elem_t)QConv_BN4_params.z2,QConv_BN4_params.sb,(elem_t)QConv_BN4_params.zb,
                          QConv_BN4_params.s4,(elem_t)QConv_BN4_params.z4,QConv_BN4_out);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d4 = %lu\n", end - start);
    //    block_print(1,QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels,QConv_BN_4_out);
    ////conv5
    start = read_cycles();
    elem_t QConv_BN5_out[BATCH_SIZE][QConv_BN5_params.output_width][QConv_BN5_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN5_params.input_width,QConv_BN5_params.in_channels,QConv_BN4_out,
                          QConv_BN5_params.kernel_size, QConv_BN5_params.out_channels, QConv_BN5_params.stride_size, QConv_BN5,
                          QConv_BN5_params.output_width,QConv_BN_bias5,QConv_BN5_params.s1,(elem_t)QConv_BN5_params.z1,
                          QConv_BN5_params.s2,(elem_t)QConv_BN5_params.z2,QConv_BN5_params.sb,(elem_t)QConv_BN5_params.zb,
                          QConv_BN5_params.s4,(elem_t)QConv_BN5_params.z4,QConv_BN5_out);
    end = read_cycles();
    printf("Cost(clock cycles) conv1d5 = %lu\n", end - start);
    //    block_print(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out);
    ////GAP
    start = read_cycles();
    elem_t QGap_out[BATCH_SIZE][QConv_BN5_params.out_channels];
    Qglobal_avg_pooling(BATCH_SIZE, QConv_BN5_params.output_width,QConv_BN5_params.out_channels,QConv_BN5_out, QGap_out);

//    mc2_1dconv_global_avg(BATCH_SIZE, QConv_BN5_params.output_width,QConv_BN5_params.out_channels,PE,
//                          (elem_t*)QConv_BN5_out, (elem_t*)QGap_out);
    end = read_cycles();
    printf("Cost(clock cycles) GAP = %lu\n", end - start);
    ////Dense
    start = read_cycles();
    elem_t QDense_out[BATCH_SIZE][gesN];
    QDense_gemmini(BATCH_SIZE, Dense1_params.K, Dense1_params.J, QGap_out, QDense_params, QDense_bias, QDense_out,Dense1_params.s1,
                   (elem_t)Dense1_params.z1, Dense1_params.s2, (elem_t)Dense1_params.z2, Dense1_params.sb, (elem_t)Dense1_params.zb,
                   Dense1_params.s3, (elem_t)Dense1_params.z3);
    end = read_cycles();
    printf("Cost(clock cycles) Dense = %lu\n", end - start);
    printf("Dense out\n");
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < Dense1_params.J; ++j) {
            printf("%d ", QDense_out[i][j]);
        }
        printf("\n");
    }
//    post_processing(BATCH_SIZE, gesN, QDense_out, GES_NUM);
//////    printf("Cost(clock cycles) = %lu\n", end - start);
//////    double t_cost = (double )(end - start) / 31250000.0;
//////    printf("Cost(Second) = %f\n", t_cost);
    printf("SUCCESS\n");
}
