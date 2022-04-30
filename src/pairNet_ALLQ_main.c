// PairNet_ALLQ_main.c
// Created by sam on 2022/1/20.
// Complete at 2022/1/30.

#include <stdio.h>
#include <stdbool.h>
//#ifndef BAREMETAL
//#include <sys/mman.h>
//#endif
#include "include/gemmini_custom.h"
#include "func.h"
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/gemmini_params.h"
#include "Qgesture_signals_984.h"
#include "Qpairnet_params12_48_984.h"

int main(){
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t start, end;
    /*****PairNet Quantized*****/
    printf("PairNet matmul with Gemmini\n");
    //printf("PairNet matmul with CPU\n");
    printf("Input Gesture : %s \n", TRUE_LABEL);
    int gesN = GESN;
//    gemmini_flush(0);
    ////conv1
//    start = read_cycles();

    conv1d_matmul_cpu(QConv_BN_1_params.batch_size,QConv_BN_1_params.input_width,QConv_BN_1_params.in_channels,gesture_signals,
                    QConv_BN_1_params.kernel_size, QConv_BN_1_params.out_channels, QConv_BN_1_params.stride_size, QConv_BN_1,
                    QConv_BN_1_params.padding_front,QConv_BN_1_params.padding_back, QConv_BN_1_params.output_width,QConv_BN_bias_1,
                    downScalar_1, z3_1,z4_1, QConv_BN_1_out);
    //end = read_cycles();
    //printf("Cost(clock cycles) conv1d1 = %lu\n", end - start);
//    block_print1(1,QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels,QConv_BN_1_out);

    ////conv2
    //start = read_cycles();

    conv1d_matmul_cpu(QConv_BN_2_params.batch_size,QConv_BN_2_params.input_width,QConv_BN_2_params.in_channels,QConv_BN_1_out,
                    QConv_BN_2_params.kernel_size, QConv_BN_2_params.out_channels, QConv_BN_2_params.stride_size, QConv_BN_2,
                    QConv_BN_2_params.padding_front,QConv_BN_2_params.padding_back, QConv_BN_2_params.output_width,QConv_BN_bias_2,
                    downScalar_2, z3_2,z4_2, QConv_BN_2_out);
//    block_print1(1,QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels,QConv_BN_2_out);
    //end = read_cycles();
    //printf("Cost(clock cycles) conv1d2 = %lu\n", end - start);
    ////conv3
    //start = read_cycles();
    conv1d_matmul_cpu(QConv_BN_3_params.batch_size,QConv_BN_3_params.input_width,QConv_BN_3_params.in_channels,QConv_BN_2_out,
                          QConv_BN_3_params.kernel_size, QConv_BN_3_params.out_channels, QConv_BN_3_params.stride_size, QConv_BN_3,
                          QConv_BN_3_params.padding_front,QConv_BN_3_params.padding_back, QConv_BN_3_params.output_width,QConv_BN_bias_3,
                          downScalar_3, z3_3,z4_3, QConv_BN_3_out);
//    block_print1(1,QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels,QConv_BN_3_out);
    //end = read_cycles();
    //printf("Cost(clock cycles) conv1d3 = %lu\n", end - start);
    ////conv4
    //start = read_cycles();
    conv1d_matmul_cpu(QConv_BN_4_params.batch_size,QConv_BN_4_params.input_width,QConv_BN_4_params.in_channels,QConv_BN_3_out,
                          QConv_BN_4_params.kernel_size, QConv_BN_4_params.out_channels, QConv_BN_4_params.stride_size, QConv_BN_4,
                          QConv_BN_4_params.padding_front,QConv_BN_4_params.padding_back, QConv_BN_4_params.output_width,QConv_BN_bias_4,
                          downScalar_4, z3_4,z4_4, QConv_BN_4_out);
//    block_print1(1,QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels,QConv_BN_4_out);
    //end = read_cycles();
    //printf("Cost(clock cycles) conv1d4 = %lu\n", end - start);
    ////conv5
    //start = read_cycles();
    conv1d_matmul_cpu(QConv_BN_5_params.batch_size,QConv_BN_5_params.input_width,QConv_BN_5_params.in_channels,QConv_BN_4_out,
                          QConv_BN_5_params.kernel_size, QConv_BN_5_params.out_channels, QConv_BN_5_params.stride_size, QConv_BN_5,
                          QConv_BN_5_params.padding_front,QConv_BN_5_params.padding_back, QConv_BN_5_params.output_width,QConv_BN_bias_5,
                          downScalar_5, z3_5,z4_5, QConv_BN_5_out);
//    block_print1(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out);
    //end = read_cycles();
    //printf("Cost(clock cycles) conv1d5 = %lu\n", end - start);
    ////GAP
    //start = read_cycles();
    Qglobal_avg_pooling(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels,
                        QConv_BN_5_out, QGap_out);

    /*mc2_1dconv_global_avg(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels,PE,
                          (elem_t*)QConv_BN_5_out, (elem_t*)QGap_out);*/
    //end = read_cycles();
    //printf("Cost(clock cycles) GAP = %lu\n", end - start);
    /*printf("\nGAP result :\n\n");
    for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
	  //printf("batch %d\n", i);
       for (int j = 0; j < QConv_BN_5_params.out_channels; ++j) {
           printf("%d\t", QGap_out[i][j]);
        }
        //printf("\n");
    }*/
    ////Dense
    //start = read_cycles();
    QDense_cpu(QConv_BN_5_params.batch_size, QConv_BN_5_params.in_channels, gesN, QGap_out, QDense_params, QDense_bias, QDense_out,downScalar_dense);
    /*printf("\nDense result :\n\n");
    for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
	  printf("batch %d\n", i);
       for (int j = 0; j < gesN; ++j) {
           printf("%d\t", QDense_out[i][j]);
        }
        printf("\n");
    }*/
    //end = read_cycles();
    //printf("Cost(clock cycles) Dense = %lu\n", end - start);
    //start = read_cycles();
    post_processing(QConv_BN_5_params.batch_size, gesN, QDense_out,LEN_LABLE);
//////    end = read_cycles();
//////    printf("Cost(clock cycles) = %lu\n", end - start);
//////    double t_cost = (double )(end - start) / 31250000.0;
//////    printf("Cost(Second) = %f\n", t_cost);
    printf("SUCCESS\n");
}
