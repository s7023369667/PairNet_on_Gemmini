/*Modified function */
/*PE8*8 Num of data = 16 */
/*To compare with original function: tiled_matmul_nn*/
/*To calculate Matrix multiple*/
// PairNet_ALLQ_main.c
// Created by sam on 2022/1/20.
// Complete at 2022/1/30.

#include <stdio.h>
#include <stdbool.h>
//#ifndef BAREMETAL
//#include <sys/mman.h>
//#endif
//#include "include/gemmini_custom.h"
#include "func.h"
//#include "include/gemmini.h"
//#include "include/gemmini_nn.h"
//#include "include/gemmini_params.h"
#include "Qgesture_signals_984.h"
#include "Qpairnet_params12_48_984.h"
//#include <stdio.h>
//#include <string.h>
//#include <stdbool.h>
//#ifndef BAREMETAL
//#include <sys/mman.h>
//#endif
//#include "include/gemmini_custom.h"
//#include "include/func.h"
//#include "include/gemmini.h"
//#include "include/gemmini_nn.h"
//#include "include/gemmini_params.h"
//#include "include/Qgesture_signals_2165.h"
//#include "include/Qpairnet_params12_64_2165.h"
//
//
//void Relu_Clip(int batch_size, int out_dim, int out_channels, elem_t C[batch_size][out_dim][out_channels],
//               elem_t z3, elem_t z4){
//    for (int i = 0; i < batch_size; ++i) {
//        for (int j = 0; j < out_dim; ++j) {
//            for (int k = 0; k < out_channels; ++k) {
//                C[i][j][k] = QRelu_Clip(round_near_even(C[i][j][k]), z3, z4, true);
//            }
//        }
//    }
//}
//void GAP(int batch_size, int input_width, int in_channels, elem_t input_feature[batch_size][input_width][in_channels],
//         elem_t output_feature[batch_size][in_channels]){
//    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
//        for (int c = 0; c < in_channels; ++c) {
//            double sum = 0;
//            for (int i = 0; i < input_width; ++i) {
//                sum += input_feature[batch_idx][i][c];
//            }
////            printf("%d\t", (elem_t)rounding(sum / (1 * input_width)));
//            output_feature[batch_idx][c] = (elem_t)rounding(sum / (1 * input_width));
//        }
//    }
//
//}
//



int main(){
//#ifndef BAREMETAL
//    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
//        perror("mlockall failed");
//        exit(1);
//    }
//#endif
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
//int main () {
//    /*****PairNet Quantized*****/
//    printf("PairNet conv1d with Gemmini\n");
//    printf("Input Gesture : %s \n", TRUE_LABEL);
//
//#ifndef BAREMETAL
//    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
//        perror("mlockall failed");
//        exit(1);
//    }
//#endif
//
//    int gesN = GESN;
//    gemmini_flush(0);
//    uint64_t start, end, cost;
//    enum tiled_matmul_type_t tiled_matmul_type = WS;
//
//
//    start = read_cycles();
//    ////1st layer
//    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_1, z4_1, (elem_t *)gesture_signals, (elem_t *)QConv_BN_mc2_1,
//                  (acc_t *)QConv_BN_bias_1, QConv_BN_1_out, PE, QConv_BN_1_params.input_width, QConv_BN_1_params.stride_size,
//                  QConv_BN_1_params.kernel_size, QConv_BN_1_params.in_channels, QConv_BN_1_params.batch_size,
//                  QConv_BN_1_params.out_channels, QConv_BN_1_params.output_width);
//
//
//
//    // block_print1(QConv_BN_1_params.batch_size,QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels,QConv_BN_1_out);
//    Relu_Clip(QConv_BN_1_params.batch_size, QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels, QConv_BN_1_out,
//              z3_1, z4_1);
//
//
//    //2nd layer
//    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_2, z4_2, (elem_t *)QConv_BN_1_out, (elem_t *)QConv_BN_mc2_2,
//                  (acc_t *)QConv_BN_bias_2, QConv_BN_2_out, PE, QConv_BN_2_params.input_width, QConv_BN_2_params.stride_size,
//                  QConv_BN_2_params.kernel_size, QConv_BN_2_params.in_channels, QConv_BN_2_params.batch_size,
//                  QConv_BN_2_params.out_channels, QConv_BN_2_params.output_width);
//
//    // for_loop();
//    Relu_Clip(QConv_BN_2_params.batch_size, QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels, QConv_BN_2_out,
//              z3_2, z4_2);
//    // block_print1(1,QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels,QConv_BN_2_out);
//
//
//    ////3rd layer
//    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_3, z4_3, (elem_t *)QConv_BN_2_out, (elem_t *)QConv_BN_mc2_3,
//                  (acc_t *)QConv_BN_bias_3, QConv_BN_3_out, PE, QConv_BN_3_params.input_width, QConv_BN_3_params.stride_size,
//                  QConv_BN_3_params.kernel_size, QConv_BN_3_params.in_channels, QConv_BN_3_params.batch_size,
//                  QConv_BN_3_params.out_channels, QConv_BN_3_params.output_width);
//    // for_loop();
//    // gemmini_fence();
//    Relu_Clip(QConv_BN_3_params.batch_size, QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels, QConv_BN_3_out,
//              z3_3, z4_3);
//    // block_print1(1,QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels,QConv_BN_3_out);
//
//
//    ////4th layer
//    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_4, z4_4, (elem_t *)QConv_BN_3_out, (elem_t *)QConv_BN_mc2_4,
//                  (acc_t *)QConv_BN_bias_4, QConv_BN_4_out, PE, QConv_BN_4_params.input_width, QConv_BN_4_params.stride_size,
//                  QConv_BN_4_params.kernel_size, QConv_BN_4_params.in_channels, QConv_BN_4_params.batch_size,
//                  QConv_BN_4_params.out_channels, QConv_BN_4_params.output_width);
//    // for_loop();
//    // gemmini_fence();
//    Relu_Clip(QConv_BN_4_params.batch_size, QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels, QConv_BN_4_out,
//              z3_4, z4_4);
//    // block_print1(1,QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels,QConv_BN_4_out);
//
//
//    ////5th layer
//    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_5, z4_5, (elem_t *)QConv_BN_4_out, (elem_t *)QConv_BN_mc2_5,
//                  (acc_t *)QConv_BN_bias_5, QConv_BN_5_out, PE, QConv_BN_5_params.input_width, QConv_BN_5_params.stride_size,
//                  QConv_BN_5_params.kernel_size, QConv_BN_5_params.in_channels, QConv_BN_5_params.batch_size,
//                  QConv_BN_5_params.out_channels, QConv_BN_5_params.output_width);
//    // for_loop();
//
//    Relu_Clip(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels, QConv_BN_5_out,
//              z3_5, z4_5);
//    // block_print1(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out);
//
//
//    // test_mc2_1dconv(tiled_matmul_type, NO_ACTIVATION, 1, 0, l1_input, l1_weight, l1_bias, l1_output, 8, 20, 1, 2, 10, 1, 10, 19);
//    // mc2_1dconv_global_avg(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels,PE,(elem_t*)QConv_BN_5_out, (elem_t*)QGap_out);
//    GAP(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels, QConv_BN_5_out, QGap_out);
//
//    // printf("QGAP\n");
//    // for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
//    //     printf("batch %d\n", i);
//    //     for (int j = 0; j < QConv_BN_5_params.out_channels; ++j) {
//    //         printf("%d\t", QGap_out[i][j]);
//    //     }
//    //     printf("\n");
//    // }
//
//    QDense_cpu(QConv_BN_5_params.batch_size,QConv_BN_5_params.in_channels, gesN, QGap_out, QDense_params, QDense_bias, QDense_out,
//           (float )downScalar_dense);
//
//    // for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
//    //     for (int j = 0; j < gesN; ++j) {
//    //         printf("%d\t", QDense_out[i][j]);
//    //     }
//    //     printf("\n");
//    // }
//
////    //QSoftMax(QConv_BN_5_params.batch_size, gesN, QDense_out,deq_softmax_out,s3_dense, z3_dense);
//    post_processing(QConv_BN_5_params.batch_size, gesN, QDense_out,LEN_LABLE);
//    end = read_cycles();
//    cost = end - start;
//    printf("Cost(clock cycles) = %lu\n", end - start);
//    double t_cost = (double )(end - start) / 31250000.0;
//    printf("Cost(Second) = %f\n", t_cost);
//    printf("SUCCESS\n");
//    exit(0);
//}