// PairNet_ALLQ_main.c
// Created by sam on 2022/1/20.
// Complete at 2022/1/30.
//#ifndef BAREMETAL
//#include <sys/mman.h>
//#endif
#include "func.h"
#include "Qgesture_signals.h"
#include "Qpairnet_params.h"



int main(){
    double q = rounding(M0_5 * (1LL << 31));
    double test = (int )q >> right_shift_5;
    printf("%f\n", q);
    printf("%f\n", test);

    clock_t start = clock();
    /*****PairNet Quantized*****/
    printf("PairNet\n");
    printf("Input Gesture : %s \n", TRUE_LABEL);
    int gesN = GESN;
    /**conv1**/
    conv1d_matmul(QConv_BN_1_params.batch_size,QConv_BN_1_params.input_width,QConv_BN_1_params.in_channels,gesture_signals,
                      QConv_BN_1_params.kernel_size, QConv_BN_1_params.out_channels, QConv_BN_1_params.stride_size, QConv_BN_1,
                      QConv_BN_1_params.padding_front,QConv_BN_1_params.padding_back, QConv_BN_1_params.output_width,QConv_BN_bias_1,
                      downScalar_1, z3_1,z4_1, QConv_BN_1_out);
//    block_print(QConv_BN_1_params.batch_size,QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels,QConv_BN_1_out);
    //write_txt(QConv_BN_1_params.batch_size,QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels,QConv_BN_1_out, "**conv1**");
    /**conv2**/
    conv1d_matmul(QConv_BN_2_params.batch_size,QConv_BN_2_params.input_width,QConv_BN_2_params.in_channels,QConv_BN_1_out,
                      QConv_BN_2_params.kernel_size, QConv_BN_2_params.out_channels, QConv_BN_2_params.stride_size, QConv_BN_2,
                      QConv_BN_2_params.padding_front,QConv_BN_2_params.padding_back, QConv_BN_2_params.output_width,QConv_BN_bias_2,
                      downScalar_2, z3_2,z4_2, QConv_BN_2_out);
//    block_print(QConv_BN_2_params.batch_size,QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels,QConv_BN_2_out);
    //write_txt(QConv_BN_2_params.batch_size,QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels,QConv_BN_2_out, "**conv2**");
    /**conv3**/
    conv1d_matmul(QConv_BN_3_params.batch_size,QConv_BN_3_params.input_width,QConv_BN_3_params.in_channels,QConv_BN_2_out,
                      QConv_BN_3_params.kernel_size, QConv_BN_3_params.out_channels, QConv_BN_3_params.stride_size, QConv_BN_3,
                      QConv_BN_3_params.padding_front,QConv_BN_3_params.padding_back, QConv_BN_3_params.output_width,QConv_BN_bias_3,
                      downScalar_3, z3_3,z4_3, QConv_BN_3_out);
//    block_print(QConv_BN_3_params.batch_size,QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels,QConv_BN_3_out);
    //write_txt(QConv_BN_3_params.batch_size,QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels,QConv_BN_3_out, "**conv3**");

    /**conv4**/
    conv1d_matmul(QConv_BN_4_params.batch_size,QConv_BN_4_params.input_width,QConv_BN_4_params.in_channels,QConv_BN_3_out,
                      QConv_BN_4_params.kernel_size, QConv_BN_4_params.out_channels, QConv_BN_4_params.stride_size, QConv_BN_4,
                      QConv_BN_4_params.padding_front,QConv_BN_4_params.padding_back, QConv_BN_4_params.output_width,QConv_BN_bias_4,
                      downScalar_4, z3_4,z4_4, QConv_BN_4_out);
//    block_print(QConv_BN_4_params.batch_size,QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels,QConv_BN_4_out);
    //write_txt(QConv_BN_4_params.batch_size,QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels,QConv_BN_4_out, "**conv4**");

    /**conv5**/
    conv1d_matmul(QConv_BN_5_params.batch_size,QConv_BN_5_params.input_width,QConv_BN_5_params.in_channels,QConv_BN_4_out,
                      QConv_BN_5_params.kernel_size, QConv_BN_5_params.out_channels, QConv_BN_5_params.stride_size, QConv_BN_5,
                      QConv_BN_5_params.padding_front,QConv_BN_5_params.padding_back, QConv_BN_5_params.output_width,QConv_BN_bias_5,
                      downScalar_5, z3_5,z4_5, QConv_BN_5_out);
//    block_print(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out);
    //write_txt(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out, "**conv5**");
    printf("QGAP\n");
    Qglobal_avg_pooling(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels, QConv_BN_5_out, QGap_out);
    for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
        for (int j = 0; j < QConv_BN_5_params.out_channels; ++j) {
            printf("%d\t", QGap_out[i][j]);
        }
        printf("\n");
    }
    QDense(QConv_BN_5_params.batch_size,QConv_BN_5_params.out_channels, gesN, QGap_out, QDense_params, QDense_bias, QDense_out, downScalar_dense);
//    printf("QMatmul\n");
//    for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
//        for (int j = 0; j < gesN; ++j) {
//            printf("%d\t", QDense_out[i][j]);
//        }
//        printf("\n");
//    }
    QSoftMax(QConv_BN_5_params.batch_size, gesN, QDense_out,deq_softmax_out,s3_dense, z3_dense);
    post_processing(QConv_BN_5_params.batch_size, gesN, deq_softmax_out,LEN_LABLE);
    clock_t end = clock();
    printf("Cost(clock cycles) = %lu\n", end - start);
    printf("SUCCESS\n");
}