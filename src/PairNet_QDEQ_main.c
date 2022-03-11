// gemmini_pairNet_main.c
// Created by sam on 2021/01/11.
// BE-AWARE : gemmini could not do so much double computation

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
//#include "include/gemmini.h"
//#include "include/gemmini_params.h"
//#include "include/gemmini_nn.h"
#include "func.h"
#include "gesture_signals.h"
#include "pairnet_params.h"


void conv1d_Gemini(size_t batch_size, size_t input_width, size_t in_channels,
                   const double in_feature[batch_size][1][input_width][in_channels],
                   size_t kernel_size,size_t out_channels,size_t stride_size,
                   const double weight[1][kernel_size][in_channels][out_channels],
                   size_t padding_front, size_t padding_back, size_t output_width,
                   double out_feature[batch_size][1][output_width][out_channels]) {
//    enum tiled_matmul_type_t tiled_matmul_type = WS;
    //kernel_size = kernel_group
    double reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = weight[0][i][j][k];
            }
        }
    }
//    printf("Kernel:\n");
//    for (int i = 0; i < kernel_size * in_channels; ++i) {
//        for (int j = 0; j < out_channels; ++j) {
//            printf("%.15f ",reshape_kernel[i][j]);
//        }
//        printf("\n");
//    }
    bool padding_front_flag = (padding_front !=0) ? true : false;

    size_t padding_shape = input_width + padding_front + padding_back;

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /*reshape & padding input_feature*/
        double reshape_feature[output_width][kernel_size * in_channels];
        size_t padding_front_idx = 0;
        if (padding_front_flag){
            padding_front_idx = 1 * in_channels;
        }
        double padding_feature[1 * padding_shape * in_channels];
        for (int i = 0; i < input_width ; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                padding_feature[padding_front_idx] = in_feature[batch_idx][0][i][j];
                padding_front_idx += 1;
            }
        }
        size_t start = 0;
        for (size_t i = 0; i < output_width; ++i) {
            for (int j = 0; j < (kernel_size * in_channels); ++j) {
                reshape_feature[i][j] = padding_feature[(start + j)];
            }
            start += stride_size * in_channels;
        }
//        printf("Feature:\n");
//        for (int i = 0; i < output_width; ++i) {
//            for (int j = 0; j < kernel_size * in_channels; ++j) {
//                printf("%.15f ",reshape_feature[i][j]);
//            }
//            printf("\n");
//        }
        /**quantize**/
        elem_t quantized_feature[output_width][kernel_size*in_channels];
        elem_t quantized_kernel[kernel_size*in_channels][out_channels];
        double quantize_scale = quantizer(output_width, kernel_size * in_channels, out_channels,
                                         reshape_feature, quantized_feature, reshape_kernel, quantized_kernel);
        /**gemmini matmul**/
        elem_t gemmini_result[output_width][out_channels];

        double down_scalar = calculate_DownScalar(output_width,kernel_size*in_channels,out_channels,quantized_feature,quantized_kernel);
//        tiled_matmul_auto(output_width, out_channels,kernel_size*in_channels, (elem_t*)quantized_feature, (elem_t*)quantized_kernel,
//                          NULL, (elem_t*)gemmini_result, kernel_size*in_channels, out_channels, out_channels, out_channels,
//                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,down_scalar,0,false,
//                          false,false,false,false,3,tiled_matmul_type);
        for (int i = 0; i < output_width; ++i) {
            int64_t tmp_res = 0;
            for (int j = 0; j < out_channels; ++j) {
                for (int k = 0; k < kernel_size * in_channels; ++k) {
                    tmp_res += quantized_feature[i][k] * quantized_kernel[k][j];
                }
                gemmini_result[i][j] = relu_clip(round_near_even((double )tmp_res * down_scalar), false);
                tmp_res = 0;
            }
        }
//        printf("\nbatch idx = %d\n", index);
//        for (int i = 0; i < output_width;++i){
//            for (int j = 0;j<out_channels;++j){
//                printf("%d\t", gemmini_result[i][j]);
//            }
//            printf("\n");
//        }
        /**dequantize**/
        //double dequantize_result[output_width][out_channels];
        //dequantize(output_width, out_channels, gemmini_result, quantize_scale, down_scalar, dequantize_result);

        for (int i = 0; i < output_width; ++i) {
            for (int j = 0; j < out_channels; ++j) {
                out_feature[batch_idx][0][i][j] = ((double )gemmini_result[i][j]) / (quantize_scale * down_scalar);
                //out_feature[index][0][i][j] = dequantize_result[i][j];
            }
        }
    }
}


int main(int argc, char * argv[]){
//#ifndef BAREMETAL
//    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
//        perror("mlockall failed");
//        exit(1);
//    }
//#endif
//    gemmini_flush(0);
    uint64_t start,end;
    /*****PairNet*****/
    /**conv1**/

    conv1d_Gemini(conv1d_1_params.batch_size, conv1d_1_params.input_width,conv1d_1_params.in_channels, gesture_signals,
                  conv1d_1_params.kernel_size, conv1d_1_params.out_channels,conv1d_1_params.stride_size ,conv1d_1,
                  conv1d_1_params.padding_front, conv1d_1_params.padding_back,conv1d_1_params.output_width, conv1d_1_out);

    batch_normalization(conv1d_1_params.batch_size,conv1d_1_params.output_width, conv1d_1_params.out_channels,conv1d_1_out,
                        batch_normalization_1);
    Relu(conv1d_1_params.batch_size,conv1d_1_params.output_width, conv1d_1_params.out_channels,conv1d_1_out);
    /**conv2**/
    conv1d_Gemini(conv1d_2_params.batch_size, conv1d_2_params.input_width,conv1d_2_params.in_channels, conv1d_1_out,
                  conv1d_2_params.kernel_size,conv1d_2_params.out_channels,conv1d_2_params.stride_size,conv1d_2,
                  conv1d_2_params.padding_front, conv1d_2_params.padding_back,conv1d_2_params.output_width, conv1d_2_out);

    batch_normalization(conv1d_2_params.batch_size,conv1d_2_params.output_width, conv1d_2_params.out_channels,conv1d_2_out,
                        batch_normalization_2);
    Relu(conv1d_2_params.batch_size,conv1d_2_params.output_width, conv1d_2_params.out_channels,conv1d_2_out);

    /**conv3**/
    conv1d_Gemini(conv1d_3_params.batch_size, conv1d_3_params.input_width,conv1d_3_params.in_channels, conv1d_2_out,
                  conv1d_3_params.kernel_size,conv1d_3_params.out_channels,conv1d_3_params.stride_size,conv1d_3,
                  conv1d_3_params.padding_front, conv1d_3_params.padding_back,conv1d_3_params.output_width, conv1d_3_out);
    batch_normalization(conv1d_3_params.batch_size,conv1d_3_params.output_width, conv1d_3_params.out_channels,conv1d_3_out,
                        batch_normalization_3);
    Relu(conv1d_3_params.batch_size,conv1d_3_params.output_width, conv1d_3_params.out_channels,conv1d_3_out);

    /**conv4**/
    conv1d_Gemini(conv1d_4_params.batch_size, conv1d_4_params.input_width,conv1d_4_params.in_channels, conv1d_3_out,
                  conv1d_4_params.kernel_size,conv1d_4_params.out_channels,conv1d_4_params.stride_size,conv1d_4,
                  conv1d_4_params.padding_front, conv1d_4_params.padding_back,conv1d_4_params.output_width, conv1d_4_out);
    batch_normalization(conv1d_4_params.batch_size,conv1d_4_params.output_width, conv1d_4_params.out_channels,conv1d_4_out,
                        batch_normalization_4);
    Relu(conv1d_4_params.batch_size,conv1d_4_params.output_width, conv1d_4_params.out_channels,conv1d_4_out);

    /**conv5**/
    conv1d_Gemini(conv1d_5_params.batch_size, conv1d_5_params.input_width,conv1d_5_params.in_channels, conv1d_4_out,
                  conv1d_5_params.kernel_size,conv1d_5_params.out_channels,conv1d_5_params.stride_size,conv1d_5,
                  conv1d_5_params.padding_front, conv1d_5_params.padding_back,conv1d_5_params.output_width, conv1d_5_out);
    batch_normalization(conv1d_5_params.batch_size,conv1d_5_params.output_width, conv1d_5_params.out_channels,conv1d_5_out,
                        batch_normalization_5);
    Relu(conv1d_5_params.batch_size,conv1d_5_params.output_width, conv1d_5_params.out_channels,conv1d_5_out);
    ////block_print(conv1d_5_params.batch_size,conv1d_5_params.output_width,conv1d_5_params.out_channels,conv1d_5_out);
    /**Global Average Pooling**/
    global_avg_pooling(conv1d_5_params.batch_size,conv1d_5_params.output_width,conv1d_5_params.out_channels,conv1d_5_out,gap_out);
    /**Fully Connection Layer**/
    Matmul(conv1d_5_params.batch_size, conv1d_5_params.out_channels, 12, gap_out, dense_1_params, dense_1_bias, dense_out);
    /**SoftMax**/
    SoftMax(conv1d_5_params.batch_size, 12, dense_out);
    /**Post-Processing**/
    post_processing(conv1d_5_params.batch_size, 12, dense_out,4);
    exit(0);
}