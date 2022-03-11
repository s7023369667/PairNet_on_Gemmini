// PairNet_main.c
// Created by sam on 2021/12/22.
//
#include "func.h"
#include "gesture_signals.h"
#include "pairnet_params.h"

 void block_print(int batch_size, int output_width, int out_channels, double out_feature[batch_size][1][output_width][out_channels]){
    for (int i = 0; i < batch_size; ++i) {
        printf("batch %d\n", i);
        for (int k = 0; k < output_width; ++k) {
            for (int l = 0; l < out_channels; ++l) {
                printf("%.8f ", out_feature[i][0][k][l]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

 int main(){
    /*****PairNet*****/
    /**conv1**/
    conv1d_TF((double *)gesture_signals, conv1d_1_params.batch_size, conv1d_1_params.input_width,
              conv1d_1_params.in_channels, (double *)conv1d_1,  conv1d_1_params.out_channels,conv1d_1_params.kernel_size,
              conv1d_1_params.stride_size,conv1d_1_params.padding_front, conv1d_1_params.padding_back,
              conv1d_1_params.output_width, conv1d_1_out);

    batch_normalization(conv1d_1_params.batch_size,conv1d_1_params.output_width, conv1d_1_params.out_channels,conv1d_1_out,
                        batch_normalization_1);
     block_print(conv1d_1_params.batch_size,conv1d_1_params.output_width,conv1d_1_params.out_channels,conv1d_1_out);

     Relu(conv1d_1_params.batch_size,conv1d_1_params.output_width, conv1d_1_params.out_channels,conv1d_1_out);

    /**conv2**/
    conv1d_TF((double *)conv1d_1_out, conv1d_2_params.batch_size, conv1d_2_params.input_width,
              conv1d_2_params.in_channels, (double *)conv1d_2, conv1d_2_params.out_channels,
              conv1d_2_params.kernel_size, conv1d_2_params.stride_size,conv1d_2_params.padding_front, conv1d_2_params.padding_back,
              conv1d_2_params.output_width, conv1d_2_out);
    batch_normalization(conv1d_2_params.batch_size,conv1d_2_params.output_width, conv1d_2_params.out_channels,conv1d_2_out,
                        batch_normalization_2);
    Relu(conv1d_2_params.batch_size,conv1d_2_params.output_width, conv1d_2_params.out_channels,conv1d_2_out);
    /**conv3**/
    conv1d_TF((double *)conv1d_2_out, conv1d_3_params.batch_size, conv1d_3_params.input_width,
              conv1d_3_params.in_channels, (double *)conv1d_3, conv1d_3_params.out_channels,
              conv1d_3_params.kernel_size, conv1d_3_params.stride_size,conv1d_3_params.padding_front, conv1d_3_params.padding_back,
              conv1d_3_params.output_width, conv1d_3_out);
    batch_normalization(conv1d_3_params.batch_size,conv1d_3_params.output_width, conv1d_3_params.out_channels,conv1d_3_out,
                        batch_normalization_3);
    Relu(conv1d_3_params.batch_size,conv1d_3_params.output_width, conv1d_3_params.out_channels,conv1d_3_out);
    /**conv4**/
    conv1d_TF((double *)conv1d_3_out, conv1d_4_params.batch_size, conv1d_4_params.input_width,
              conv1d_4_params.in_channels, (double *)conv1d_4,  conv1d_4_params.out_channels,
              conv1d_4_params.kernel_size, conv1d_4_params.stride_size,conv1d_4_params.padding_front, conv1d_4_params.padding_back,
              conv1d_4_params.output_width, conv1d_4_out);
    batch_normalization(conv1d_4_params.batch_size,conv1d_4_params.output_width, conv1d_4_params.out_channels,conv1d_4_out,
                        batch_normalization_4);
    Relu(conv1d_4_params.batch_size,conv1d_4_params.output_width, conv1d_4_params.out_channels,conv1d_4_out);
    /**conv5**/
    conv1d_TF((double *)conv1d_4_out, conv1d_5_params.batch_size, conv1d_5_params.input_width,
              conv1d_5_params.in_channels, (double *)conv1d_5,  conv1d_5_params.out_channels,
              conv1d_5_params.kernel_size, conv1d_5_params.stride_size,conv1d_5_params.padding_front, conv1d_5_params.padding_back,
              conv1d_5_params.output_width, conv1d_5_out);
    batch_normalization(conv1d_5_params.batch_size,conv1d_5_params.output_width, conv1d_5_params.out_channels,conv1d_5_out,
                        batch_normalization_5);
    Relu(conv1d_5_params.batch_size,conv1d_5_params.output_width, conv1d_5_params.out_channels,conv1d_5_out);
    //block_print(conv1d_5_params.batch_size,conv1d_5_params.output_width,conv1d_5_params.out_channels,conv1d_5_out);
    /**Global Average Pooling**/
    global_avg_pooling(conv1d_5_params.batch_size,conv1d_5_params.output_width,conv1d_5_params.out_channels,conv1d_5_out,gap_out);
    /**Fully Connection Layer**/
    Matmul(conv1d_5_params.batch_size, conv1d_5_params.out_channels, 12, gap_out, dense_1_params, dense_1_bias ,dense_out);
    /**SoftMax**/
    SoftMax(conv1d_5_params.batch_size, 12, dense_out);
    /**Post-Processing**/
    post_processing(conv1d_5_params.batch_size, 12, dense_out,4);
}
