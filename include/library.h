//
// Created by sam on 2022/4/23.
//
#include <stdbool.h>
#ifndef GEMMINI_PROJECTS_LIBRARY_H
#define GEMMINI_PROJECTS_LIBRARY_H

static void group_normalize(int batch_size, int input_width, int in_channels, double input_feature[batch_size][1][input_width][in_channels],
                            double params[2][in_channels], int G){
    /**paper : https://arxiv.org/pdf/1803.08494v3.pdf**/
    /**flatten**/
    double flattend_feature[1][batch_size*input_width*in_channels];
    int index = 0;
    for (int i = 0; i < batch_size; ++i) {
        for (int k = 0; k < input_width; ++k) {
            for (int l = 0; l < in_channels; ++l) {
                flattend_feature[0][index] = input_feature[i][0][k][l];
                index += 1;
            }
        }
    }
    /**reshape : (N, G, C//G, H, W)**/
    index = 0;
    double reshape_feature[batch_size][G][(int )(in_channels/G)][1][input_width];
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < G; ++j) {
            for (int k = 0; k < (int )(in_channels/G); ++k) {
                for (int l = 0; l < input_width; ++l) {
                    reshape_feature[i][j][k][0][l] = flattend_feature[0][index];
                    index += 1;
                }
            }
        }
    }

    double eps = 0.00001;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {

        for (int group = 0; group < G; ++group) {
            double sum = 0;
            double sum_var = 0;
            /**calculate means*/
            for (int channel = 0; channel < (int )(in_channels/G); ++channel) {
                for (int w = 0; w < input_width; ++w) {
                    sum += reshape_feature[batch_idx][group][channel][0][w];
                }
            }
            double means = sum / (input_width * ((int )(in_channels/G)));
            /**calculate variance*/
            for (int channel = 0; channel < (int )(in_channels/G); ++channel) {
                for (int w = 0; w < input_width; ++w) {
                    sum_var += pow((reshape_feature[batch_idx][group][channel][0][w] - means), 2);
                }
            }
            double variance = sum_var / (input_width * ((int )(in_channels/G)));
            /**normalization*/
            for (int channel = 0; channel < (int )(in_channels/G); ++channel) {
                for (int w = 0; w < input_width; ++w) {
                    reshape_feature[batch_idx][group][channel][0][w] = ((reshape_feature[batch_idx][group][channel][0][w] - means) / (sqrt(variance + eps))) ;
                }
            }
        }
    }
    /**flatten*/
    index = 0;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < G; ++j) {
            for (int k = 0; k < (int) (in_channels / G); ++k) {
                for (int l = 0; l < input_width; ++l) {
                    flattend_feature[0][index] = reshape_feature[i][j][k][0][l];
                    index += 1;
                }
            }
        }
    }
    /**reshape : (N, H, W, C)*/
    index = 0;
    for (int i = 0; i < batch_size; ++i) {
        for (int k = 0; k < input_width; ++k) {
            for (int l = 0; l < in_channels; ++l) {
                input_feature[i][0][k][l] = flattend_feature[0][index];
                index += 1;
            }
        }
    }
    /**consider gamma & beta*/
    for (int i = 0; i < batch_size; ++i) {
        int col = 0;
        for (int k = 0; k < in_channels; ++k) {
            for (int l = 0; l < input_width; ++l) {
                input_feature[i][0][l][k] = input_feature[i][0][l][k] * params[0][col] + params[1][col];
            }
            col += 1;
        }
    }
}
static void average_pooling(int batch_size, int input_width ,int in_channel, const double input_feature[batch_size][1][input_width][in_channel],
                            int output_width,int out_channel,double output_feature[batch_size][1][output_width][out_channel],
                            int stride_size, int kernel_size) {
    /**padding = valid*/
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int i = 0; i < in_channel; ++i) {
            bool stop_flag = false;
            int index = 0;
            for (int j=0;j< input_width;j+=stride_size){
                double sum = 0.0;
                for (int k=0;k<kernel_size;++k){
                    int curr = j + k;
                    if (curr < input_width){
                        sum += input_feature[batch_idx][0][curr][i];
                    } else{
                        stop_flag = true;
                        break;
                    }
                }
                if (stop_flag != true & index < output_width) {
                    double tmp = (sum / (double) kernel_size);
                    output_feature[batch_idx][0][index][i] = tmp;
                    index += 1;
                }else{
                    break;
                }
            }
        }
    }
}

#endif //GEMMINI_PROJECTS_LIBRARY_H


