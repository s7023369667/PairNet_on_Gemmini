// func.h
// Created by sam on 2021/12/22.
// BE-AWARE :
//  Gemmini only accept to import
//  your custom function in <file.h>,
//  unaccept to import <file.c>.

#ifndef GEMMINI_PROJECTS_FUNC_H
#define GEMMINI_PROJECTS_FUNC_H
//#include "include/gemmini.h"
//#include "include/gemmini_params.h"
//#include "include/gemmini_nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <time.h>

typedef int8_t elem_t;
typedef int32_t acc_t;

static double flooring(double x){
    if (x >= 0.0){
        if (x < (double )LLONG_MAX){
            return (double )(long long )x;
        }
        return x;
    } else if (x < 0.0){
        if (x >= LLONG_MIN){
            long long ix = (long long )x;
            return ((double )ix == x) ? x : (double )(ix - 1);
        }
        return x;
    }
    return x;
}
static double rounding(double x){
    double y,r;
    y = flooring(x);
    r = x - y;
    if (r > 0.5){
        y += 1.0;
    }
    if (r == 0.5){
        r = y - 2.0 * flooring(0.5 * y);
        if (r == 1){
            y += 1.0;
        }
    }
    //else : round down
    return y;
}

static int32_t round_near_even(double x){
    const double floatx = x;
    const int32_t  intx = (int32_t )floatx;
    int32_t  next = 0;
    int32_t result = 0;
    if (floatx < 0){
        next = (int32_t )floatx - 1;
    }else{
        next = (int32_t )floatx + 1;
    }
    double remain = floatx- (double )intx;
    if (remain<0){
        remain = -remain;
    }
    if (remain<0.5){
        result = intx;
    }else{
        if(remain > 0.5){
            result = next;
        }else{
            if (intx % 2 == 0){
                result = intx;
            }else{
                result = next;
            }
        }
    }
    return result;
}

static double expansion_function(double x){
    /** we cannot use exp() in riscv-unknown-elf compiler
     * reference : https://codereview.stackexchange.com/questions/123970/implementation-of-exp-function-in-c-using-taylor-series-expansion
     * **/
    double precision = 0.00000001;
    int n = 0;
    double x1 = 1;
    double sum = 0.0;
    do {
        sum += x1;
        x1 *= (x / (++n));
        // Stops when the next term to add becomes smaller than precision
    } while (x1 > precision);
//    printf("sum = %f\n", sum);
//    printf("exp = %f\n", exp(x));
    return sum;
}

static double Dequantization(acc_t q, double scale, double zeroPoint){
    return scale * (q - zeroPoint);
}
static int32_t Quantization(double r, double scale, double zeroPoint){
    return (int32_t)rounding((r/scale)+zeroPoint);
}

static void SoftMax(int batch_size, int in_channels, double input_feature[batch_size][in_channels]){
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        double sum = 0.0;
        for (int i = 0; i < in_channels; ++i) {
            input_feature[batch_idx][i] = expansion_function(input_feature[batch_idx][i]);
            sum += input_feature[batch_idx][i];
        }
        for (int i = 0; i < in_channels; ++i) {
            input_feature[batch_idx][i] = input_feature[batch_idx][i] / sum;
        }

    }
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < in_channels; ++j) {
            printf("%f ", input_feature[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    //write_result(batch_size, in_channels,input_feature);
}

static void QSoftMax(int batch_size, int in_channels,const elem_t input_feature[batch_size][in_channels],
                     double deq_feature[batch_size][in_channels], const double S_softmax, const elem_t Z_softmax){
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        double sum = 0.0;
        for (int i = 0; i < in_channels; ++i) {
            double deq_val = Dequantization(input_feature[batch_idx][i], S_softmax, (double )Z_softmax);
            deq_feature[batch_idx][i] = expansion_function(deq_val);
            sum += deq_feature[batch_idx][i];
        }
        for (int i = 0; i < in_channels; ++i) {
            deq_feature[batch_idx][i] = deq_feature[batch_idx][i] / sum;
        }

    }
//    for (int i = 0; i < batch_size; ++i) {
//        for (int j = 0; j < in_channels; ++j) {
//            printf("%d ", Quantization(deq_feature[i][j], S_softmax, Z_softmax));
//        }
//        printf("\n");
//    }
//    printf("\n");
}

static void batch_normalization(int batch_size, int input_width, int in_channels, double  input_feature[batch_size][1][input_width][in_channels],
                                const double params[4][in_channels]){
    /**
     * Reference :
     * https://d2l.ai/chapter_convolutional-modern/batch-norm.html
     * https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
     * https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
     **/
    int gamma = 0;int betta = 1;int moving_means = 2;int moving_var = 3;
    double eps = 1e-3;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int i = 0; i < input_width; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                input_feature[batch_idx][0][i][j] = params[gamma][j] *((input_feature[batch_idx][0][i][j] - params[moving_means][j]) / sqrt(params[moving_var][j] + eps)) + params[betta][j];
            }
        }
    }
}

static void global_avg_pooling(int batch_size, int input_width, int in_channels, double input_feature[batch_size][1][input_width][in_channels],
                               double output_feature[batch_size][in_channels]){
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int c = 0; c < in_channels; ++c) {
            double sum = 0.0;
            for (int i = 0; i < input_width; ++i) {
                sum += input_feature[batch_idx][0][i][c];
            }
            output_feature[batch_idx][c] = sum / (1 * input_width);
        }
    }
}

static void Qglobal_avg_pooling(int batch_size, int input_width, int in_channels, elem_t input_feature[batch_size][input_width][in_channels],
                                elem_t output_feature[batch_size][in_channels]){
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int c = 0; c < in_channels; ++c) {
            double sum = 0;
            for (int i = 0; i < input_width; ++i) {
                sum += input_feature[batch_idx][i][c];
            }
//            printf("%d\t", (elem_t)rounding(sum / (1 * input_width)));
            output_feature[batch_idx][c] = (elem_t)rounding(sum / (1 * input_width));
        }
    }
}

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

static void Relu(int batch_size, int input_width, int in_channels, double input_feature[batch_size][1][input_width][in_channels]){
    ////for PairNet_QDEQ_main.c
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int i = 0; i < input_width; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                if (input_feature[batch_idx][0][i][j] < 0){
                    input_feature[batch_idx][0][i][j] = 0;
                }
            }
        }
    }
}
static elem_t relu_clip(int64_t x, bool use_relu){
    ////for PairNet_QDEQ_main.c
    elem_t tmp;
    elem_t elem_t_max = 127;
    if (x < 0) {
        if (use_relu){
            tmp = 0;
        }else{
            if (x < -128){
                tmp = -128;
            }else{
                tmp = (elem_t)x;
            }
        }

    }else{
        if (x > elem_t_max){
            tmp = elem_t_max;
        }else{
            tmp=(elem_t)x;
        }
    }
    return tmp;
}

static elem_t QRelu_Clip(int32_t x,elem_t Z3, elem_t Z4 ,bool use_relu){

    elem_t tmp;
    elem_t elem_t_max = 127 ,elem_t_min = -128;
    if (use_relu){
        if (x < (int32_t)Z3){
            tmp = (elem_t)Z4;
        }else if (x > elem_t_max){
            tmp = (elem_t)elem_t_max;
        }else{
            tmp = (elem_t)x;
        }
    }else{
        if (x > elem_t_max){
            tmp = elem_t_max;
        }else if (x < elem_t_min){
            tmp = elem_t_min;
        }else{
            tmp = (elem_t)x;
        }
    }
    return (elem_t)tmp;
}

static void matix_multiply(double ** matrixA, double ** matrixB, double **matrixC,size_t I, size_t K, size_t J){
    for (int i = 0; i < I; ++i) {
        double tmp_res = 0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += matrixA[i][k] * matrixB[k][j];
            }
            matrixC[i][j] = tmp_res;
            tmp_res = 0;
        }
    }
}

static void Matmul(size_t I, size_t K, size_t J, double matrixA[I][K],const double matrixB[K][J],const double bias[J], double matrixC[I][J]){
    ////for PairNet_QDEQ_main.c
    for (int i = 0; i < I; ++i) {
        double tmp_res = 0.0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += matrixA[i][k] * matrixB[k][j];
            }
            matrixC[i][j] = tmp_res + bias[j];
            tmp_res = 0;
        }
    }
}


static void block_print(int batch_size, int output_width, int out_channels,
                        elem_t out_feature[batch_size][1][output_width][out_channels]){
    for (int i = 0; i < batch_size; ++i) {
        printf("\tbatch %d\n", i);
        for (int k = 0; k < output_width; ++k) {
            for (int l = 0; l < out_channels; ++l) {
                printf("\t%d\t ", out_feature[i][0][k][l]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

static void block_print1(int batch_size, int output_width, int out_channels,
                         elem_t out_feature[batch_size][output_width][out_channels]){
    for (int i = 0; i < batch_size; ++i) {
        printf("\tbatch %d\n", i);
        for (int k = 0; k < output_width; ++k) {
            for (int l = 0; l < out_channels; ++l) {
                printf("\t%d\t ", out_feature[i][k][l]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

static void block_DEQ_print(int batch_size, int output_width, int out_channels, elem_t out_feature[batch_size][1][output_width][out_channels],
                            double S4, elem_t Z4){
    for (int i = 0; i < batch_size; ++i) {
        printf("batch %d\n", i);
        for (int k = 0; k < output_width; ++k) {
            for (int l = 0; l < out_channels; ++l) {
                printf("%f\t ", Dequantization(out_feature[i][0][k][l], S4, Z4));
            }
            printf("\n");
        }
        printf("\n");
    }
}

static void write_result(int batch_size, int output_width, int out_channels, elem_t out_feature[batch_size][1][output_width][out_channels],
                         char layer_name[10]){
    //Before you run again , deleting LayerResult.txt
    FILE *file = fopen("./LayerResult.txt", "a+");
    for (int i = 0; i < 9; ++i) {
        fprintf(file, "%c",layer_name[i]);
    }
    fprintf(file, "\n");
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < out_channels; ++k) {
//                printf("%d\t ", out_feature[i][0][j][k]);
                fprintf(file, "%d\t", out_feature[i][0][j][k]);
            }
//            printf("\n");
            fprintf(file, "\n");
//    block_print(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out, s4_convBN_5, z4_convBN_5);
        }
//        printf("\n");
        fprintf(file, "\n");
    }
    fclose(file);
}

static double find_max(size_t intput_width, size_t in_channels, double input_feature[intput_width][in_channels]){
    ////for PairNet_QDEQ_main.c
    double maxx = -1000.0;
    for (int j = 0; j < intput_width; ++j) {
        for (int k = 0; k < in_channels; ++k) {
            if (maxx < input_feature[j][k]){
                maxx = input_feature[j][k];
            }
        }
    }
    return maxx;
}
static double find_min(size_t intput_width, size_t in_channels, double input_feature[intput_width][in_channels]){
    ////for PairNet_QDEQ_main.c
    double minn = 1000.0;
    for (int j = 0; j < intput_width; ++j) {
        for (int k = 0; k < in_channels; ++k) {
            if (minn > input_feature[j][k]){
                minn = input_feature[j][k];
            }
        }
    }
    return minn;
}

static void dequantizer(size_t output_width, size_t out_channels, elem_t output_feature[output_width][out_channels],
                        double quantize_scale, double down_scalar, double dequantized_feature[output_width][out_channels]){
    ////for PairNet_QDEQ_main.c
    for (int j = 0; j < output_width; ++j) {
        for (int k = 0; k < out_channels; ++k) {
            dequantized_feature[j][k] = (double )output_feature[j][k] / (quantize_scale * down_scalar);
        }
    }
}
static double quantizer(size_t I, size_t K, size_t J , double input_feature[I][K], elem_t quantized_feature[I][K],
                        double kernel[K][J], elem_t quantized_kernel[K][J]) {
    ////for PairNet_QDEQ_main.c

    /* Quantize input_feature in linear way*/
    //quantize feature
    double fp_max = find_max(I, K, input_feature);
    double fp_min = find_min(I, K, input_feature);
    double abs_max = fp_max, abs_min = fp_min;
    if (abs_min < 0) {
        abs_min = 0.0 - abs_min;
    }
    if (abs_max < 0) {
        abs_max = 0.0 - abs_max;
    }
    double maxx = (abs_max > abs_min) ? abs_max : abs_min;
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            /*Region input_feature from [fp_min,fp_max] to [-1,1]*/
            double tmp = input_feature[i][k] / maxx;
            /*Region input_feature from [-1,1] to [-128,127]*/
            tmp = rounding(tmp * 128);
            //clip
            if (tmp > 127) { tmp = 127; }
            else if (tmp < -128) { tmp = -128; }
            quantized_feature[i][k] = (elem_t) tmp;
        }
    }
    //quantize kernel
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < J; ++j) {
            double tmp = rounding(kernel[k][j] * 128);
            //clip
            if (tmp > 127) { tmp = 127; }
            else if (tmp < -128) { tmp = -128; }
            quantized_kernel[k][j] = (elem_t) tmp;
        }
    }
    double quantize_scale = 128 * 128 / maxx;
    return quantize_scale;
}

static double calculate_DownScalar(size_t I, size_t K, size_t J, elem_t quantized_feature[I][K], elem_t quantized_kernel[K][J]){
    ////for PairNet_QDEQ_main.c
    /*matmul*/
    int64_t result[I][J];
    for (int i = 0; i < I; ++i) {
        int64_t tmp_res = 0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += quantized_feature[i][k] * quantized_kernel[k][j];
            }
            result[i][j] = tmp_res;
            tmp_res = 0;
        }
    }
    int64_t res_max = -1000;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            if (res_max < result[i][j]){
                res_max = result[i][j];
            }
        }
    }
    int64_t res_min = 1000;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            if (res_min > result[i][j]){
                res_min = result[i][j];
            }
        }
    }
    int64_t abs_res_max = res_max , abs_res_min = res_min;
    if ( abs_res_max < 0){abs_res_max = 0 - abs_res_max;}
    if ( abs_res_min < 0){abs_res_min = 0 - abs_res_min;}
    int64_t maxx = (abs_res_max > abs_res_min) ? abs_res_max : abs_res_min;

//    printf("DownScalar = %f\n",128.0 / (double )maxx);
    return 128.0 / (double )maxx;
}

static void QMatmul(size_t I, size_t K, size_t J, const elem_t matrixA[I][K],const elem_t matrixB[K][J],const acc_t bias[J],
                    elem_t matrixC[I][J],double S1, elem_t Z1,double S2 , elem_t Z2,double S2_bias, elem_t Z2_bias,double S3, elem_t Z3){
    /**with pre-compute bias**/
    acc_t total_bias[I][J];
    for (int i = 0; i < I; ++i) {
        double tmp_res = 0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += (double )((Z2 * Z1) - (matrixB[k][j] * Z1) - (matrixA[i][k] * Z2)) ;
            }
            total_bias[i][j] = (acc_t)rounding(tmp_res + ((S2_bias / (S2 * S1)) * (bias[j] - Z2_bias))
                                               + (Z3 / ((S1 * S2) / S3)));
            tmp_res = 0;
        }
    }
    /**Gemmini*/
//    enum tiled_matmul_type_t tiled_matmul_type = WS;
//    tiled_matmul_auto(I, J,K, (elem_t*)matrixA, (elem_t*)matrixB,(acc_t*)total_bias, (elem_t*)matrixC,
//                      K, J, J, J, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,
//                      (S1 * S2 / S3),0,false,false,false,false,false,3,tiled_matmul_type);
//    for (int i = 0; i < I; ++i) {
//        for (int j = 0; j < J; ++j) {
//            matrixC[i][j] = QRelu_Clip(matrixC[i][j], Z3, 0, false);
//        }
//    }
    /**cpu*/
    for (int i = 0; i < I; ++i) {
        double tmp_res = 0.0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += (double )(matrixA[i][k] * matrixB[k][j]) ;
            }
            matrixC[i][j] = QRelu_Clip(round_near_even((((S1 * S2) / S3) * (tmp_res + total_bias[i][j]))),0, 0, false);
            tmp_res = 0;
        }
    }
}

static void QDense_Gemmini(int I, int K, int J, const elem_t matrixA[I][K],const elem_t matrixB[K][J],const acc_t bias[I][J],
                   elem_t matrixC[I][J], double downScalar){
    /**Gemmini**/
    enum tiled_matmul_type_t tiled_matmul_type = WS;
    tiled_matmul_auto(I, J, K, (elem_t*)matrixA, (elem_t*)matrixB,(acc_t*)bias, (elem_t*)matrixC,
                      K, J, J, J, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,
                      (float )downScalar,0,false,false,false,false,false,3,tiled_matmul_type);

static void QDense_cpu(int I, int K, int J, const elem_t matrixA[I][K],const elem_t matrixB[K][J],const acc_t bias[I][J],
                   elem_t matrixC[I][J], double downScalar){
    /**CPU**/
    for (int i = 0; i < I; ++i) {
        double tmp_res = 0.0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += (double )(matrixA[i][k] * matrixB[k][j]) ;
            }
            matrixC[i][j] = (elem_t)round_near_even((downScalar * (tmp_res + bias[i][j])));
            tmp_res = 0;
        }
    }
}

static void conv1d_TF(const double *in_feature, size_t input_batch_size, size_t input_width, size_t in_channels,
                      const double *weight,size_t out_channels,size_t kernel_size,size_t stride_size,
                      size_t padding_front, size_t padding_back, size_t output_width,
                      double out_feature[input_batch_size][1][output_width][out_channels]) {
    /**tensorflow conv1d**/
    //kernel_size = kernel_group
    double reshape_kernel[kernel_size * in_channels][out_channels];
    int kernel_idx = 0;
    for (int i = 0; i < kernel_size * in_channels; ++i) {
        for (int j = 0; j < out_channels; ++j) {
            reshape_kernel[i][j] = weight[kernel_idx];
            kernel_idx += 1;
        }
    }
//    printf("Kernel:\n");
//    pprint(kernel_size * in_channels,out_channels,reshape_kernel);

    bool padding_front_flag = (padding_front !=0) ? true : false;

    size_t padding_shape = input_width + padding_front + padding_back;

    int index = 0;
//    int batch_num = 0;
    for (size_t batch_idx = 0;batch_idx < (input_batch_size * input_width * in_channels); batch_idx += (input_width * in_channels)) {
//        printf("%d\n",batch_num);
//        batch_num += 1;

        /*reshape & padding input_feature*/
        double reshape_feature[output_width][kernel_size * in_channels];
        size_t padding_front_idx = 0;
        if (padding_front_flag){
            padding_front_idx = 1 * in_channels;
        }
        double padding_feature[1 * padding_shape * in_channels];
        for (int i = 0; i < input_width * in_channels; ++i) {
            padding_feature[padding_front_idx + i] = in_feature[batch_idx + i];
        }
        size_t start = 0;
        for (size_t i = 0; i < output_width; ++i) {
            for (int j = 0; j < (kernel_size * in_channels); ++j) {
                reshape_feature[i][j] = padding_feature[(start + j)];
            }
            start += stride_size * in_channels;
        }
//        printf("Feature:\n");
//        pprint(output_width,kernel_size * in_channels,reshape_feature);
        /*matmul*/
        double result_matrix[output_width][out_channels];
        for (int i = 0; i < output_width; ++i) {
            double tmp_res = 0.0;
            for (int j = 0; j < out_channels; ++j) {
                for (int k = 0; k < kernel_size * in_channels; ++k) {
                    tmp_res += reshape_feature[i][k] * reshape_kernel[k][j];
                }
                result_matrix[i][j] = tmp_res;
                tmp_res = 0;
            }
        }
        for (int i = 0; i < output_width; ++i) {
            for (int j = 0; j < out_channels; ++j) {
                out_feature[index][0][i][j] = result_matrix[i][j];
            }
        }
        index += 1;
    }
};

static void conv1d_original(size_t batch_size, size_t in_channels,size_t input_width,elem_t feature[batch_size][in_channels][input_width],
                            size_t kernel_group, size_t kernel_width,size_t kernel_size,elem_t kernel[kernel_group][kernel_width][kernel_size],
                            int stride_size,size_t output_width,elem_t result[output_width][kernel_group]){
    /**The stride version conv1d**/
    if (kernel_width != in_channels){
        printf("Error Exists.\n");
    }
    elem_t padding_feature[batch_size][in_channels][input_width + (kernel_size -1)];
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < in_channels; ++j) {
            for (int k = 0; k < (input_width + (kernel_size -1)); ++k) {
                if (k >= input_width){
                    padding_feature[i][j][k] = 0;
                }else{
                    padding_feature[i][j][k] = feature[i][j][k];
                }
                //printf("%d ",padding_feature[i][j][k]);
            }
            //printf("\n");
        }
    }
    for (int batch = 0; batch < batch_size; ++batch) {
        int col = 0;
        for (int group = 0; group < kernel_group; ++group) {
            int row = 0;
            for (int i = 0; i < (input_width + (kernel_size -1)); i+=stride_size) {
                if (i + stride_size > (input_width + (kernel_size -1) -1)){
                    break;
                }
                int sum = 0;
                for (int j = 0; j < in_channels; ++j) { //0 1 2
                    for (int k = 0; k < kernel_size; ++k) { // 0 1
                        sum += padding_feature[batch][j][i+k] * kernel[group][j][k];
                        //printf("%d += %d * %d\n", sum,feature[batch][j][i+k], kernel[group][j][k]);
                    }
                }
                result[row][col] = (elem_t)sum;
                row += 1;
            }
            col += 1;
        }
    }
}

static void conv1d_matmul_QDEQ(size_t batch_size, size_t input_width, size_t in_channels,
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

static void conv1d_TF_folding(size_t batch_size, size_t input_width, size_t in_channels,
                              const elem_t in_feature[batch_size][input_width][in_channels],
                              size_t kernel_size, size_t out_channels,size_t stride_size,
                              const elem_t conv1d_weight[kernel_size][in_channels][out_channels],
                              size_t padding_front, size_t padding_back, size_t output_width,
                              const acc_t conv1d_bias[output_width][out_channels],double S1, elem_t Z1,
                              double S2 , elem_t Z2,double S2_bias, elem_t Z2_bias,double S3, elem_t Z3,
                              double S4, elem_t Z4,elem_t out_feature[batch_size][output_width][out_channels]) {
    /** We folding trained-params of Batch_Normalization layers into Conv1d trained-Weights become trained-Weights,trained-Bias.
     *  We quantized all the trained-Weights,trained-Bias, Input Signals into region int8[-128, 127].
     *  S1,Z1 : the scalar and zeroPoint we found from the previous layer.
     *  S2,Z2 : the scalar and zeroPoint we found from trained-Weights,trained-Bias.
     *  S3,Z3 : the scalar and zeroPoint we found from the result of the conv1d.
     *  S4,Z4 : the scalar and zeroPoint we found from the result of Relu&Clip.
     *  BE-AWARE Relu&Clip :
     *   Because the elements distribute in [Z3, 127] after Relu&clip, not [0, 127].
     *   S4 = (r4_max - r4_min) / (127 - Z3)
     *   Z4 = 127 - (r4_max) / S4
     * **/
    /**reshape kernel**/
    elem_t reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = conv1d_weight[i][j][k];
            }
        }
    }
    bool padding_front_flag = (padding_front !=0) ? true : false;

    size_t padding_shape = input_width + padding_front + padding_back;

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /**reshape & padding input_feature**/
        elem_t reshape_feature[output_width][kernel_size * in_channels];
        size_t padding_front_idx = 0;
        if (padding_front_flag){
            padding_front_idx = 1 * in_channels;
        }
        elem_t padding_feature[1 * padding_shape * in_channels];
        for (int i = 0; i < input_width ; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                padding_feature[padding_front_idx] = in_feature[batch_idx][i][j];
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
        /**gemmini matmul**/
        ////pre-calculation bias
        acc_t total_bias[output_width][out_channels];
        for (int i = 0; i < output_width; ++i) {
            double tmp_bias = 0;
            for (int j = 0; j < out_channels; ++j) {
                for (int k = 0; k < kernel_size * in_channels; ++k) {
                    tmp_bias += (double )((Z2 * Z1 ) - (reshape_kernel[k][j] * Z1) -
                                          (reshape_feature[i][k] * Z2) );
                }
                total_bias[i][j] = (acc_t)rounding(tmp_bias + ((S2_bias / (S2 * S1)) * (conv1d_bias[i][j] - Z2_bias))
                                                   + (Z3 / ((S1 * S2) / S4)));
                tmp_bias = 0;
            }
        }
//        enum tiled_matmul_type_t tiled_matmul_type = WS;
//        elem_t gemmini_result[output_width][out_channels];
//        tiled_matmul_auto(output_width, out_channels,kernel_size*in_channels, (elem_t*)reshape_feature, (elem_t*)reshape_kernel,
//                          (acc_t*)total_bias, (elem_t*)gemmini_result, kernel_size*in_channels, out_channels, out_channels, out_channels,
//                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,(S1 * S2 / S4),0,false,
//                          false,false,false,false,3,tiled_matmul_type);
//        for (int i = 0; i < output_width; ++i) {
//            for (int j = 0; j < out_channels; ++j) {
//                out_feature[batch_idx][0][i][j] = QRelu_Clip(gemmini_result[i][j], Z3, Z4, true);
//            }
//        }
        for (int i = 0; i < output_width; ++i) {
            double tmp_res = 0;
            for (int j = 0; j < out_channels; ++j) {
                for (int k = 0; k < kernel_size * in_channels; ++k) {
                    tmp_res += (double )(reshape_feature[i][k] * reshape_kernel[k][j]);
                }
                out_feature[batch_idx][i][j] = QRelu_Clip(round_near_even((((S1 * S2) / S4 ) * (tmp_res + total_bias[i][j]))),Z3, Z4, true);
//                out_feature[batch_idx][0][i][j] = round_near_even((((S1 * S2) / S4 ) * (tmp_res + total_bias[i][j])));

                tmp_res = 0;
            }
        }
    }
}

static void conv1d_matmul_GemminiRelu(size_t batch_size, size_t input_width, size_t in_channels,
                          const elem_t in_feature[batch_size][input_width][in_channels],
                          size_t kernel_size, size_t out_channels,size_t stride_size,
                          const elem_t conv1d_weight[kernel_size][in_channels][out_channels],
                          size_t padding_front, size_t padding_back, size_t output_width,
                          const acc_t total_bias[batch_size][output_width][out_channels],double downScalar,
                          elem_t Z3, elem_t Z4,elem_t out_feature[batch_size][output_width][out_channels]){
    /**without pre-compute bias**/
    /**reshape kernel**/
    elem_t reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = conv1d_weight[i][j][k];
            }
        }
    }
    bool padding_front_flag = (padding_front !=0) ? true : false;

    size_t padding_shape = input_width + padding_front + padding_back;

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /**reshape & padding input_feature**/
        elem_t reshape_feature[output_width][kernel_size * in_channels];
        size_t padding_front_idx = 0;
        if (padding_front_flag) {
            padding_front_idx = 1 * in_channels;
        }
        elem_t padding_feature[1 * padding_shape * in_channels];
        for (int i = 0; i < input_width; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                padding_feature[padding_front_idx] = in_feature[batch_idx][i][j];
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
        elem_t gemmini_result[output_width][out_channels];
        /*enum tiled_matmul_type_t tiled_matmul_type = WS;
        tiled_matmul_auto2(output_width, out_channels, kernel_size * in_channels, (elem_t *) reshape_feature,
                           (elem_t *) reshape_kernel,
                           (acc_t *) total_bias[batch_idx][0], (elem_t *) gemmini_result, kernel_size * in_channels,
                           out_channels, out_channels, out_channels,
                           MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, RELU_NUM,
                           (float) downScalar, 0, false,
                           false, false, false, false, 3, tiled_matmul_type, Z4);*/
        for (int i = 0; i < output_width; ++i) {
            for (int j = 0; j < out_channels; ++j) {
                out_feature[batch_idx][i][j] = gemmini_result[i][j];
            }
        }
    }
};

static void conv1d_matmul_Gemmini(size_t batch_size, size_t input_width, size_t in_channels,
                          const elem_t in_feature[batch_size][input_width][in_channels],
                          size_t kernel_size, size_t out_channels,size_t stride_size,
                          const elem_t conv1d_weight[kernel_size][in_channels][out_channels],
                          size_t padding_front, size_t padding_back, size_t output_width,
                          const acc_t total_bias[batch_size][output_width][out_channels],double downScalar,
                          elem_t Z3, elem_t Z4,elem_t out_feature[batch_size][output_width][out_channels]){
    /**without pre-compute bias**/
    /**reshape kernel**/
    elem_t reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = conv1d_weight[i][j][k];
            }
        }
    }
    bool padding_front_flag = (padding_front !=0) ? true : false;

    size_t padding_shape = input_width + padding_front + padding_back;

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /**reshape & padding input_feature**/
        elem_t reshape_feature[output_width][kernel_size * in_channels];
        size_t padding_front_idx = 0;
        if (padding_front_flag) {
            padding_front_idx = 1 * in_channels;
        }
        elem_t padding_feature[1 * padding_shape * in_channels];
        for (int i = 0; i < input_width; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                padding_feature[padding_front_idx] = in_feature[batch_idx][i][j];
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
        elem_t gemmini_result[output_width][out_channels];
        /*enum tiled_matmul_type_t tiled_matmul_type = WS;
        tiled_matmul_auto(output_width, out_channels, kernel_size * in_channels, (elem_t *) reshape_feature,
                          (elem_t *) reshape_kernel,
                          (acc_t *) total_bias[batch_idx][0], (elem_t *) gemmini_result, kernel_size * in_channels,
                          out_channels, out_channels, out_channels,
                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, NO_ACTIVATION,
                          (float) downScalar, 0, false,
                          false, false, false, false, 3, tiled_matmul_type);*/

        for (int i = 0; i < output_width; ++i) {
            for (int j = 0; j < out_channels; ++j) {
                out_feature[batch_idx][i][j] = QRelu_Clip(gemmini_result[i][j], Z3, Z4, true);
            }
        }
    }
};

static void conv1d_matmul_cpu(size_t batch_size, size_t input_width, size_t in_channels,
                          const elem_t in_feature[batch_size][input_width][in_channels],
                          size_t kernel_size, size_t out_channels,size_t stride_size,
                          const elem_t conv1d_weight[kernel_size][in_channels][out_channels],
                          size_t padding_front, size_t padding_back, size_t output_width,
                          const acc_t total_bias[batch_size][output_width][out_channels],double downScalar,
                          elem_t Z3, elem_t Z4,elem_t out_feature[batch_size][output_width][out_channels]){
    /**without pre-compute bias**/
    /**reshape kernel**/
    elem_t reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = conv1d_weight[i][j][k];
            }
        }
    }
    bool padding_front_flag = (padding_front !=0) ? true : false;

    size_t padding_shape = input_width + padding_front + padding_back;

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /**reshape & padding input_feature**/
        elem_t reshape_feature[output_width][kernel_size * in_channels];
        size_t padding_front_idx = 0;
        if (padding_front_flag) {
            padding_front_idx = 1 * in_channels;
        }
        elem_t padding_feature[1 * padding_shape * in_channels];
        for (int i = 0; i < input_width; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                padding_feature[padding_front_idx] = in_feature[batch_idx][i][j];
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
        elem_t gemmini_result[output_width][out_channels];
        for (int i = 0; i < output_width; ++i) {
            double tmp_res = 0;
            for (int j = 0; j < out_channels; ++j) {
                for (int k = 0; k < kernel_size * in_channels; ++k) {
                    tmp_res += (double) (reshape_feature[i][k] * reshape_kernel[k][j]);
                }
                out_feature[batch_idx][i][j] = QRelu_Clip(
                        round_near_even(downScalar * (tmp_res + total_bias[batch_idx][i][j])), Z3, Z4, true);
                tmp_res = 0;
            }
        }
    }
};


void plot_result(int batch_size, double y_axis[batch_size]);
static void post_processing(int batch_size, int gesN, elem_t result_matrix[batch_size][gesN],int K){
    /**Top-K Majority Voting Rule
     * for speedup we use the result from dense layer**/
    int max_res1D[batch_size];
    double y_axis[batch_size+2]; //for plot
    for (int i = 0; i < batch_size; ++i) {
        int idx_max = 0;
        double idx_double = 0.0;
        double maxx = 0.0;
        for (int j = 0; j < gesN; ++j) {
            if (result_matrix[i][j] > maxx) {
                maxx = result_matrix[i][j];
                idx_max = j;
                idx_double = j;
            }
        }
        max_res1D[i] = idx_max;
        y_axis[i] = idx_max;
    }

    y_axis[batch_size] = 12, y_axis[batch_size+1] = 0; //for plot boundary

    //plot_result(batch_size, y_axis);
    //find argmax from argmax result
    int count = 0;
    int previous_predict = max_res1D[0];

    int pred_label[batch_size], pred_count[batch_size];
    int idx = 0;
    for (int i = 0; i < batch_size; ++i) {
        if (previous_predict == max_res1D[i]){
            count += 1;
        }else{
            pred_label[idx] = previous_predict;
            pred_count[idx] = count;
            idx += 1;
            count = 1;
        }
        previous_predict = max_res1D[i];
    }
    pred_label[idx] = previous_predict;
    pred_count[idx] = count;
    idx += 1;
//     for (int i = 0; i < idx; ++i) {
//         printf("label: %d\t , count: %d\n", pred_label[i], pred_count[i]);
//     }
    //avoid duplicat result
    int hash_count[gesN]; //recording gestures counts.
    int hash_time[gesN]; //recording gestures appeared moment.
    for (int i = 0; i < gesN; ++i) {
        hash_count[i] = 0;
        hash_time[i] = 0;
    }
    int time = 1;
    for (int i = 0; i < idx; ++i) {
        if (hash_count[pred_label[i]] < pred_count[i]){
            hash_count[pred_label[i]] = pred_count[i];
            hash_time[pred_label[i]] = time;
            time += 1;
        }
    }
    for (int i = 0; i < gesN; ++i) {
        printf("%d\t", i);
    }
    printf("\n");
    for (int i = 0; i < gesN; ++i) {
        printf("%d\t", hash_count[i]);
    }
    printf("\n");
    for (int i = 0; i < gesN; ++i) {
        printf("%d\t", hash_time[i]);
    }
    printf("\n");

    /**Top-kth voting
     * gesN :  0  1  2  3  "4"  5  6  7  "8"  "9"  10  11
     * counts: 0  1  1  0 "41"  0 42  0  "82" "44" 0   0
     * times : 0  8  6  0 "13"  0 10  0  "12" "1 " 0   0
     * step1: find top-k maximum in counts.
     * step2: find appear time from the result of step1.
     * **/
    int step1_ans[K];
    int pre_max = 1000;
    for (int i = 0; i < K; ++i) {
        int max = -1;
        int label = -1;
        for (int j = 0; j < gesN; ++j) {
            if (max < hash_count[j] && hash_count[j] <= pre_max){
                max = hash_count[j];
                label = j;
            }
        }
//         printf("pre_max = %d\n",pre_max);
        pre_max = max ;
        hash_count[label] += max; // dealing with equal problem
        step1_ans[i] = label;
    }
//     for (int i = 0; i < K; ++i) {
//         printf("%d\t", step1_ans[i]);
//     }
//     printf("\n");
    int step2_ans[K];
    int pre_min = 0;
    for (int i = 0; i < K; ++i) {
        int min_time = 100;
        int label = -1;
        for (int j = 0; j < K; ++j) {
            if (min_time > hash_time[step1_ans[j]] && hash_time[step1_ans[j]] > pre_min){
                min_time = hash_time[step1_ans[j]];
                label = step1_ans[j];
            }
        }
        pre_min = min_time;
        step2_ans[i] = label;
    }
    for (int i = 0; i < K; ++i) {
        if (step2_ans[i] == 0){
            if (i != K-1){
                step2_ans[i] = step2_ans[i+1];
            }else{
                step2_ans[i] = step2_ans[i-1];
            }
        }
    }
    printf("Predict Result:\t");
    for (int i = 0; i < K; ++i) {
        printf("%d\t", step2_ans[i]);
    }
    printf("\n");
};

#endif //GEMMINI_PROJECTS_FUNC_H
