// func.h
// Created by sam on 2021/12/22.
////Gemmini only accept to import your custom function in <file.h>,
/// unaccept to import <file.c>.

#ifndef GEMMINI_PROJECTS_FUNC_H
#define GEMMINI_PROJECTS_FUNC_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define elem_t_max 127
#define elem_t_min -128
typedef int8_t elem_t;
typedef int32_t acc_t;

static void block_print(int batch_size, int output_width, int out_channels,
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
    /**rounding to nearest integer with one more constrain even number**/
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

static double Dequantization(acc_t q, double scale, double zeroPoint){
    return scale * (q - zeroPoint);
}
static int32_t Quantization(double r, double scale, double zeroPoint){
    return (int32_t)rounding((r/scale)+zeroPoint);
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
//    for (int i = 0; i < batch_size; ++i) {
//        for (int j = 0; j < in_channels; ++j) {
//            printf("%f ", input_feature[i][j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
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

static elem_t clip(int64_t x){
    elem_t tmp;
    if (x < elem_t_min){
        tmp = elem_t_min;
    }else if (x > elem_t_max){
        tmp = elem_t_max;
    }else{
        tmp=(elem_t)x;
    }
    return tmp;
}

static elem_t QRelu_Clip(int32_t x, elem_t Z4 ,bool use_relu){
    elem_t tmp;
    if (use_relu){
        if (x < (int32_t)Z4){
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

static void Matmul(size_t I, size_t K, size_t J, double matrixA[I][K],const double matrixB[K][J],const double bias[J], double matrixC[I][J]){
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

static void QDense_gemmini(size_t I, size_t K, size_t J,const elem_t matrixA[I][K],const elem_t matrixB[K][J],const acc_t bias[J],
                           elem_t matrixC[I][J],double S1, elem_t Z1,double S2 , elem_t Z2,double S2_bias, elem_t Z2_bias,double S3, elem_t Z3){
    elem_t A[I][K], B[K][J];
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            A[i][k] = clip(matrixA[i][k] - Z1);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < J; ++j) {
            B[k][j] = clip(matrixB[k][j] - Z2);
        }
    }
    /**Gemmini*/
//    enum tiled_matmul_type_t tiled_matmul_type = WS;
//    tiled_matmul_auto(I, J,K, (elem_t*)A, (elem_t*)B,(acc_t*)bias, (elem_t*)matrixC,
//                      K, J, J, J, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,
//                      ((S1 * S2) / S3),0,true,false,false,false,false,3,tiled_matmul_type);
}

static void QDense_cpu(size_t I, size_t K, size_t J, const elem_t matrixA[I][K],const elem_t matrixB[K][J],const acc_t bias[J],
                       elem_t matrixC[I][J],double S1, elem_t Z1,double S2 , elem_t Z2,double S2_bias, elem_t Z2_bias,double S3, elem_t Z3){
    elem_t A[I][K], B[K][J];
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            A[i][k] = clip(matrixA[i][k] - Z1);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < J; ++j) {
            B[k][j] = clip(matrixB[k][j] - Z2);
        }
    }
    /**cpu*/
    for (int i = 0; i < I; ++i) {
        double tmp_res = 0.0;
        for (int j = 0; j < J; ++j) {
            for (int k = 0; k < K; ++k) {
                tmp_res += (double )((A[i][k]) * (B[k][j])) ;
            }
            matrixC[i][j] = QRelu_Clip(round_near_even((((S1 * S2) / S3) * (tmp_res +  bias[j]))), 0, false);
            tmp_res = 0;
        }
    }
}
static void padding_same_gemmini(int batch_size,int input_width,int in_channels,const elem_t feature[batch_size][input_width][in_channels],
                                 int padding_size, int stride,elem_t padd_feature[batch_size][input_width+padding_size][in_channels]){
    int start_idx = 0;
    if (padding_size % 2 == 0 && stride==1){
//        printf("padding case 1\n");
        //padding into front and back with same size.
        start_idx = (int)(padding_size / 2);
    }else{
//        printf("padding case 2\n");
    }
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_width; ++j) {
            for (int k = 0; k < in_channels; ++k) {
                padd_feature[i][j+start_idx][k] = feature[i][j][k];
            }
        }
    }
}

static void conv1d_original_padd(int batch_size,int input_width,int in_channels,const elem_t feature[batch_size][input_width][in_channels],
                                 int kernel_size,int out_channels,int stride_size,const elem_t kernel[kernel_size][in_channels][out_channels],
                                 int output_width, int padding_size,elem_t result[batch_size][output_width][out_channels]){
    int padding_shape = input_width + padding_size;
    elem_t padd_feature[batch_size][padding_shape][in_channels];
    memset(padd_feature, 0, (batch_size*padding_shape*in_channels)*sizeof(elem_t));
    padding_same_gemmini(batch_size, input_width, in_channels, feature, padding_size, stride_size, padd_feature);
    for(int b = 0; b < batch_size; b++){
        int col = 0;
        for(int out = 0; out < out_channels; out++){
            bool stop_flag = false;
            int row = 0;
            for(int i = 0; i < input_width && !stop_flag; i+=stride_size){
                acc_t sum = 0;
                for(int j = 0; j < in_channels; j++){
                    for (int k = 0; k < kernel_size; ++k) {
//                        printf("%d * %d\n", feature[b][i+k][j], kernel[k][j][out]);
                        sum += padd_feature[b][i+k][j] * kernel[k][j][out];
                    }
                }
//                printf("%d ", sum);
                result[b][row][col] = clip(sum);
                row++;
                if (row >= output_width){
                    stop_flag = true;
                }
            }
            col++;
        }
    }
}


static void conv1d_original(int batch_size,int input_width,int in_channels,const elem_t feature[batch_size][input_width][in_channels],
                            int kernel_size,int out_channels,int stride_size,const elem_t kernel[kernel_size][in_channels][out_channels],
                            int padding_front, int padding_back,int output_width,const acc_t bias[batch_size][output_width][out_channels],
                            double downScalar, elem_t Z3, elem_t Z4,elem_t result[batch_size][output_width][out_channels]){
    for(int b = 0; b < batch_size; b++){
        int col = 0;
        for(int out = 0; out < out_channels; out++){
            int row = 0;
            for(int i = 0; i < input_width; i+=stride_size){
                int sum = 0;
                for(int j = 0; j < in_channels; j++){
                    for (int k = 0; k < kernel_size; ++k) {
//                        printf("%d * %d\n", feature[b][i+k][j], kernel[k][j][out]);
                        sum += feature[b][i+k][j] * kernel[k][j][out];
                    }
                }
                result[b][row][col] = QRelu_Clip(round_near_even((sum+bias[b][row][col])*downScalar),Z4, true);
//                printf("\n%d\n", result[b][row][col]);
                row++;
            }
//            printf("\n");
            col++;
        }
    }
}


static void pre_compute_bias(size_t in_channels , size_t kernel_size, size_t output_width, size_t out_channels,
                             const elem_t feature[output_width][kernel_size*in_channels],
                             const elem_t kernel[kernel_size*in_channels][out_channels], const acc_t bias[out_channels],
                             acc_t total_bias[output_width][out_channels], double S1, elem_t Z1,double S2 , elem_t Z2,
                             double S2_bias, elem_t Z2_bias, double S4, elem_t Z4){
    for (int i = 0; i < output_width; ++i) {
        double tmp_bias = 0;
        for (int j = 0; j < out_channels; ++j) {
            for (int k = 0; k < kernel_size * in_channels; ++k) {
                tmp_bias += (double )((Z2 * Z1 ) - (kernel[k][j] * Z1) - (feature[i][k] * Z2) );
            }
            total_bias[i][j] = (acc_t)rounding(tmp_bias + ((S2_bias / (S2 * S1)) * (bias[j] - Z2_bias))
                                               + (Z4 / ((S1 * S2) / S4)));
            tmp_bias = 0;
        }
    }
}

static void conv1d2matmul_cpu(size_t batch_size, size_t input_width, size_t in_channels,
                              const elem_t in_feature[batch_size][input_width][in_channels],
                              size_t kernel_size, size_t out_channels,size_t stride_size,
                              const elem_t conv1d_weight[kernel_size][in_channels][out_channels],
                              size_t output_width,const acc_t conv1d_bias[out_channels],
                              double S1, elem_t Z1,double S2 , elem_t Z2,double S2_bias,elem_t Z2_bias,
                              double S4, elem_t Z4, elem_t out_feature[batch_size][output_width][out_channels]) {
    /** We folding BatchNorm weights into Conv1d weights and Bias.
     *  q4 = M *( matmul((q1 - z1), (q2 - z2)) + (sb / (s1*s2))*(bias - zb))+(z4/(s1 * s2/s4)) )
     *  q4 = q4 < Z4 ? Z4 : q4
     * **/
    /**reshape kernel**/
    elem_t reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = clip(conv1d_weight[i][j][k] - Z2);
            }
        }
    }

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /**reshape input_feature**/
        elem_t reshape_feature[output_width][kernel_size * in_channels];
        size_t reshape_idx = 0;
        elem_t flatten_feature[1*input_width*in_channels];
        for (int i = 0; i < input_width ; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                flatten_feature[reshape_idx] = in_feature[batch_idx][i][j];
                reshape_idx += 1;
            }
        }
        size_t start = 0;
        for (size_t i = 0; i < output_width; ++i) {
            for (int j = 0; j < (kernel_size * in_channels); ++j) {
                reshape_feature[i][j] = clip(flatten_feature[(start + j)] - Z1);
            }
            start += stride_size * in_channels;
        }
        /**cpu matmul**/
        for (int i = 0; i < output_width; ++i) {
            acc_t tmp_res = 0;
            for (int j = 0; j < out_channels; ++j) {
                for (int k = 0; k < kernel_size * in_channels; ++k) {
                    tmp_res += (acc_t )(reshape_feature[i][k] * reshape_kernel[k][j]);
                }
                out_feature[batch_idx][i][j] = QRelu_Clip(round_near_even((((S1 * S2) / S4 ) * (tmp_res + conv1d_bias[j]))), Z4, true);
                tmp_res = 0;
            }
        }
    }
}

static void conv1d2matmul_gemmini(size_t batch_size, size_t input_width, size_t in_channels,
                                  const elem_t in_feature[batch_size][input_width][in_channels],
                                  size_t kernel_size, size_t out_channels,size_t stride_size,
                                  const elem_t conv1d_weight[kernel_size][in_channels][out_channels],
                                  size_t output_width,const acc_t conv1d_bias[out_channels],double S1, elem_t Z1,
                                  double S2 , elem_t Z2,double S2_bias, elem_t Z2_bias,double S4, elem_t Z4,
                                  elem_t out_feature[batch_size][output_width][out_channels]) {
    /** We folding BatchNorm weights into Conv1d weights and Bias.
     *  q4 = M *( matmul((q1 - z1), (q2 - z2)) + (sb / (s1*s2))*(bias - zb))+(z4/(s1 * s2/s4)) )
     *  q4 = q4 < Z4 ? Z4 : q4
     * **/
    /**reshape kernel**/
    elem_t reshape_kernel[kernel_size * in_channels][out_channels];
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0;j < in_channels;++j) {
            for (int k = 0;k < out_channels;++k) {
                reshape_kernel[(i* in_channels)+j][k] = clip(conv1d_weight[i][j][k] - Z2);
            }
        }
    }

    for (size_t batch_idx = 0;batch_idx < batch_size; ++batch_idx) {

        /**reshape input_feature**/
        elem_t reshape_feature[output_width][kernel_size * in_channels];
        size_t reshape_idx = 0;
        elem_t flatten_feature[1*input_width*in_channels];
        for (int i = 0; i < input_width ; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                flatten_feature[reshape_idx] = in_feature[batch_idx][i][j];
                reshape_idx += 1;
            }
        }
        size_t start = 0;
        for (size_t i = 0; i < output_width; ++i) {
            for (int j = 0; j < (kernel_size * in_channels); ++j) {
                reshape_feature[i][j] = clip(flatten_feature[(start + j)] - Z1);
            }
            start += stride_size * in_channels;
        }
        /**gemmini matmul**/
        elem_t gemmini_result[output_width][out_channels];
//        enum tiled_matmul_type_t tiled_matmul_type = WS;
//        tiled_matmul_auto(output_width, out_channels,kernel_size*in_channels, (elem_t*)reshape_feature, (elem_t*)reshape_kernel,
//                          (acc_t*)conv1d_bias, (elem_t*)gemmini_result, kernel_size*in_channels, out_channels, out_channels, out_channels,
//                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,((S1 * S2) / S4),0,true,
//                          false,false,false,false,3,tiled_matmul_type);
        for (int i = 0; i < output_width; ++i) {
            for (int j = 0; j < out_channels; ++j) {
                out_feature[batch_idx][i][j] = QRelu_Clip(gemmini_result[i][j], Z4, true);
            }
        }

    }
}


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
    int pre_max = INT_MAX;
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
        int min_time = INT_MAX;
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
