/*Modified function */
/*PE8*8 Num of data = 16 */
/*To compare with original function: tiled_matmul_nn*/
/*To calculate Matrix multiple*/

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/func.h"
#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/gemmini_params.h"
#include "include/Qgesture_signals.h"
//#include "include/Qpairnet_params_trans.h"
#include "include/Qpairnet_mc2conv1d_params.h"
#define PE 16
// #include "include/test.h"

void test_mc2_1dconv(int dataflow, int act, acc_scale_t scale, size_t relu6_shift,  const elem_t * A, const elem_t * B, const acc_t* D, void * C,
                     int PE_N, int in_dim, int stride, int kernel_dim, int in_channels, int batch_size, int out_channels, int out_dim)

{
//    printf("D = %p", D);

    gemmini_mc2_config_ex(dataflow, act, scale, relu6_shift);
    gemmini_mc2_config_st(out_channels);
    gemmini_mc2_config_ldA(stride * in_channels, DIM);
    gemmini_mc2_config_ldB(out_channels, DIM);
    gemmini_extended3_config_ld(out_channels * sizeof(acc_t), 1, 0, 2);

    //Address pre setting
    const uint32_t A_sp_addr_start = 0;
    const uint64_t B_sp_addr_start = 0X00002000 ;
    //if we need Accumulate all output C must store at 0XC0000000
    const uint32_t C_sp_addr_start = 0XC0000000 ;
    const uint32_t D_sp_addr_start = 0X80000000 ;


    //mvin bias
//    printf("mvin bias\n");
    for(int bcol =0; bcol < out_channels; bcol += PE_N){
        const int D_col = out_channels - bcol > PE_N ? PE_N : out_channels - bcol;
        const uint32_t D_sp_addr = D_sp_addr_start + (bcol / PE_N) * out_dim;
        // const uint64_t D_dram =(acc_t)D + bcol;
        // const uint64_t D_dram = (uint64_t)D + bcol * sizeof(acc_t);
//        printf("h1\n");
//        printf("D_col = %d\n",D_col);
        for (int brow = 0; brow < out_dim; brow += PE_N){
            const int D_row = out_dim - brow > PE_N ? PE_N : (out_dim - brow) > 0 ? (out_dim - brow) : 0;
//            printf("D_row = %d\n",D_row);
            gemmini_extended_mvin3(D + brow * out_channels  + bcol , D_sp_addr + brow, D_col, D_row);
//            printf("D_row = %d, D_col = %d, D_sp_addr = %p, D_dram = %p\n", D_row, D_col, D_sp_addr + brow, D + (bcol / PE_N) * out_dim + brow);
        }
    }

    //mvin weight
//    printf("mvin weight\n");
    for(int i = 0; i < kernel_dim; i++){
        for(int kcol = 0; kcol < out_channels; kcol += PE_N){
            const int B_col = out_channels - kcol > PE_N ? PE_N : out_channels - kcol;
            const uint32_t B_sp_addr = B_sp_addr_start + (kcol / PE_N) * in_channels + i * in_channels * ((out_channels / PE_N) + (out_channels % PE_N != 0));
            for(int krow = 0; krow < in_channels; krow += PE_N){

                const int B_row = in_channels - krow > PE_N ? PE_N : in_channels - krow;
                gemmini_mc2_mvin2(B + krow * out_channels + i * out_channels * in_channels + kcol, B_sp_addr + krow , B_col, B_row);
//                printf("B_row = %d, B_col = %d, B_sp_addr = %p, B_dram = %p\n", B_row, B_col, B_sp_addr + krow, B + krow + i * out_channels * in_channels + kcol);
            }
        }
    }

    //mvin input
//    printf("mvin input\n");
    for(int i = 0; i < kernel_dim; i++){
        for(int icol = 0; icol < in_channels; icol += PE_N){
            const int A_col = in_channels - icol > PE_N ? PE_N : in_channels - icol;
            for(int irow = 0; irow < in_dim; irow += PE_N){
                const uint32_t A_sp_addr = A_sp_addr_start + irow + (icol / PE_N) * out_dim + i * out_dim * (in_channels / PE_N + (in_channels % PE_N !=0));
                const int A_row = (in_dim / stride) - irow > PE_N ? PE_N : ((out_dim - irow) > 0 ? (out_dim - irow) : 0);
                // const int A_row = (in_dim / stride) - irow > PE_N ? (((in_dim / stride) - PE_N) > PE_N ? PE_N : out_dim - irow) : ((out_dim - irow) > 0 ? (out_dim - irow) : 0);
                if(A_row && A_col != 0){
                    gemmini_mc2_mvin(A + icol + irow * stride * in_channels + i * in_channels , A_sp_addr  , A_col, A_row);
//                    printf("A_row = %d, A_col = %d, A_sp_addr = %p, A_dram = %p\n", A_row, A_col, A_sp_addr, A + icol + irow * stride * in_channels + i * in_channels);
                }
            }
        }
    }


    //compute
//    printf("compute\n");
    for(int i = 0; i < kernel_dim; i++){
        for(int ocol = 0; ocol < out_channels; ocol += PE_N){
            const int cb_col = out_channels - ocol > PE_N ? PE_N : out_channels - ocol;
            // printf("cb_col= %d\n", cb_col);
            for(int ich = 0; ich < in_channels; ich += PE_N){
                bool new_weight = true;
                const int ab_col_row = in_channels - ich > PE_N ? PE_N : in_channels - ich;
                // printf("ab_col_row= %d\n", ab_col_row);
                const uint32_t pre_sp = B_sp_addr_start + ich + (ocol / PE_N) * in_channels + i * ((out_channels / PE_N) + (out_channels % PE_N != 0)) * in_channels;
                for(int orow = 0; orow < out_dim; orow += PE_N){

                    const int ac_row = out_dim - orow > PE_N ? PE_N : out_dim - orow;
                    // printf("ac_row= %d\n", ac_row);
                    const uint32_t out_sp_addr = C_sp_addr_start + (ocol / PE_N) * out_dim + orow;
                    // const uint32_t compute_sp = A_sp_addr_start + orow;
                    const uint32_t compute_sp = A_sp_addr_start + orow + (ich / PE_N) * out_dim  + i * out_dim * ((in_channels / PE_N) + (in_channels % PE_N !=0));
                    gemmini_mc2_preload(pre_sp, out_sp_addr, cb_col, ab_col_row, cb_col, ac_row);
                    if(new_weight){
                        gemmini_mc2_compute(compute_sp, GARBAGE_ADDR, ab_col_row, ac_row, PE_N, PE_N);
                    }
                    else{
                        gemmini_extended_compute_accumulated(compute_sp, GARBAGE_ADDR, ab_col_row, ac_row, PE_N, PE_N);
                    }
                    new_weight = false;
//                    printf("brow = %d, bcol = %d, B_sp_addr = %p, output=%p\n", ab_col_row, cb_col, pre_sp, out_sp_addr);
//                    printf("arow = %d, acol = %d, A_sp_addr = %p\n", ac_row, ab_col_row, compute_sp);
                }
            }
        }
    }

    //mvout output
    // printf("mvout\n");
    for(int ccol = 0; ccol < out_channels; ccol+=PE_N){
        const int C_col = out_channels - ccol > PE_N ? PE_N : out_channels - ccol;
        for(int crow = 0; crow < out_dim; crow+=PE_N){
            const uint32_t C_sp_addr = C_sp_addr_start + crow + (ccol / PE_N) * out_dim;
            const int C_row = out_dim - crow > PE_N ? PE_N : out_dim - crow;
            gemmini_mc2_mvout(C + crow * out_channels + ccol, C_sp_addr, C_col, C_row);
//            printf("C_row = %d, C_col = %d, C_sp_addr = %p, C_dram = %p\n", C_row, C_col, C_sp_addr, C + crow * out_channels + ccol);
        }
    }



    // printf("finish!\n");


    gemmini_fence();

}

void batch_forloop(int dataflow, int act, acc_scale_t scale, size_t relu6_shift, const elem_t * A, const elem_t * B,
                   const acc_t * D, void * C, int PE_N, int in_dim, int stride, int kernel_dim, int in_channels,
                   int batch_size, int out_channels, int out_dim){

    for(int b = 0; b < batch_size; b++){
        test_mc2_1dconv(dataflow, NO_ACTIVATION, scale, 0, A + b * in_dim * in_channels, B, D+ b * out_dim * out_channels,
                        C + b * out_dim * out_channels, PE, in_dim, stride, kernel_dim, in_channels, batch_size,
                        out_channels, out_dim);
    }
}

void Dense(int I, int K, int J, const elem_t matrixA[I][K],const elem_t matrixB[K][J],const acc_t bias[I][J],
           elem_t matrixC[I][J], float downScalar){

    enum tiled_matmul_type_t tiled_matmul_type = WS;
    tiled_matmul_auto(I, J, K, (elem_t*)matrixA, (elem_t*)matrixB,(acc_t*)bias, (elem_t*)matrixC,
                      K, J, J, J, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,NO_ACTIVATION,
                      downScalar,0,false,false,false,false,false,3,tiled_matmul_type);
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            matrixC[i][j] = QRelu_Clip(matrixC[i][j], 0, 0, false);
        }
    }
//    for (int i = 0; i < I; ++i) {
//        double tmp_res = 0.0;
//        for (int j = 0; j < J; ++j) {
//            for (int k = 0; k < K; ++k) {
//                tmp_res += (double )(matrixA[i][k] * matrixB[k][j]) ;
//            }
//            matrixC[i][j] = QRelu_Clip(round_near_even((downScalar * (tmp_res + bias[i][j]))),0, 0, false);
//            tmp_res = 0;
//        }
//    }
}

void Relu_Clip(int batch_size, int out_dim, int out_channels, elem_t C[batch_size][out_dim][out_channels],
                elem_t z3, elem_t z4){
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            for (int k = 0; k < out_channels; ++k) {
                C[i][j][k] = QRelu_Clip(round_near_even(C[i][j][k]), z3, z4, true);
            }
        }
    }
}
 void GAP(int batch_size, int input_width, int in_channels, elem_t input_feature[batch_size][input_width][in_channels],
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


void test_mc2_1dconv_global_avg(int batch, int out_dim, int out_channels, int PE_N, const elem_t * A, const elem_t * C) {
    double scale = 1.0 / out_dim;
    gemmini_mc2_config_ex(WS, NO_ACTIVATION, scale, 0);
// gemmini_extended4_config_ld(PE_N*sizeof(elem_t), MVIN_SCALE_IDENTITY, true, 1, 0);
    gemmini_extended3_config_ld(PE_N * sizeof(elem_t), MVIN_SCALE_IDENTITY, true, 0);
// gemmini_mc2_config_ldA(out_channels*sizeof(elem_t), DIM);
    gemmini_mc2_config_st(PE_N);
    int out_rows = (out_channels / (PE_N*PE_N)) + (out_channels % PE_N != 0) ;
    const uint32_t C_sp_addr_start = 0XC0000000;
    for(int b = 0; b < batch; b++){
//mvin
        for(int row = 0; row < out_dim; row ++){
            int A_row = (out_channels / PE_N) + (out_channels % PE_N != 0);
            if(A_row <= PE_N){
                gemmini_mc2_mvin(A + row * out_channels + b * out_dim * out_channels, C_sp_addr_start +100+ b * A_row , PE, A_row);
// printf("A = %p, C = %p , A_row = %d\n" , A + row, C_sp_addr_start , A_row);
            }
            else{
                for(int i = 0; i < out_rows; i++){
                    int A_row2 = (out_channels - (i * PE_N*PE_N)) > PE_N * PE_N ? PE_N : ((out_channels - (i * PE_N*PE_N)) / PE_N ) + ((out_channels - (i * PE_N*PE_N))%PE_N!=0) ;
                    gemmini_mc2_mvin(A + row * out_channels + b * out_dim * out_channels, C_sp_addr_start+100 + A_row *( b + (i * PE_N)) , PE, A_row2);
                }
            }
        }
//mvout
        int out_row = (out_channels / PE_N) > PE_N ? PE_N : (out_channels / PE_N) + (out_channels % PE_N !=0);
        uint32_t output = C_sp_addr_start + (out_channels > (PE_N * PE_N) ? PE_N : 0) + b * out_row+100;
        if(out_row <= PE_N){
            gemmini_mc2_mvout(C + ((out_channels > (PE_N * PE_N)) ? PE_N : 0) + b * out_channels, output, PE_N, out_row);
// printf("output = %p, C = %p , C_col = %d, C_row = %d\n" , output, C + (out_channels > PE_N * PE_N ? PE_N : 0), out_col, out_row);
        }
        else{
            for(int i = 0; i < out_rows; i++){
                int out_row2 = (out_channels - (i * PE_N*PE_N)) > PE_N * PE_N ? PE_N : ((out_channels - (i * PE_N*PE_N)) / PE_N ) + ((out_channels - (i * PE_N*PE_N))%PE_N!=0) ;
                gemmini_mc2_mvout(C + i * PE_N + b * out_channels, output + i * PE_N, PE_N, out_row2);
            }
        }
    }
    gemmini_fence();
}

int main () {
    /*****PairNet Quantized*****/
    printf("PairNet conv1d with Gemmini\n");
    printf("Input Gesture : %s \n", TRUE_LABEL);
    int gesN = GESN;
    gemmini_flush(0);
    uint64_t start, end, cost;
    enum tiled_matmul_type_t tiled_matmul_type = WS;
    start = read_cycles();
    //1st layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_1, 0, (elem_t *)gesture_signals, (elem_t *)QConv_BN_1,
                  (acc_t *)QConv_BN_bias_1, QConv_BN_1_out, PE, QConv_BN_1_params.input_width, QConv_BN_1_params.stride_size,
                  QConv_BN_1_params.kernel_size, QConv_BN_1_params.in_channels, QConv_BN_1_params.batch_size,
                  QConv_BN_1_params.out_channels, QConv_BN_1_params.output_width);
    Relu_Clip(QConv_BN_1_params.batch_size, QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels, QConv_BN_1_out,
              z3_1, z4_1);
    //block_print1(QConv_BN_1_params.batch_size,QConv_BN_1_params.output_width, QConv_BN_1_params.out_channels,QConv_BN_1_out);

    //2nd layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_2, 0, (elem_t *)QConv_BN_1_out, (elem_t *)QConv_BN_2,
                  (acc_t *)QConv_BN_bias_2, QConv_BN_2_out, PE, QConv_BN_2_params.input_width, QConv_BN_2_params.stride_size,
                  QConv_BN_2_params.kernel_size, QConv_BN_2_params.in_channels, QConv_BN_2_params.batch_size,
                  QConv_BN_2_params.out_channels, QConv_BN_2_params.output_width);

    Relu_Clip(QConv_BN_2_params.batch_size, QConv_BN_2_params.output_width, QConv_BN_2_params.out_channels, QConv_BN_2_out,
              z3_2, z4_2);

    //3rd layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_3, 0, (elem_t *)QConv_BN_2_out, (elem_t *)QConv_BN_3,
                  (acc_t *)QConv_BN_bias_3, QConv_BN_3_out, PE, QConv_BN_3_params.input_width, QConv_BN_3_params.stride_size,
                  QConv_BN_3_params.kernel_size, QConv_BN_3_params.in_channels, QConv_BN_3_params.batch_size,
                  QConv_BN_3_params.out_channels, QConv_BN_3_params.output_width);
    Relu_Clip(QConv_BN_3_params.batch_size, QConv_BN_3_params.output_width, QConv_BN_3_params.out_channels, QConv_BN_3_out,
              z3_3, z4_3);
    //4th layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_4, 0, (elem_t *)QConv_BN_3_out, (elem_t *)QConv_BN_4,
                  (acc_t *)QConv_BN_bias_4, QConv_BN_4_out, PE, QConv_BN_4_params.input_width, QConv_BN_4_params.stride_size,
                  QConv_BN_4_params.kernel_size, QConv_BN_4_params.in_channels, QConv_BN_4_params.batch_size,
                  QConv_BN_4_params.out_channels, QConv_BN_4_params.output_width);
    Relu_Clip(QConv_BN_4_params.batch_size, QConv_BN_4_params.output_width, QConv_BN_4_params.out_channels, QConv_BN_4_out,
              z3_4, z4_4);
    //5th layer
    batch_forloop(tiled_matmul_type, NO_ACTIVATION, (float )downScalar_5, 0, (elem_t *)QConv_BN_4_out, (elem_t *)QConv_BN_5,
                  (acc_t *)QConv_BN_bias_5, QConv_BN_5_out, PE, QConv_BN_5_params.input_width, QConv_BN_5_params.stride_size,
                  QConv_BN_5_params.kernel_size, QConv_BN_5_params.in_channels, QConv_BN_5_params.batch_size,
                  QConv_BN_5_params.out_channels, QConv_BN_5_params.output_width);
    Relu_Clip(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels, QConv_BN_5_out,
              z3_5, z4_5);
//    block_print1(QConv_BN_5_params.batch_size,QConv_BN_5_params.output_width, QConv_BN_5_params.out_channels,QConv_BN_5_out);

    // test_mc2_1dconv(tiled_matmul_type, NO_ACTIVATION, 1, 0, l1_input, l1_weight, l1_bias, l1_output, 8, 20, 1, 2, 10, 1, 10, 19);
    test_mc2_1dconv_global_avg(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels,PE,(elem_t*)QConv_BN_5_out, (elem_t*)QGap_out);
    //GAP(QConv_BN_5_params.batch_size, QConv_BN_5_params.output_width,QConv_BN_5_params.out_channels, QConv_BN_5_out, QGap_out);
    printf("QGAP\n");
//    for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
//        for (int j = 0; j < QConv_BN_5_params.out_channels; ++j) {
//            printf("%d\t", QGap_out[i][j]);
//        }
//        printf("\n");
//    }
    Dense(QConv_BN_5_params.batch_size,QConv_BN_5_params.in_channels, gesN, QGap_out, QDense_params, QDense_bias, QDense_out,
          (float )downScalar_dense);
//    for (int i = 0; i < QConv_BN_5_params.batch_size; ++i) {
//        for (int j = 0; j < gesN; ++j) {
//            printf("%d\t", QDense_out[i][j]);
//        }
//        printf("\n");
//    }
    QSoftMax(QConv_BN_5_params.batch_size, gesN, QDense_out,deq_softmax_out,s3_dense, z3_dense);
    post_processing(QConv_BN_5_params.batch_size, gesN, deq_softmax_out,LEN_LABLE);
    end = read_cycles();
    cost = end - start;
    printf("spent time: %lu\n", cost);
    printf("\n");
    printf("SUCCESS\n");
    exit(0);
}
