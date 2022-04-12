// gemmini_custom.h
// Created by sam on 2022/3/14.
// for custom GAP & conv1d & relu

#ifndef GEMMINI_PROJECTS_GEMMINI_CUSTOM_H
#define GEMMINI_PROJECTS_GEMMINI_CUSTOM_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include "include/gemmini.h"
#include "include/gemmini_testutils.h"
//#include "include/gemmini_params.h"
//#include "rocc-software/src/xcustom.h"
#define PE 4

/**gemmini global average pooling*/
static void global_avg(int out_dim, int out_channels, int PE_N, elem_t * A, elem_t * C){
    double scale = 1.0 / out_dim;

    gemmini_mc2_config_ex(1, 0);
    gemmini_global_ld(PE_N);
    gemmini_mc2_config_st(PE_N, 0, scale);
    const uint32_t C_sp_addr_start = 0XC0000000 ;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    int out_rows = (out_channels / (PE_N*PE_N)) + (out_channels % (PE_N*PE_N) != 0);
    int out_rows2 = (out_channels / PE_N) + (out_channels % PE_N != 0);

    //mvin
    for(int row = 0; row < out_dim; row++){
        for(int i = 0; i < out_rows; i++){
            int A_row = ((out_channels / PE_N) - (i * PE_N)) > PE_N ? PE_N : ((out_channels / PE_N) - (i * PE_N) + (out_channels % PE_N != 0));

            if(row == 0){
                gemmini_extended_mvin(A + i * (PE_N*PE_N) + row * out_channels , D_sp_addr_start + i * PE_N , PE, A_row);
            }
            else{
                gemmini_extended_mvin(A + i * (PE_N*PE_N) + row * out_channels , C_sp_addr_start + i * PE_N , PE, A_row);

            }

        }
    }
    //mvout

    for(int j = 0; j < out_rows; j++){
        int orow = ((out_channels / PE_N) - (j * PE_N)) > PE_N ? PE_N : ((out_channels / PE_N) - (j * PE_N) + (out_channels % PE_N != 0));
        uint32_t output = C_sp_addr_start + j * PE_N ;
        gemmini_extended_mvout(C + (j * ((out_channels > (PE_N * PE_N)) ? PE_N *PE_N : 0)) , output, PE_N, orow);
    }

    gemmini_fence();
}

static void mc2_1dconv_global_avg(int batch, int out_dim, int out_channels, int PE_N, elem_t * A, elem_t * C){

    for(int b = 0; b < batch; b++){

        global_avg(out_dim, out_channels, PE_N, A + b * out_dim * out_channels, C + b * out_channels);
    }

}

/**gemmini conv1d*/
static void mc2_conv1d(int dataflow, int act, acc_scale_t scale, elem_t relu_num,  const elem_t * A, const elem_t * B, const acc_t* D, void * C,
                       int PE_N, int in_dim, int stride, int kernel_dim, int in_channels, int batch_size, int out_channels, int out_dim)

{

//    printf("D = %p", D);
    gemmini_mc2_config_ex(dataflow, act);
    gemmini_mc2_config_st(out_channels, act, scale);
    gemmini_mc2_config_ldA(stride * in_channels, 1.0);
    gemmini_mc2_config_ldB(out_channels, 1.0);
    gemmini_mc2_config_ldD(out_channels * sizeof(acc_t), 1.0);
    /****************************************/
    /* gemmini LOOP_WS config*/
    /****************************************/
    //   gemmini_mc2_config_ex(dataflow,act,1 ,0);
    // //gemmini_extended_config_ex(dataflow, act, 0, scale, relu6_shift, 1, a_transpose, b_transpose);
    //     gemmini_mc2_config_st(out_channels, scale);
    //     gemmini_extended3_config_ld(stride * in_channels, 1.0, false, 0);
    //     gemmini_extended3_config_ld(out_channels, 1.0, false, 1);
    //     gemmini_extended3_config_ld(out_channels * sizeof(acc_t), 1.0, 0, 2);

    //Address pre setting
    const uint32_t A_sp_addr_start = 0;
    // const uint32_t B_sp_addr_start = 0X00003000;
    const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - in_channels * in_dim * PE_N * PE_N;
    //if we need Accumulate all output C must store at 0XC0000000
    const uint32_t C_sp_addr_start = 0XC0000000 ;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);

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
            const acc_t * D_dram_addr = D + brow * out_channels  + bcol;
            gemmini_extended_mvin3(D_dram_addr, D_sp_addr + brow, D_col, D_row);
//            printf("D_row = %d, D_col = %d, D_sp_addr = %p, D_dram = %p\n", D_row, D_col, D_sp_addr + brow, D + (bcol / PE_N) * out_dim + brow);
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
                    gemmini_extended_mvin(A + icol + irow * stride * in_channels + i * in_channels , A_sp_addr  , A_col, A_row);
//                    printf("A_row = %d, A_col = %d, A_sp_addr = %p, A_dram = %p\n", A_row, A_col, A_sp_addr, A + icol + irow * stride * in_channels + i * in_channels);
                }
            }
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
                gemmini_extended_mvin2(B + krow * out_channels + i * out_channels * in_channels + kcol, B_sp_addr + krow , B_col, B_row);
//                printf("B_row = %d, B_col = %d, B_sp_addr = %p, B_dram = %p\n", B_row, B_col, B_sp_addr + krow, B + krow + i * out_channels * in_channels + kcol);
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
                    gemmini_extended_preload(pre_sp, out_sp_addr, cb_col, ab_col_row, cb_col, ac_row);
                    if(new_weight){
                        gemmini_extended_compute_preloaded(compute_sp, GARBAGE_ADDR, ab_col_row, ac_row, PE_N, PE_N);
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
            void * const C_dram_addr = (int8_t*)C + crow * out_channels + ccol;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_col, C_row);
//            printf("C_row = %d, C_col = %d, C_sp_addr = %p, C_dram = %p\n", C_row, C_col, C_sp_addr, C + crow * out_channels + ccol);
        }
    }

    // printf("finish!\n");
    gemmini_fence();
    // printMatrix((elem_t*)C);

}

static void batch_forloop(int dataflow, int act, acc_scale_t scale, elem_t relu_num, const elem_t * A, const elem_t * B,
                          const acc_t * D, void * C, int PE_N, int in_dim, int stride, int kernel_dim, int in_channels,
                          int batch_size, int out_channels, int out_dim){


    for(int b = 0; b < batch_size; b++){
        mc2_conv1d(dataflow, act, scale, relu_num, A + b * in_dim * in_channels, B, D+ b * out_dim * out_channels,
                   C + b * out_dim * out_channels, PE, in_dim, stride, kernel_dim, in_channels, batch_size,
                   out_channels, out_dim);

    }
}

static void batch_forloop2(int dataflow, int act, acc_scale_t scale, elem_t relu_num,int in_dim, int stride, int kernel_dim, int in_channels, int batch_size, int out_channels, int out_dim, const elem_t A[batch_size][in_dim][in_channels], const elem_t B[kernel_dim * in_channels][out_channels],
                           const acc_t D[batch_size][out_dim][out_channels], void * C, int PE_N)
{


    for(int b = 0; b < batch_size; b++){
        mc2_conv1d(dataflow, act, scale, relu_num, (elem_t*)A + (b * in_dim * in_channels), (elem_t*)B, (acc_t*)D + (b * out_dim * out_channels),
                   C + b * out_dim * out_channels, PE, in_dim, stride, kernel_dim, in_channels, batch_size,
                   out_channels, out_dim);
        gemmini_fence();

    }
    gemmini_fence();
}

/**custom relu*/
// static void sp_tiled_matmul_ws2(const elem_t * A, const elem_t * B,
//                                 const void * D, void * C,
//                                 scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//                                 size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
//                                 size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
//                                 bool a_transpose, bool b_transpose,
//                                 bool full_C, bool low_D,
//                                 bool no_bias, bool repeating_bias,
//                                 uint8_t weightA, elem_t relu_num) {

//     // Combined loop
//     gemmini_loop_ws(I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
//                     A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
//                     a_transpose, b_transpose,
//                     full_C, low_D, !no_bias || D == NULL,
//                     weightA);
// }

// static void tiled_matmul_outer2(size_t dim_I, size_t dim_J, size_t dim_K,
//                                 const elem_t* A, const elem_t* B,
//                                 const void * D, void * C,
//                                 size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
//                                 scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//                                 size_t tile_I, size_t tile_J, size_t tile_K,
//                                 int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
//                                 bool a_transpose, bool b_transpose,
//                                 bool full_C, bool low_D,
//                                 uint8_t weightA,
//                                 int dataflow, elem_t relu_num) {


//     const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
//     const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
//     const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

//     const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
//     const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
//     const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

//     // These lines here are supposed to help us deal with when the dimensions of
//     // the systolic array aren't divisible by the tiling factors
//     const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
//     const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
//     const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

//     // These lines are supposed to figure out how much padding the hardware is
//     // supposed to add for the final tile
//     const size_t padding_I = dim_I_padded - dim_I;
//     const size_t padding_J = dim_J_padded - dim_J;
//     const size_t padding_K = dim_K_padded - dim_K;

//     const bool no_bias = D == NULL;

//     if (no_bias) {
//         D = (void*) 1; // Dummy address which isn't NULL
//     }

//     const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
//     const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
//     gemmini_mc2_config_ex(dataflow, act, 1, relu_num);
//     //gemmini_extended_config_ex(dataflow, act, 0, scale, relu6_shift, 1, a_transpose, b_transpose);
//     gemmini_mc2_config_st(stride_C * sizeof_C, scale);
//     gemmini_extended3_config_ld(stride_A * sizeof(elem_t), A_scale_factor, false, 0);
//     gemmini_extended3_config_ld(stride_B * sizeof(elem_t), B_scale_factor, false, 1)
//     gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);

//     void (*inner)(const elem_t *, const elem_t *, const void *, void *,
//                   scale_t, scale_t, scale_acc_t,
//                   size_t, size_t, size_t, size_t, size_t, size_t,
//                   size_t, size_t, size_t, size_t,
//                   bool, bool,
//                   bool, bool,
//                   bool, bool,
//                   uint8_t, elem_t);


//     inner = &sp_tiled_matmul_ws2;


//     for (size_t i0 = 0; i0 < I0; i0++)
//         for (size_t j0 = 0; j0 < J0; j0++)
//             for (size_t k0 = 0; k0 < K0; k0++) {

//                 const void * pre;
//                 if (k0 != 0) {
//                     pre = NULL;
//                 } else {
//                     size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
//                     // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
//                     pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
//                 }

//                 void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

//                 const size_t I = i0 < I0-1 ? tile_I : last_I;
//                 const size_t J = j0 < J0-1 ? tile_J : last_J;
//                 const size_t K = k0 < K0-1 ? tile_K : last_K;

//                 const size_t pad_I = i0 == I0-1 ? padding_I : 0;
//                 const size_t pad_J = j0 == J0-1 ? padding_J : 0;
//                 const size_t pad_K = k0 == K0-1 ? padding_K : 0;

//                 const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
//                                                : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

//                 const elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
//                                                : (B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

//                 (*inner)(a, b, pre, out,
//                          A_scale_factor, B_scale_factor, D_scale_factor,
//                          I, J, K,
//                          pad_I, pad_J, pad_K,
//                          stride_A, stride_B, stride_D, stride_C,
//                          a_transpose, b_transpose,
//                          full_C, low_D,
//                          no_bias, repeating_bias,
//                          weightA, relu_num);
//             }

//     gemmini_fence();
// }

// #undef GEMMINI_SCALE

// static void tiled_matmul2(size_t dim_I, size_t dim_J, size_t dim_K,
//                           const elem_t* A, const elem_t* B,
//                           const void * D, void* C,
//                           size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
//                           scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//                           int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
//                           size_t tile_I, size_t tile_J, size_t tile_K,
//                           bool transpose_A, bool transpose_B,
//                           bool full_C, bool low_D,
//                           uint8_t weightA,
//                           enum tiled_matmul_type_t tiled_matmul_type, elem_t relu_num) {

// #ifdef GEMMINI_ASSERTIONS
//     // Make sure that the tiling factors make sense
//     if (tile_I <= 0) {
//         printf("tile_I is non-positive\n");
//         exit(1);
//     } else if (tile_J <= 0) {
//         printf("tile_J is non-positive\n");
//         exit(1);
//     } else if (tile_K <= 0) {
//         printf("tile_K is non-positive\n");
//         exit(1);
//     }

//     const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
//     const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
//     const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

//     if (tile_I * DIM > dim_I_padded) {
//         printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
//         exit(1);
//     } else if (tile_J * DIM > dim_J_padded) {
//         printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
//         exit(1);
//     } else if (tile_K * DIM > dim_K_padded) {
//         printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
//         exit(1);
//     }

//     const bool double_buffered = tiled_matmul_type == WS;

//     const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
//                                    BANK_NUM * BANK_ROWS;
//     const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

//     const size_t total_spad_rows =
//             (tile_I * tile_K * DIM) +   // Rows to store A
//             (tile_K * tile_J * DIM);    // Rows to store B

//     if (total_spad_rows > total_spad_size) {
//         printf("Not enough space in scratchpad to store A and B matrices\n");
//         exit(1);
//     }

//     const size_t total_acc_rows =
//             tile_I * tile_J * DIM;      // Rows to store C

//     if (total_acc_rows > total_acc_size) {
//         printf("Not enough space in accumulator to store C\n");
//         exit(1);
//     }

//     if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
//         printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
//         exit(1);
//     }

//     char matmul_type_str[][4] = {"OS", "WS", "CPU"};

//     // Check if transpose options are correct
//     if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
//         (tiled_matmul_type == WS && transpose_A && transpose_B)) {
//         printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
//         exit(1);
//     }

//     // Check if full_C options are correct
//     if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
//         (tiled_matmul_type == OS && low_D)) {
//         printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
//     }
// #endif

//     // Run a tiled matrix multiplication on either Gemmini or the CPU
//     if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
//         tiled_matmul_outer2(dim_I, dim_J, dim_K,
//                             A, B, D, C,
//                             stride_A, stride_B, stride_D, stride_C,
//                             A_scale_factor, B_scale_factor, D_scale_factor,
//                             tile_I, tile_J, tile_K,
//                             act, scale, relu6_shift, repeating_bias,
//                             transpose_A, transpose_B,
//                             full_C, low_D,
//                             weightA,
//                             (int)tiled_matmul_type, relu_num);
//     } else /*if (tiled_matmul_type == CPU)*/ {
//         matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
//                    A, B, (const acc_t*) D, (elem_t*)C,
//                    stride_A, stride_B, stride_D, stride_C,
//                    A_scale_factor, B_scale_factor, D_scale_factor,
//                    act, scale, relu6_shift, repeating_bias);
//     }
// }


// static void tiled_matmul_auto2(size_t dim_I, size_t dim_J, size_t dim_K,
//                                const elem_t* A, const elem_t* B,
//                                const void * D, void * C,
//                                size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
//                                scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
//                                int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
//                                bool transpose_A, bool transpose_B,
//                                bool full_C, bool low_D,
//                                uint8_t weightA,
//                                enum tiled_matmul_type_t tiled_matmul_type, elem_t relu_num) {


// #define partition_rows (BANK_NUM * BANK_ROWS / 2)
// #define mats_in_partition (partition_rows / DIM)
// #define mats_in_acc (ACC_ROWS / DIM)
// #define max_tile_i_j ((size_t)sqrt(mats_in_acc))
// #define max_tile_k (mats_in_partition / max_tile_i_j)

//     // "db_" means "double-buffered"
// #define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
// #define db_mats_in_partition (db_partition_rows / DIM)
// #define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
// #define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
// #define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

//     const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
//     const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
//     const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

//     const bool double_buffered = tiled_matmul_type == WS;

//     const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
//                                  BANK_NUM * BANK_ROWS;
//     const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

//     size_t tile_I, tile_J, tile_K;

//     if (double_buffered) {
//         tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
//         tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
//         tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
//     } else {
//         tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
//         tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
//         tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
//     }

//     // Fill scratchpad as much as possible
//     while (true) {
//         bool increased = false;

//         if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
//             tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
//             (tile_J+1) * DIM <= dim_J_padded) {
//             tile_J++;
//             increased = true;
//         }

//         if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
//             tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
//             (tile_I+1) * DIM <= dim_I_padded) {
//             tile_I++;
//             increased = true;
//         }

//         if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
//             (tile_K+1) * DIM <= dim_K_padded) {
//             tile_K++;
//             increased = true;
//         }

//         if (!increased)
//             break;
//     }

//     /*
//     const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
//     const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

//     printf("tile_I: %d\n", tile_I);
//     printf("tile_J: %d\n", tile_J);
//     printf("tile_K: %d\n\n", tile_J);

//     printf("spad_rows: %d\n", spad_rows);
//     printf("acc_rows: %d\n\n", acc_rows);

//     printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
//     printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);
//     */

//     tiled_matmul2(dim_I, dim_J, dim_K,
//                   A, B, D, C,
//                   stride_A, stride_B, stride_D, stride_C,
//                   A_scale_factor, B_scale_factor, D_scale_factor,
//                   act, scale, relu6_shift, repeating_bias,
//                   tile_I, tile_J, tile_K,
//                   transpose_A, transpose_B,
//                   full_C, low_D,
//                   weightA,
//                   tiled_matmul_type, relu_num);

// #undef partition_rows
// #undef mats_in_partition
// #undef mats_in_acc
// #undef max_tile_i_j
// #undef max_tile_k
// }
#endif //GEMMINI_PROJECTS_GEMMINI_CUSTOM_H
