// gemmini_custom.h
// Created by sam on 2022/3/14.
// for custom GAP & Conv1d

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
#define PE 8

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
                       int PE_N, int in_dim, int stride, int kernel_dim, int in_channels, int out_channels, int out_dim)

{


    // Compute Method Cofigure
    gemmini_config_ex(dataflow, 0, 0, 0);

    //Load Configure
    /*load input configure*/
    gemmini_extended3_config_ld(stride * in_channels, 1.0, 0, 0);
    /*load weight configure*/
    gemmini_extended3_config_ld(out_channels, 1.0, 0, 1);
    /*load bias configure*/
    gemmini_extended3_config_ld(out_channels*sizeof(acc_t), 1.0, 0, 2);

    //Store Configure
    gemmini_extended_config_st(out_channels, 0, scale);

    //Address pre setting
    const uint32_t A_sp_addr_start = 0;
    // const uint32_t B_sp_addr_start = 0X00003000;
    const uint32_t B_sp_addr_start = 0X00000400;
    //if we need Accumulate all output C must store at 0XC0000000
    const uint32_t C_sp_addr_start = 0XC0000000 ;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);

    int test_orow = out_dim / PE_N + (out_dim % PE_N !=0);
    int test_irow = in_dim / PE_N + (in_dim % PE_N !=0);
    int test_ich = in_channels / PE_N + (in_channels % PE_N !=0);
    int test_och = out_channels / PE_N + (out_channels % PE_N!=0);

    for(int bcol =0; bcol < out_channels; bcol += PE_N){
        int D_col = out_channels - bcol > PE_N ? PE_N : out_channels - bcol;
        uint32_t D_sp_addr = D_sp_addr_start + (bcol / PE_N) * out_dim;
        for (int brow = 0; brow < test_orow; brow ++){
            int D_row = out_dim - brow*PE_N > PE_N ? PE_N : (out_dim - brow*PE_N) > 0 ? (out_dim - brow * PE_N) : 0;
            acc_t * D_dram_addr = D + bcol;
            gemmini_extended_mvin3(D_dram_addr, D_sp_addr + brow * PE_N, D_col, D_row);

        }
    }


    for(int i = 0; i < kernel_dim; i++){
        for(int icol = 0; icol < in_channels; icol += PE_N){
            int A_col = in_channels - icol > PE_N ? PE_N : in_channels - icol;
            for(int irow = 0; irow < in_dim; irow += PE_N){
                uint32_t A_sp_addr = A_sp_addr_start + irow + (icol / PE_N) * out_dim + i * out_dim * (in_channels / PE_N + (in_channels % PE_N !=0));
                int A_row = (in_dim / stride) - irow > PE_N ? PE_N : ((out_dim - irow) > 0 ? (out_dim - irow) : 0);
                if(A_row && A_col != 0){
                    gemmini_extended_mvin(A + icol + irow * stride * in_channels + i * in_channels , A_sp_addr  , A_col, A_row);
                }
            }
        }
    }


    //compute
    for(int i = 0; i < kernel_dim; i++){
        for(int ocol = 0; ocol < test_och; ocol ++){
            int cb_col = out_channels - ocol*PE_N > PE_N ? PE_N : out_channels - ocol*PE_N;
            uint32_t B_sp_addr = B_sp_addr_start + ocol * in_channels + i * in_channels * ((out_channels / PE_N) + (out_channels % PE_N != 0));
            for(int ich = 0; ich < test_ich; ich ++){
                int ab_col_row = in_channels - ich*PE_N > PE_N ? PE_N : in_channels - ich * PE_N;
                gemmini_extended_mvin2(B + ich*PE_N * out_channels + i * out_channels * in_channels + ocol * PE_N, B_sp_addr + ich * PE_N , cb_col, ab_col_row);
                uint32_t pre_sp = B_sp_addr_start + ich*PE_N + ocol* in_channels + i * ((out_channels / PE_N) + (out_channels % PE_N != 0)) * in_channels;
                for(int orow = 0; orow < out_dim; orow += PE_N){
                    int ac_row = out_dim - orow > PE_N ? PE_N : out_dim - orow;
                    uint32_t out_sp_addr = C_sp_addr_start + ocol * out_dim + orow;
                    uint32_t compute_sp = A_sp_addr_start + orow + ich * out_dim  + i * out_dim * ((in_channels / PE_N) + (in_channels % PE_N !=0));
                    gemmini_extended_preload(pre_sp, out_sp_addr, cb_col, ab_col_row, cb_col, ac_row);
                    gemmini_extended_compute_preloaded(compute_sp, GARBAGE_ADDR, ab_col_row, ac_row, PE_N, PE_N);

                }
            }
        }
    }



    //mvout output
    for(int ccol = 0; ccol < test_och; ccol++){
        int C_col = out_channels - ccol*PE_N > PE_N ? PE_N : out_channels - ccol*PE_N;
        for(int crow = 0; crow < out_dim; crow+=PE_N){
            uint32_t C_sp_addr = C_sp_addr_start + crow + ccol * out_dim;
            int C_row = out_dim - crow > PE_N ? PE_N : out_dim - crow;
            void *  C_dram_addr = (int8_t*)C + crow * out_channels + ccol*PE_N;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_col, C_row);
        }
    }


    // gemmini_fence();

}

static void batch_forloop(int dataflow, int act, acc_scale_t scale, elem_t relu_num, const elem_t * A, const elem_t * B,
                          const acc_t * D, void * C, int PE_N, int in_dim, int stride, int kernel_dim, int in_channels,
                          int batch_size, int out_channels, int out_dim){


    for(int b = 0; b < batch_size; b++){
        mc2_conv1d(dataflow, act, scale, relu_num, A + b * in_dim * in_channels, B, D+ b * out_dim * out_channels,
                   C + b * out_dim * out_channels, PE, in_dim, stride, kernel_dim, in_channels, out_channels, out_dim);

    }
}


#endif //GEMMINI_PROJECTS_GEMMINI_CUSTOM_H