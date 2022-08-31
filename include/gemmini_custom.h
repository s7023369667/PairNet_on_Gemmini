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

/***********************************************/
/**add this mc2_config into gemmini.h**/
//#define gemmini_mc2_config_ex(dataflow, sys_act) \
//  { \
//    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)1.0) << 32) | ((uint64_t)(1) << 16) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(1) << 48) , k_CONFIG); \
//  }
//
//#define gemmini_mc2_config_st(stride, acc_act, acc_scale) \
//ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)
//
//#define gemmini_mc2_config_ldA(stride, scale) \
//  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(DIM) << 16) | ((uint64_t)(1) << 8) | CONFIG_LD, stride, k_CONFIG)
//
//
//#define gemmini_mc2_config_ldB(stride, scale) \
//ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(DIM) << 16) | ((uint64_t)(1) << 8) |((1) << 3)| CONFIG_LD, stride, k_CONFIG)
//
//
//#define gemmini_mc2_config_ldD(stride, scale) \
//  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(DIM) << 16) | ((uint64_t)(1) << 8) |((2) << 3)| CONFIG_LD, stride, k_CONFIG)
//
//#define gemmini_global_ld(stride) \
//  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)(scale_t_to_scale_t_bits(1)) << 32) | ((uint64_t)(DIM) << 16) | ((1) << 2) | CONFIG_LD, stride, k_CONFIG)
/***********************************************/

/**gemmini global average pooling*/
static void global_avg(int out_dim, int out_channels, int PE_N, const elem_t * A, const elem_t * C){
    double scale = 1.0 / out_dim;
    gemmini_mc2_config_ex(1, 0);
    gemmini_global_ld(PE_N);
    gemmini_mc2_config_st(PE_N, 0, scale);
    const uint32_t C_sp_addr_start = 0XC0000000 ;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    int out_rows = (out_channels / (PE_N*PE_N)) + (out_channels % (PE_N*PE_N) != 0);
    int out_rows2 = (out_channels / PE_N) + (out_channels % PE_N != 0);
    ////mvin
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
    ////mvout
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
/**Gemmini conv1d**/
static void mc2_conv1d(int dataflow, int act, acc_scale_t out_scale, const elem_t * A, const elem_t * B, acc_t* D, elem_t * C,
                       int PE_N, int in_dim, int stride, int kernel_dim, int in_channels, int out_channels, int out_dim)
{
    uint64_t s, e;
    //// Compute Method Cofigure
    gemmini_config_ex(dataflow, 0, 0, 0);
    ////Load Configure
    /**load input configure*/
    gemmini_extended3_config_ld(stride * in_channels, 1, 0, 0);
    /**load weight configure*/
    gemmini_extended3_config_ld(out_channels, 1.0, 0, 1);
    /**load bias configure*/
    gemmini_extended3_config_ld(0, 1.0, 0, 2);
    ////Store Configure
    gemmini_extended_config_st(out_channels, act, out_scale);
    ////Address pre setting
    uint32_t A_sp_addr_start = 0;
    uint32_t B_sp_addr_start = 0X00001000;
    ////If we need Accumulate all output C must store at 0XC0000000
    uint32_t C_sp_addr_start = 0XC0000000;
    uint32_t D_sp_addr_start = 0X80000000;
    int in_col = in_channels > MAX_BYTES ? MAX_BYTES : in_channels;
    int out_col = out_channels > MAX_BYTES ? MAX_BYTES : out_channels;
    int test_orow = out_dim / PE_N + (out_dim % PE_N !=0);
    int test_irow = in_dim / PE_N + (in_dim % PE_N !=0);
    int test_ich = in_channels / PE_N + (in_channels % PE_N !=0);
    int test_och = out_channels / PE_N + (out_channels % PE_N!=0);
    //mvin bias

    for(int bcol =0; bcol < out_channels; bcol += PE_N){
        int D_col = out_channels - bcol > PE_N ? PE_N : out_channels - bcol;
        uint32_t D_sp_addr = D_sp_addr_start + (bcol / PE_N) * out_dim;
        for (int brow = 0; brow < out_dim; brow += PE_N){
            int D_row = out_dim - brow > PE_N ? PE_N : (out_dim - brow) > 0 ? (out_dim - brow) : 0;
// acc_t * D_dram_addr = D + brow * out_channels + bcol;
            acc_t * D_dram_addr = D + bcol;
            gemmini_extended_mvin3(D_dram_addr, D_sp_addr + brow, D_col, D_row);

        }
    }

//mvin input
    for(int i = 0; i < kernel_dim; i++){
        for(int icol = 0; icol < in_channels; icol += PE_N){
            int A_col = in_channels - icol > PE_N ? PE_N : in_channels - icol;
            for(int irow = 0; irow < in_dim; irow += PE_N){
                uint32_t A_sp_addr = A_sp_addr_start + irow + (icol / PE_N) * out_dim + i * out_dim * (in_channels / PE_N + (in_channels % PE_N !=0));
                int A_row = (in_dim / stride) - irow > PE_N ? PE_N : ((out_dim - irow) > 0 ? (out_dim - irow) : 0);
                if(A_row && A_col != 0){
                    gemmini_extended_mvin(A + icol + irow * stride * in_channels + i * in_channels , A_sp_addr , A_col, A_row);
                }
            }
        }
    }



//compute
    for(int i = 0; i < kernel_dim; i++){
        for(int ocol = 0; ocol < out_channels; ocol += PE_N){
            int cb_col = out_channels - ocol > PE_N ? PE_N : out_channels - ocol;
            uint32_t B_sp_addr = B_sp_addr_start + (ocol / PE_N) * in_channels + i * in_channels * ((out_channels / PE_N) + (out_channels % PE_N != 0));
            for(int ich = 0; ich < in_channels; ich += PE_N){
                int ab_col_row = in_channels - ich > PE_N ? PE_N : in_channels - ich;
                gemmini_extended_mvin2(B + ich * out_channels + i * out_channels * in_channels + ocol, B_sp_addr + ich , cb_col, ab_col_row);
                uint32_t pre_sp = B_sp_addr_start + ich + (ocol / PE_N) * in_channels + i * ((out_channels / PE_N) + (out_channels % PE_N != 0)) * in_channels;
                for(int orow = 0; orow < out_dim; orow += PE_N){
                    int ac_row = out_dim - orow > PE_N ? PE_N : out_dim - orow;
                    uint32_t out_sp_addr = C_sp_addr_start + (ocol / PE_N) * out_dim + orow;
                    uint32_t compute_sp = A_sp_addr_start + orow + (ich / PE_N) * out_dim + i * out_dim * ((in_channels / PE_N) + (in_channels % PE_N !=0));
                    gemmini_extended_preload(pre_sp, out_sp_addr, cb_col, ab_col_row, cb_col, ac_row);
                    gemmini_extended_compute_preloaded(compute_sp, GARBAGE_ADDR, ab_col_row, ac_row, PE_N, PE_N);

                }
            }
        }
    }


//mvout output
    for(int ccol = 0; ccol < out_channels; ccol+=PE_N){
        int C_col = out_channels - ccol > PE_N ? PE_N : out_channels - ccol;
        for(int crow = 0; crow < out_dim; crow+=PE_N){
            uint32_t C_sp_addr = C_sp_addr_start + crow + (ccol / PE_N) * out_dim;
            int C_row = out_dim - crow > PE_N ? PE_N : out_dim - crow;
            void * C_dram_addr = (int8_t*)C + crow * out_channels + ccol;
            gemmini_extended_mvout(C_dram_addr, C_sp_addr, C_col, C_row);
        }
    }


    gemmini_fence();

}
static void batch_forloop(int dataflow, int act, acc_scale_t out_scale,  elem_t * A, elem_t * B, acc_t* D, elem_t * C,
                          int PE_N, int in_dim, int stride, int kernel_dim, int in_channels, int batch_size, int out_channels, int out_dim){
    int A_offset = in_dim * in_channels;
    int C_offset = out_dim * out_channels;
    for(int i = 0; i < batch_size; i++){
        mc2_conv1d(dataflow, act, out_scale, A + i * A_offset , B, D, C + i * C_offset , PE_N, in_dim, stride, kernel_dim, in_channels, out_channels, out_dim);
    }
}
#endif //GEMMINI_PROJECTS_GEMMINI_CUSTOM_H
