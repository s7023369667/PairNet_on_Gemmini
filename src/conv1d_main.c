// conv1d_main.c
// Created by sam on 2022/6/2.
//
#include <string.h>
#include <stdio.h>
#include "include/gemmini_custom.h"
#include "include/conv1d_func.h"

int main() {
    int batch_size ;
    int input_width;
    int input_channel;
    int kernel_size;
    int output_channel;
    int stride_size;
    int padding_size;
    int output_width;
    const int feature[1][12][3] = {{{2, 1, 1},
                                    {3, 1, 1},
                                    {2, 0, 0},
                                    {3, 2, 2},
                                    {1, 1, 1},
                                    {0, 0, 0},
                                    {3, 2, 0},
                                    {0, 0, 1},
                                    {0, 1, 1},
                                    {1, 3, 3},
                                    {1, 1, 1},
                                    {0, 1, 1}}};

    ////case 1
    printf("------case1-----------\n");
    batch_size = 1;
    input_width = 12;
    input_channel = 3;
    output_channel = 2;
    kernel_size = 2;
    stride_size = 1;
    padding_size = check_padding_size(true, kernel_size);
    output_width = (int )((input_width - kernel_size + padding_size)/stride_size + 1);

    const int case1_trained_weights[2][3][2] = {{{1, 2},
                                               {0, 1},
                                               {1, 2}},
                                              {{0, 0},
                                               {1, 1},
                                               {1, 1}}};
    int case1_conv1d_result[batch_size][output_width][output_channel];
    conv1d(batch_size, input_width, input_channel, feature, kernel_size, output_channel,
           stride_size, case1_trained_weights, output_width, padding_size, case1_conv1d_result);
    printf("Conv1d Result:\n");
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < output_channel; ++k) {
                printf("%d\t\t", case1_conv1d_result[i][j][k]);
            }
            printf("\n");
        }
    }
    ////case 2
    printf("------case2-----------\n");
    batch_size = 1;
    input_width = 12;
    input_channel = 3;
    output_channel = 2;
    kernel_size = 2;
    stride_size = 2;
    padding_size = check_padding_size(true, kernel_size);
    output_width = (int )((input_width - kernel_size + padding_size)/stride_size + 1);

    int case2_conv1d_result[batch_size][output_width][output_channel];
    conv1d(batch_size, input_width, input_channel, feature, kernel_size, output_channel, stride_size,
           case1_trained_weights, output_width, padding_size, case2_conv1d_result);
    printf("Conv1d Result:\n");
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < output_channel; ++k) {
                printf("%d\t\t", case2_conv1d_result[i][j][k]);
            }
            printf("\n");
        }
    }
    ////case 3
    printf("------case3-----------\n");
    batch_size = 1;
    input_width = 12;
    input_channel = 3;
    output_channel = 2;
    stride_size = 1;
    kernel_size = 3;
    padding_size = check_padding_size(true, kernel_size);
    output_width = (int )((input_width - kernel_size + padding_size)/stride_size + 1);
    const int case3_trained_weights[3][3][2] = {{{1, 2},
                                                 {0, 1},
                                                 {1, 2}},
                                                {{0, 0},
                                                 {1, 1},
                                                 {1, 1}},
                                                {{1, 2},
                                                 {1, 2},
                                                 {0, 0}}};
    int conv1d_case3_result[batch_size][output_width][output_channel];
    conv1d(batch_size, input_width, input_channel, feature, kernel_size, output_channel, stride_size,
           case3_trained_weights,output_width, padding_size, conv1d_case3_result);
    printf("Conv1d Result:\n");
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < output_channel; ++k) {
                printf("%d\t\t", conv1d_case3_result[i][j][k]);
            }
            printf("\n");
        }
    }
    ////case 4
    printf("------case4-----------\n");
    batch_size = 1;
    input_width = 12;
    input_channel = 3;
    output_channel = 2;
    kernel_size = 3;
    stride_size = 2;
    padding_size = check_padding_size(true, kernel_size);
    output_width = (int )((input_width - kernel_size + padding_size)/stride_size + 1);
    int conv1d_case4_result[batch_size][output_width][output_channel];
    conv1d(batch_size, input_width, input_channel, feature, kernel_size, output_channel, stride_size,
           case3_trained_weights,output_width, padding_size, conv1d_case4_result);
    printf("Conv1d Result:\n");
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < output_channel; ++k) {
                printf("%d\t\t", conv1d_case4_result[i][j][k]);
            }
            printf("\n");
        }
    }
    /////case additional
    printf("------case additional-----------\n");
    batch_size = 1;
    input_width = 12;
    input_channel = 3;
    output_channel = 2;
    kernel_size = 5;
    stride_size = 1;
    const int case_add_trained_weights[5][3][2] = {{{1, 2},
                                                 {0, 1},
                                                 {1, 2}},
                                                {{0, 0},
                                                 {1, 1},
                                                 {1, 1}},
                                                {{1, 2},
                                                 {1, 2},
                                                 {0, 0}},
                                                {{0, 2},
                                                 {1, 1},
                                                 {2, 2}},
                                                {{1, 1},
                                                 {1, 1},
                                                 {1, 1}}};
    padding_size = check_padding_size(true, kernel_size);
    output_width = (int )((input_width - kernel_size + padding_size)/stride_size + 1);
    int conv1d_case_additional_result[batch_size][output_width][output_channel];
    conv1d(batch_size, input_width, input_channel, feature, kernel_size, output_channel, stride_size,
           case_add_trained_weights,output_width, padding_size, conv1d_case_additional_result);
    printf("Conv1d Result:\n");
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < output_channel; ++k) {
                printf("%d\t\t", conv1d_case_additional_result[i][j][k]);
            }
            printf("\n");
        }
    }

    ////case demo
    printf("------------case demo-----------\n");
    batch_size = 1;
    input_width = 32;
    input_channel = 4;
    output_channel = 4;
    kernel_size = 3;
    stride_size = 2;
    const int input_signal[1][32][4] = {{{5, 2, 3, 4}, {7, 3, 4, 5}, {9, 4, 5, 6}, {9, 5, 6, 7},
                                         {9, 4, 5, 6}, {13, 2, 4, 5}, {44, 3, 5, 6}, {55, 4, 6, 7},
                                         {1, 5, 3, 4}, {1, 6, 4, 5}, {1, 2, 5, 6}, {1, 3, 6, 7},
                                         {1, 4, 3, 4}, {1, 5, 4, 5}, {1, 6, 5, 6}, {1, 2, 6, 7},
                                         {5, 3, 3, 4}, {7, 4, 4, 5}, {9, 5, 5, 6}, {9, 6, 6, 7},
                                         {9, 2, 3, 4}, {13, 3, 4, 5}, {44, 4, 5, 6}, {55, 5, 6, 7},
                                         {1, 6, 3, 4}, {1, 2, 4, 5}, {1, 3, 5, 6}, {1, 4, 6, 7},
                                         {1, 5, 3, 4}, {1, 6, 4, 5}, {1, 2, 5, 6}, {1, 3, 6, 7}}};

    const int trained_weights[3][4][4] = {{{1, 0, 2, 2},
                                                  {2, 1, 0, 0},
                                                  {3, 0, 1, 5},
                                                  {4, 1, 1, 6}},
                                          {{2, 4, 6, 8},
                                                  {3, 2, 0, 0},
                                                  {4, 5, 2, 2},
                                                  {5, 1, 0, 0}},
                                          {{3, 7, 0, 5},
                                                  {4, 4, 4, 2},
                                                  {5, 3, 0, 0},
                                                  {6, 1, 1, 1}}};
    const int trained_weights2[3][4][4] = {{{1,  0, -2,   2},
                                                   {2,  1,  0,   0},
                                                   {3,  0,  1,  -5},
                                                   {4,  1,  1,  -6}},
                                           {{-2, -4, -1,   1},
                                                   {-3,  2,  0,   0},
                                                   {-4, -5, -2,   2},
                                                   {-5,  1,  0,   0}},
                                           {{ 3, -1,  0,   0},
                                                   { 4,  0,  0,  -2},
                                                   { 0,  3,  3,   0},
                                                   { 0,  0, -1,   1}}};
    const int bias[1][15][4]={{{1,1,1,1},{1,1,1,1},{1,1,1,1},
                               {1,1,1,1},{1,1,1,1},{1,1,1,1},
                               {1,1,1,1},{1,1,1,1},{1,1,1,1},
                               {1,1,1,1},{1,1,1,1},{1,1,1,1},
                               {1,1,1,1},{1,1,1,1},{1,1,1,1}}};
    padding_size = check_padding_size(false, kernel_size);
    output_width = (int) ((input_width - kernel_size + padding_size) / stride_size + 1);
    int conv1d_case_demo_result[batch_size][output_width][output_channel];
    conv1d(batch_size, input_width, input_channel, input_signal, kernel_size, output_channel, stride_size,
           trained_weights2, output_width, padding_size, bias, conv1d_case_demo_result);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_width; ++j) {
            for (int k = 0; k < output_channel; ++k) {
                printf("%8d", conv1d_case_demo_result[i][j][k]);
            }
            printf("\n");
        }
    }
    return 0;
}
