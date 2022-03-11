// PairNet_post_processing.c
// Created by sam on 2022/1/9.
//
#include "func.h"
#include "pbPlots.h"
#include "supportLib.h"

 void plot_result(int batch_size, double y_axis[batch_size]){
    /**pbPlots github : https://github.com/InductiveComputerScience/pbPlots **/
    _Bool success;
    StringReference *errorMessage;
    RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();

    ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();

    double x_axis [batch_size+2];
    for (int i = 0; i < batch_size+2; ++i) {
        x_axis[i] = (double )i;
    }
    series->xs = x_axis;
    series->xsLength = batch_size+2;
    series->ys = y_axis;
    series->ysLength = batch_size+2;
    series->linearInterpolation = false;
    series->pointType = L"dots";
    series->pointTypeLength = wcslen(series->pointType);
    series->color = CreateRGBColor(0, 0, 1);
    ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
    settings->width = 1000;
    settings->height = 800;
    settings->autoBoundaries = true;
    settings->autoPadding = true;
    settings->title = L"Result";
    settings->titleLength = wcslen(settings->title);
    settings->xLabel = L"Batch Index";
    settings->xLabelLength = wcslen(settings->xLabel);
    settings->yLabel = L"Gesture";
    settings->yLabelLength = wcslen(settings->yLabel);
    ScatterPlotSeries *s [] = {series};
    settings->scatterPlotSeries = s;
    settings->scatterPlotSeriesLength = 1;
    errorMessage = (StringReference *)malloc(sizeof(StringReference));
    success = DrawScatterPlotFromSettings(imageReference, settings, errorMessage);

    if(success){
        size_t length;
        double *pngdata = ConvertToPNG(&length, imageReference->image);
        WriteToFile(pngdata, length, "pairnet_predicted.png");
        DeleteImage(imageReference->image);
    }else{
        fprintf(stderr, "Error: ");
        for(int i = 0; i < errorMessage->stringLength; i++){
            fprintf(stderr, "%c", errorMessage->string[i]);
        }
        fprintf(stderr, "\n");
    }
}

