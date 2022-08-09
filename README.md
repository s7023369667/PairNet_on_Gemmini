# PairNet on Gemmini

* PairNet paper reference : https://ieeexplore.ieee.org/document/8875280


1. 根據PairNet的神經網路架構，我們針對每層卷積層以不同數量的Filter，設計三種不同的實驗神經網路架構NN1(Neural Network1)、NN2(Neural Network2)、NN3(Neural Network3)。

1D CNN configure : (Kernel Size, Stride Size, Output Channels)
![](https://i.imgur.com/G0HvZLq.png)

由於在Gemmini上的計算都必須是整數8-bit的資料型態，因此需要將手勢訊號和模型進行量化。並且模型架構我們根據以下兩點來進行必要的修改。第一點，由於我們將本論文以矩陣乘法實現一維卷積計算的方法，應用在NN1、NN2及NN3上，使得Gemmini可以進一步對整體一維卷積計算進行加速，為了達成此目，在每層一維卷積計算前都在必須經過矩陣重新排列(Rearrange Data)的額外處理，才得以正確的進行矩陣乘法。第二點，為了簡化模型辨識的計算複雜度，本論文利用BN Folding的技術將BN和一維卷積層的參數合併，這使我們在一維卷積計算的過程中即可一併完成BN計算。對模型的每一層以此兩點進行調整。

2. 模型實驗架構會分成A、B兩種情境來進行，來比較模型在不同情境時的計速度：
    * 實驗情境A中NN1、NN2和NN3，紅色粗框的部分則都由Gemmini來進行計算，而矩陣重新排列和ReLU的部分，則使用Rocket Core來進行計算。
    ![](https://i.imgur.com/6n1zIIb.png)

    * 實驗情境B中NN1、NN2和NN3，每層計算都是由Rocket Core來執行，不會使用Gemmini來進行計算。
    ![](https://i.imgur.com/VPqPiDO.png)


3. 實驗會將量化後的手機手勢訊號及模型佈署在Genesys2 FPGA開發板上，採用RISC-V為主搭載Rocket Core及Gemmini的SoC，並將Gemmini的Systolic Array硬體架構分為4X4及8X8兩種大小來進行實驗。
![](https://i.imgur.com/q3ngSbT.png)
