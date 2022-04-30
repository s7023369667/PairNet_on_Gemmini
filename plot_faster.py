import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def spike():
    cpu16, cpu32, cpu64 = 41681, 131947, 460162
    conv1d16, conv1d32, conv1d64 = 329, 694, 1870
    matmul16, matmul32, matmul64 = 2505, 4351, 8086
    total_conv1d = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    total_matmul = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    plt.plot(total_conv1d, '-*')
    plt.plot(total_matmul, '-*')
    plt.xticks(np.arange(len(total_matmul)), ['16', '32', '64'])
    plt.xlabel("Model Channels")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Gemmini Conv1d', 'Gemmini Matmul'])
    plt.savefig('4x4.png')


def genesys2_4x4():
    clockRate = 31250
    cpu16_clock = 507892287
    cpu32_clock = 1625796589
    cpu64_clock = 6462754799
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate

    conv1d16_clock = 33405364
    conv1d32_clock = 85128724
    conv1d64_clock = 227579157
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    """Scala speedup"""
    matmul16_clock = 28070088
    matmul32_clock = 52841863
    matmul64_clock = 112093360
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate

    total_conv1d = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    total_matmul = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    plt.plot(total_conv1d, '-*')
    plt.plot(total_matmul, '-*')
    plt.xticks(np.arange(len(total_matmul)), ['16', '32', '64'])
    plt.xlabel("Model Channels")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Gemmini Conv1d', 'Gemmini Matmul'])
    plt.show()
    plt.close()


def genesys2_8x8_L2_Layers():
    clockRate = 25000
    """With L2 cache"""
    # # Layers
    # # 1st
    cpu16_clock = 115460479
    cpu32_clock = 240955245
    cpu64_clock = 453420947
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 5005430
    conv1d32_clock = 10809650
    conv1d64_clock = 20810729
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 9104864
    matmul32_clock = 14625336
    matmul64_clock = 24978971
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dFirst = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulFirst = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]
    # # 2nd
    cpu16_clock = 106360415
    cpu32_clock = 393543252
    cpu64_clock = 1628614599
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 3094900
    conv1d32_clock = 6821817
    conv1d64_clock = 16288570
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 7104570
    matmul32_clock = 14279500
    matmul64_clock = 27955125
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dSecond = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulSecond = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    # # 3rd
    cpu16_clock = 53155647
    cpu32_clock = 207464185
    cpu64_clock = 816205085
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 1754710
    conv1d32_clock = 4044696
    conv1d64_clock = 10189717
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 3596277
    matmul32_clock = 7085296
    matmul64_clock = 15864866
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dThird = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulThird = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]
    # # 4th
    cpu16_clock = 68040122
    cpu32_clock = 259745333
    cpu64_clock = 1115527956
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 1916526
    conv1d32_clock = 4648755
    conv1d64_clock = 12295313
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 2694889
    matmul32_clock = 5579952
    matmul64_clock = 11617676
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dForth = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulForth = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]
    # # 5th
    cpu16_clock = 64313392
    cpu32_clock = 260565296
    cpu64_clock = 1794143348
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 1923775
    conv1d32_clock = 5266334
    conv1d64_clock = 16870522
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 2273172
    matmul32_clock = 4812390
    matmul64_clock = 10824284
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dFifth = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulFifth = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    matmul_16 = [matmulFirst[0], matmulSecond[0], matmulThird[0], matmulForth[0], matmulFifth[0]]
    matmul_32 = [matmulFirst[1], matmulSecond[1], matmulThird[1], matmulForth[1], matmulFifth[1]]
    matmul_64 = [matmulFirst[2], matmulSecond[2], matmulThird[2], matmulForth[2], matmulFifth[2]]

    conv_16 = [conv1dFirst[0], conv1dSecond[0], conv1dThird[0], conv1dForth[0], conv1dFifth[0]]
    conv_32 = [conv1dFirst[1], conv1dSecond[1], conv1dThird[1], conv1dForth[1], conv1dFifth[1]]
    conv_64 = [conv1dFirst[2], conv1dSecond[2], conv1dThird[2], conv1dForth[2], conv1dFifth[2]]

    plt.plot(matmul_16, '-*')
    plt.plot(matmul_32, '-*')
    plt.plot(matmul_64, '-*')
    plt.plot(conv_16, '-*')
    plt.plot(conv_32, '-*')
    plt.plot(conv_64, '-*')
    plt.xticks(np.arange(len(matmul_16)), ['1', '2', '3', '4', '5'])
    plt.xlabel("Model Layer")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Matmul16', 'Matmul32', 'Matmul64', 'Conv1d16', 'Conv1d32', 'Conv1d64'])
    plt.title("8x8 With L2 Cache Layer by Layer")
    plt.savefig('8x8 With L2 Cache Layer by Layer.png')
    plt.close()


def genesys2_8x8_NOL2_Layers():
    clockRate = 25000
    """Without L2 cache"""
    # # Layers
    # # 1st
    cpu16_clock = 119911445
    cpu32_clock = 249741576
    cpu64_clock = 471085720
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 6831794
    conv1d32_clock = 12613081
    conv1d64_clock = 24217219
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 10319012
    matmul32_clock = 16001183
    matmul64_clock = 26418422
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dFirst = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulFirst = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]
    # # 2nd
    cpu16_clock = 109480174
    cpu32_clock = 407182214
    cpu64_clock = 1719480394
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 3701364
    conv1d32_clock = 8048840
    conv1d64_clock = 18435781
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 8145393
    matmul32_clock = 15418583
    matmul64_clock = 29482845
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dSecond = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulSecond = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    # # 3rd
    cpu16_clock = 54447196
    cpu32_clock = 213845022
    cpu64_clock = 905164886
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 1961044
    conv1d32_clock = 4999456
    conv1d64_clock = 12148427
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 4139728
    matmul32_clock = 7613943
    matmul64_clock = 17133955
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dThird = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulThird = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]
    # # 4th
    cpu16_clock = 69893868
    cpu32_clock = 269466897
    cpu64_clock = 1242281750
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 2164115
    conv1d32_clock = 5229737
    conv1d64_clock = 15544384
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 3055433
    matmul32_clock = 6259209
    matmul64_clock = 12314820
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dForth = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulForth = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]
    # # 5th
    cpu16_clock = 66603717
    cpu32_clock = 271797991
    cpu64_clock = 2129566873
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 2468342
    conv1d32_clock = 6736296
    conv1d64_clock = 23633582
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 2599296
    matmul32_clock = 5619376
    matmul64_clock = 11743454
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1dFifth = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmulFifth = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    matmul_16 = [matmulFirst[0], matmulSecond[0], matmulThird[0], matmulForth[0], matmulFifth[0]]
    matmul_32 = [matmulFirst[1], matmulSecond[1], matmulThird[1], matmulForth[1], matmulFifth[1]]
    matmul_64 = [matmulFirst[2], matmulSecond[2], matmulThird[2], matmulForth[2], matmulFifth[2]]

    conv_16 = [conv1dFirst[0], conv1dSecond[0], conv1dThird[0], conv1dForth[0], conv1dFifth[0]]
    conv_32 = [conv1dFirst[1], conv1dSecond[1], conv1dThird[1], conv1dForth[1], conv1dFifth[1]]
    conv_64 = [conv1dFirst[2], conv1dSecond[2], conv1dThird[2], conv1dForth[2], conv1dFifth[2]]

    plt.plot(matmul_16, '-.')
    plt.plot(matmul_32, '-v')
    plt.plot(matmul_64, '-s')
    plt.plot(conv_16, '-D')
    plt.plot(conv_32, '-x')
    plt.plot(conv_64, '-*')
    plt.xticks(np.arange(len(matmul_16)), ['1', '2', '3', '4', '5'])
    plt.xlabel("Model Layer")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Matmul16', 'Matmul32', 'Matmul64', 'Conv1d16', 'Conv1d32', 'Conv1d64'])
    plt.title("8x8 Without L2 Cache Layer by Layer")
    plt.savefig('8x8 Without L2 Cache Layer by Layer.png')
    plt.close()


def genesys2_8x8_NOL2_FULL():
    clockRate = 25000
    """Without L2 cache"""
    # # 1st
    cpu16_clock = 527493816
    cpu32_clock = 1700630792
    cpu64_clock = 7333774536
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 17622814
    conv1d32_clock = 38027144
    conv1d64_clock = 93896325
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 27619101
    matmul32_clock = 49413874
    matmul64_clock = 93151715
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1d = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmul = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    plt.plot(conv1d, '-*')
    plt.plot(matmul, '-*')
    plt.xticks(np.arange(len(conv1d)), ['16', '32', '64'])
    plt.xlabel("Model Channels")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Conv1d', 'Matmul'])
    plt.title("8x8 Without L2 Cache Full Layers")
    plt.savefig('8x8 Without L2 Cache Full Layers.png')
    plt.close()


def genesys2_8x8_L2_FULL():
    clockRate = 25000
    """Without L2 cache"""
    # # 1st
    cpu16_clock = 510446403
    cpu32_clock = 1631119537
    cpu64_clock = 6487786484
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 15416292
    conv1d32_clock = 31779400
    conv1d64_clock = 77989823
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 23718848
    matmul32_clock = 43404594
    matmul64_clock = 85844502
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1d = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmul = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    plt.plot(conv1d, '-*')
    plt.plot(matmul, '-*')
    plt.xticks(np.arange(len(conv1d)), ['16', '32', '64'])
    plt.xlabel("Model Channels")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Conv1d', 'Matmul'])
    plt.title("8x8 With L2 Cache Full Layers")
    plt.savefig('8x8 With L2 Cache Full Layers.png')
    plt.close()


def genesys2_8x8_noL2_1window():
    clockRate = 25000
    """Without L2 cache"""
    # # 1st
    cpu16_clock = 1667908
    cpu32_clock = 5184222
    cpu64_clock = 22273887
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 68782
    conv1d32_clock = 111492
    conv1d64_clock = 277867
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 172075
    matmul32_clock = 365369
    matmul64_clock = 926824
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1d = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmul = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    plt.plot(conv1d, '-*')
    plt.plot(matmul, '-*')
    plt.xticks(np.arange(len(conv1d)), ['16', '32', '64'])
    plt.xlabel("Model Channels")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Conv1d', 'Matmul'])
    plt.title("8x8 With L2 Cache 1 window")
    plt.savefig('8x8 With L2 Cache 1 window.png')
    plt.close()


def genesys2_8x8_L2_1window():
    clockRate = 25000
    """Without L2 cache"""
    # # 1st
    cpu16_clock = 1604298
    cpu32_clock = 5061226
    cpu64_clock = 19693300
    cpu16, cpu32, cpu64 = cpu16_clock / clockRate, cpu32_clock / clockRate, cpu64_clock / clockRate
    conv1d16_clock = 57710
    conv1d32_clock = 98213
    conv1d64_clock = 224338
    conv1d16, conv1d32, conv1d64 = conv1d16_clock / clockRate, conv1d32_clock / clockRate, conv1d64_clock / clockRate
    matmul16_clock = 130827
    matmul32_clock = 312423
    matmul64_clock = 887256
    matmul16, matmul32, matmul64 = matmul16_clock / clockRate, matmul32_clock / clockRate, matmul64_clock / clockRate
    conv1d = [cpu16 / conv1d16, cpu32 / conv1d32, cpu64 / conv1d64]
    matmul = [cpu16 / matmul16, cpu32 / matmul32, cpu64 / matmul64]

    plt.plot(conv1d, '-*')
    plt.plot(matmul, '-*')
    plt.xticks(np.arange(len(conv1d)), ['16', '32', '64'])
    plt.xlabel("Model Channels")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Conv1d', 'Matmul'])
    plt.title("8x8 Without L2 Cache 1 window")
    plt.savefig('8x8 Without L2 Cache 1 window.png')
    plt.close()


def layerByLayer(cvsPath: str):
    cvs = pd.read_csv(cvsPath)
    df = pd.DataFrame(cvs)
    df = df.set_index('Models')
    print(df)

    matmulFirst = [df['Layer1']['CPU_matmul_16'] / df['Layer1']['Gemmini_matmul_16'],
                   df['Layer1']['CPU_matmul_32'] / df['Layer1']['Gemmini_matmul_32'],
                   df['Layer1']['CPU_matmul_48'] / df['Layer1']['Gemmini_matmul_48'],
                   df['Layer1']['CPU_matmul_56'] / df['Layer1']['Gemmini_matmul_56'],
                   df['Layer1']['CPU_matmul_64'] / df['Layer1']['Gemmini_matmul_64'],
                   df['Layer1']['CPU_matmul_72'] / df['Layer1']['Gemmini_matmul_72'],
                   df['Layer1']['CPU_matmul_80'] / df['Layer1']['Gemmini_matmul_80'],
                   df['Layer1']['CPU_matmul_88'] / df['Layer1']['Gemmini_matmul_88'],
                   df['Layer1']['CPU_matmul_96'] / df['Layer1']['Gemmini_matmul_96'],
                   df['Layer1']['CPU_matmul_128'] / df['Layer1']['Gemmini_matmul_128']]
    matmulSecond = [df['Layer2']['CPU_matmul_16'] / df['Layer2']['Gemmini_matmul_16'],
                    df['Layer2']['CPU_matmul_32'] / df['Layer2']['Gemmini_matmul_32'],
                    df['Layer2']['CPU_matmul_48'] / df['Layer2']['Gemmini_matmul_48'],
                    df['Layer2']['CPU_matmul_56'] / df['Layer2']['Gemmini_matmul_56'],
                    df['Layer2']['CPU_matmul_64'] / df['Layer2']['Gemmini_matmul_64'],
                    df['Layer2']['CPU_matmul_72'] / df['Layer2']['Gemmini_matmul_72'],
                    df['Layer2']['CPU_matmul_80'] / df['Layer2']['Gemmini_matmul_80'],
                    df['Layer2']['CPU_matmul_88'] / df['Layer2']['Gemmini_matmul_88'],
                    df['Layer2']['CPU_matmul_96'] / df['Layer2']['Gemmini_matmul_96'],
                    df['Layer2']['CPU_matmul_128'] / df['Layer2']['Gemmini_matmul_128']]
    matmulThird = [df['Layer3']['CPU_matmul_16'] / df['Layer3']['Gemmini_matmul_16'],
                   df['Layer3']['CPU_matmul_32'] / df['Layer3']['Gemmini_matmul_32'],
                   df['Layer3']['CPU_matmul_48'] / df['Layer3']['Gemmini_matmul_48'],
                   df['Layer3']['CPU_matmul_56'] / df['Layer3']['Gemmini_matmul_56'],
                   df['Layer3']['CPU_matmul_64'] / df['Layer3']['Gemmini_matmul_64'],
                   df['Layer3']['CPU_matmul_72'] / df['Layer3']['Gemmini_matmul_72'],
                   df['Layer3']['CPU_matmul_80'] / df['Layer3']['Gemmini_matmul_80'],
                   df['Layer3']['CPU_matmul_88'] / df['Layer3']['Gemmini_matmul_88'],
                   df['Layer3']['CPU_matmul_96'] / df['Layer3']['Gemmini_matmul_96'],
                   df['Layer3']['CPU_matmul_128'] / df['Layer3']['Gemmini_matmul_128']]
    matmulForth = [df['Layer4']['CPU_matmul_16'] / df['Layer4']['Gemmini_matmul_16'],
                   df['Layer4']['CPU_matmul_32'] / df['Layer4']['Gemmini_matmul_32'],
                   df['Layer4']['CPU_matmul_48'] / df['Layer4']['Gemmini_matmul_48'],
                   df['Layer4']['CPU_matmul_56'] / df['Layer4']['Gemmini_matmul_56'],
                   df['Layer4']['CPU_matmul_64'] / df['Layer4']['Gemmini_matmul_64'],
                   df['Layer4']['CPU_matmul_72'] / df['Layer4']['Gemmini_matmul_72'],
                   df['Layer4']['CPU_matmul_80'] / df['Layer4']['Gemmini_matmul_80'],
                   df['Layer4']['CPU_matmul_88'] / df['Layer4']['Gemmini_matmul_88'],
                   df['Layer4']['CPU_matmul_96'] / df['Layer4']['Gemmini_matmul_96'],
                   df['Layer4']['CPU_matmul_128'] / df['Layer4']['Gemmini_matmul_128']]
    matmulFifth = [df['Layer5']['CPU_matmul_16'] / df['Layer5']['Gemmini_matmul_16'],
                   df['Layer5']['CPU_matmul_32'] / df['Layer5']['Gemmini_matmul_32'],
                   df['Layer5']['CPU_matmul_48'] / df['Layer5']['Gemmini_matmul_48'],
                   df['Layer5']['CPU_matmul_56'] / df['Layer5']['Gemmini_matmul_56'],
                   df['Layer5']['CPU_matmul_64'] / df['Layer5']['Gemmini_matmul_64'],
                   df['Layer5']['CPU_matmul_72'] / df['Layer5']['Gemmini_matmul_72'],
                   df['Layer5']['CPU_matmul_80'] / df['Layer5']['Gemmini_matmul_80'],
                   df['Layer5']['CPU_matmul_88'] / df['Layer5']['Gemmini_matmul_88'],
                   df['Layer5']['CPU_matmul_96'] / df['Layer5']['Gemmini_matmul_96'],
                   df['Layer5']['CPU_matmul_128'] / df['Layer5']['Gemmini_matmul_128']]
    matmul_16 = [matmulFirst[0], matmulSecond[0], matmulThird[0], matmulForth[0], matmulFifth[0]]
    matmul_32 = [matmulFirst[1], matmulSecond[1], matmulThird[1], matmulForth[1], matmulFifth[1]]
    matmul_48 = [matmulFirst[2], matmulSecond[2], matmulThird[2], matmulForth[2], matmulFifth[2]]
    matmul_56 = [matmulFirst[3], matmulSecond[3], matmulThird[3], matmulForth[3], matmulFifth[3]]
    matmul_64 = [matmulFirst[4], matmulSecond[4], matmulThird[4], matmulForth[4], matmulFifth[4]]
    matmul_72 = [matmulFirst[5], matmulSecond[5], matmulThird[5], matmulForth[5], matmulFifth[5]]
    matmul_80 = [matmulFirst[6], matmulSecond[6], matmulThird[6], matmulForth[6], matmulFifth[6]]
    matmul_88 = [matmulFirst[7], matmulSecond[7], matmulThird[7], matmulForth[7], matmulFifth[7]]
    matmul_96 = [matmulFirst[8], matmulSecond[8], matmulThird[8], matmulForth[8], matmulFifth[8]]
    matmul_128 = [matmulFirst[9], matmulSecond[9], matmulThird[9], matmulForth[9], matmulFifth[9]]

    plt.plot(matmul_16, '-o')
    plt.plot(matmul_32, '-v')
    plt.plot(matmul_48, '-s')
    plt.plot(matmul_56, '-D')
    plt.plot(matmul_64, '-X')
    plt.plot(matmul_88, '-8')
    plt.plot(matmul_128, '-*')
    plt.xticks(np.arange(5), ['1', '2', '3', '4', '5'])
    plt.xlabel("Model Layer")
    plt.ylabel("Times faster than CPU")
    plt.legend(['Model channel 16', 'Model channel 32', 'Model channel 48', 'Model channel 56', 'Model channel 64',
                'Model channel 88', 'Model channel 128'])
    plt.title("8x8 CPU Matmul V.S Gemmini Matmul Layer by Layer")
    plt.savefig('8x8 CPU Matmul V.S Gemmini Matmul Layer by Layer.png')
    plt.show()


if __name__ == '__main__':
    # genesys2_8x8_L2_Layers()
    # genesys2_8x8_NOL2_Layers()
    #
    # genesys2_8x8_L2_FULL()
    # genesys2_8x8_NOL2_FULL()
    #
    # genesys2_8x8_L2_1window()
    # genesys2_8x8_noL2_1window()
    # genesys2_4x4()
    layerByLayer('./layerBylayer_allwindows.csv')
