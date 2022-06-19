import tensorflow as tf
import numpy as np

batch_size = 1
input_width = 12
input_channel = 3
output_channel = 2


def test(feature, kernel, ks, stride, padding):
    feature = np.reshape(feature, [batch_size, input_width, input_channel])
    kernel = np.reshape(kernel, [ks, input_channel, output_channel])
    conv1d_res = tf.nn.conv1d(feature, kernel, stride, padding)
    print(conv1d_res)


if __name__ == '__main__':
    feature = np.array([[2, 1, 1],
                        [3, 1, 1],
                        [2, 0, 0],
                        [3, 2, 2],
                        [1, 1, 1],
                        [0, 0, 0],
                        [3, 2, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [1, 3, 3],
                        [1, 1, 1],
                        [0, 1, 1]], dtype=np.float32)
    kernel2 = np.array([[[1, 2],
                         [0, 1],
                         [1, 2]],
                        [[0, 0],
                         [1, 1],
                         [1, 1]]], dtype=np.float32)
    kernel3 = np.array([[[1, 2],
                         [0, 1],
                         [1, 2]],
                        [[0, 0],
                         [1, 1],
                         [1, 1]],
                        [[1, 2],
                         [1, 2],
                         [0, 0]]], dtype=np.float32)
    kernel5 = np.array([[[1, 2],
                         [0, 1],
                         [1, 2]],
                        [[0, 0],
                         [1, 1],
                         [1, 1]],
                        [[1, 2],
                         [1, 2],
                         [0, 0]],
                        [[0, 2],
                         [1, 1],
                         [2, 2]],
                        [[1, 1],
                         [1, 1],
                         [1, 1]]], dtype=np.float32)

    demo_input = np.array([[[5, 2, 3, 4], [7, 3, 4, 5], [9, 4, 5, 6], [9, 5, 6, 7],
                            [9, 4, 5, 6], [13, 2, 4, 5], [44, 3, 5, 6], [55, 4, 6, 7],
                            [1, 5, 3, 4], [1, 6, 4, 5], [1, 2, 5, 6], [1, 3, 6, 7],
                            [1, 4, 3, 4], [1, 5, 4, 5], [1, 6, 5, 6], [1, 2, 6, 7],
                            [5, 3, 3, 4], [7, 4, 4, 5], [9, 5, 5, 6], [9, 6, 6, 7],
                            [9, 2, 3, 4], [13, 3, 4, 5], [44, 4, 5, 6], [55, 5, 6, 7],
                            [1, 6, 3, 4], [1, 2, 4, 5], [1, 3, 5, 6], [1, 4, 6, 7],
                            [1, 5, 3, 4], [1, 6, 4, 5], [1, 2, 5, 6], [1, 3, 6, 7]]], dtype=np.float32)

    demo_kernel = np.array([[[1, 0, 2, 2],
                             [2, 1, 0, 0],
                             [3, 0, 1, 5],
                             [4, 1, 1, 6]],

                            [[2, 4, 6, 8],
                             [3, 2, 0, 0],
                             [4, 5, 2, 2],
                             [5, 1, 0, 0]],

                            [[3, 7, 0, 5],
                             [4, 4, 4, 2],
                             [5, 3, 0, 0],
                             [6, 1, 1, 1]]], dtype=np.float32)
    print('case1:')
    test(feature, kernel2, ks=2, stride=1, padding='SAME')
    print('case2:')
    test(feature, kernel2, ks=2, stride=2, padding='SAME')
    print('case3:')
    test(feature, kernel3, ks=3, stride=1, padding='SAME')
    print('case4:')
    test(feature, kernel3, ks=3, stride=2, padding='SAME')
    print('case 5:')
    test(feature, kernel5, ks=5, stride=1, padding='SAME')
    print('case demo')
    conv1d_res = tf.nn.conv1d(demo_input, demo_kernel, 2, 'SAME')
    print(conv1d_res)
