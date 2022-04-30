import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    batch_size = 1
    input_width = 12
    kernel_size = 2
    input_channel = 3
    output_channel = 2
    stride_size = 2
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
                        [0, 1, 1]], dtype=np.float32).reshape([batch_size, input_width, input_channel])
    kernel = np.array([[[1, 2],
                        [0, 1],
                        [1, 2]],
                       [[0, 0],
                        [1, 1],
                        [1, 1]]], dtype=np.float32).reshape([kernel_size, input_channel, output_channel])
    print(feature)
    print(kernel)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=2, kernel_size=kernel_size, strides=stride_size, use_bias=False,
                                     input_shape=(input_width, input_channel), name='conv1d_1'))
    print(model.get_layer('conv1d_1').get_weights())
    model.get_layer('conv1d_1').set_weights([kernel])
    output = model.predict(feature)
    print(output)
    conv1d_res = tf.nn.conv1d(feature, kernel, stride_size, 'VALID')
    print(conv1d_res)
    # bias = np.array(np.arange(1, 10), dtype=np.float32).reshape([1, 9, 1])
    # conv1d_res = tf.add(conv1d_res, bias)
    # print(conv1d_res)
