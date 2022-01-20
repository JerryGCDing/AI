import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.625)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

xs = tf.placeholder(tf.float32, [None, 100, 100, 3])
ys = tf.placeholder(tf.float32, [None, 50])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_2d(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')


def dense(stage, block, inputs, growth_rate=32, keep_prob=keep_prob):
    global list_i
    try:
        x_input = tf.layers.batch_normalization(inputs, axis=3)
    except Exception:
        x_input = tf.layers.batch_normalization(tf.concat(inputs, axis=3))
    x_relu = tf.nn.relu(x_input)

    # 1*1 convolution
    conv1 = conv_2d(x_input, weight_variable([1, 1, 1, growth_rate*4]))
    conv1_dropout = tf.nn.dropout(conv1, keep_prob=keep_prob)
    # 3*3 convolution
    conv1_relu = tf.nn.relu(tf.layers.batch_normalization(conv1_dropout, axis=3))
    conv2 = conv_2d(conv1_relu, weight_variable([3, 3, 1, growth_rate]))
    # print(list_i)
    conv2_dropout = tf.nn.dropout(conv2, keep_prob=keep_prob)

    list_i.append(conv2_dropout)
    return conv2_dropout
