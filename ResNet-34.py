# -*-coding: utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# claim variables
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0)

x_image = tf.reshape(xs, [-1, 28, 28, 1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def biases_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define Building Blocks
def identity(inputs, out_size, stage, block):
    x_shout_cut = inputs
    block_name = 'res'+str(stage)+str(block)
    with tf.variable_scope(block_name):

        # convolution layer 1
        conv1 = slim.conv2d(inputs, out_size, 3, stride=1, padding='SAME')
        output_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

        # convolution layer 2
        conv2 = (slim.conv2d(output_conv1, out_size, 3, stride=1, padding='SAME'))
        conv2_BN = tf.layers.batch_normalization(conv2, axis=3)
        output_conv2 = tf.nn.relu(conv2_BN+x_shout_cut)

    return output_conv2


def change(inputs, out_size, stage, block):
    x_short_cut = inputs
    block_name = 'res'+str(stage)+str(block)
    with tf.variable_scope(block_name):

        # convolution layer 1
        conv1 = slim.conv2d(inputs, out_size, 3, stride=2, padding='SAME')
        output_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

        # convolution layer 2
        conv2 = slim.conv2d(output_conv1, out_size, 3, stride=1, padding='SAME')
        output_conv2 = tf.layers.batch_normalization(conv2, axis=3)

        # input reshape
        input_conv = slim.conv2d(x_short_cut, out_size, 3, stride=2, padding='SAME')
        input_reshape = tf.layers.batch_normalization(input_conv, axis=3)

        # output
        output = tf.nn.relu(output_conv2+input_reshape)

    return output


# stage 1
conv1 = slim.conv2d(x_image, 64, 7, stride=2, padding='VALID')
conv1_norm = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))
h1_pool = tf.nn.max_pool(conv1_norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# stage 2
id_2_1 = identity(h1_pool, 64, 2, 1)
id_2_2 = identity(id_2_1, 64, 2, 2)
id_2_3 = identity(id_2_2, 64, 2, 3)

# stage 3
conv_3_1 = change(id_2_3, 128, 3, 1)
id_3_2 = identity(conv_3_1, 128, 3, 2)
id_3_3 = identity(id_3_2, 128, 3, 3)
id_3_4 = identity(id_3_3, 128, 3, 4)

# stage 4
conv_4_1 = change(id_3_4, 256, 4, 1)
id_4_2 = identity(conv_4_1, 256, 4, 2)
id_4_3 = identity(id_4_2, 256, 4, 3)
id_4_4 = identity(id_4_3, 256, 4, 4)
id_4_5 = identity(id_4_4, 256, 4, 5)
id_4_6 = identity(id_4_5, 256, 4, 6)

# stage 5
conv_5_1 = change(id_4_6, 512, 5, 1)
id_5_2 = identity(conv_5_1, 512, 5, 2)
id_5_3 = identity(id_5_2, 512, 5, 3)

# pooling
# h_pool = tf.nn.avg_pool(id_5_3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')  # [1, 1, 512]
h_pool_flaten = tf.reshape(id_5_3, [-1, 1*1*512])
# print(h_pool_flaten)

# stage 6
# full connect
w_fc1 = weight_variable([1*1*512, 10])
b_fc1 = biases_variable([10])
h_fc1 = tf.nn.dropout(tf.matmul(h_pool_flaten, w_fc1)+b_fc1, keep_prob=keep_prob)
prediction = tf.nn.softmax(h_fc1)
# print(prediction)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))  # loss
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train step
learning_rate = tf.train.exponential_decay(0.1, global_step, staircase=True, decay_rate=0.96, decay_steps=5000)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step)

sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 1})
    update = tf.assign(global_step, i)
    sess.run(update)
    if (i+1) % 100 == 0:
        print((i+1), sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1}))

sess.close()
