# -*-coding: utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def biases(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def identity(inputs, out_size, ksize, stage, block):
    x_short_cut = inputs
    block_name = 'res'+str(stage)+str(block)
    with tf.variable_scope(block_name):

        # convolution layer 1
        conv1 = slim.conv2d(inputs, out_size, ksize, stride=1, padding='SAME')
        conv1_output = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

        # convolution layer 2
        conv2 = slim.conv2d(conv1_output, out_size, ksize, stride=1, padding='SAME')
        conv2_BN = tf.layers.batch_normalization(conv2, axis=3)
        conv2_output = tf.nn.relu(conv2_BN+x_short_cut)

    return conv2_output


def conv(inputs, out_size, ksize, stage, block):
    x_short_cut = inputs
    block_name = 'res'+str(stage)+str(block)
    with tf.variable_scope(block_name):

        # convolution layer 1
        conv1 = slim.conv2d(inputs, out_size, ksize, stride=2, padding='SAME')
        conv1_output = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

        # input reshape
        x_conv = slim.conv2d(x_short_cut, out_size, ksize, stride=2, padding='SAME')
        x_reshape = tf.layers.batch_normalization(x_conv, axis=3)

        # convolution layer 2
        conv2 = slim.conv2d(conv1_output, out_size, ksize, stride=1, padding='SAME')
        conv2_output = tf.layers.batch_normalization(conv2, axis=3)

        output = tf.nn.relu(x_reshape+conv2_output)

    return output


# claim variables
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
global_step = tf.Variable(0)
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])

# main()
# stage 1
conv1 = slim.conv2d(x_image, 64, 7, stride=2, padding='VALID')
conv1_relu = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

# stage 2
h1_pool = tf.nn.max_pool(conv1_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
id_2_1 = identity(h1_pool, 64, 3, 2, 1)
id_2_2 = identity(id_2_1, 64, 3, 2, 2)

# stage 3
conv_3_1 = conv(id_2_2, 128, 3, 3, 1)
id_3_2 = identity(conv_3_1, 128, 3, 3, 2)

# stage 4
conv_4_1 = conv(id_3_2, 256, 3, 4, 1)
id_4_2 = identity(conv_4_1, 256, 3, 4, 2)

# stage 5
conv_5_1 = conv(id_4_2, 512, 3, 5, 1)
id_5_2 = identity(conv_5_1, 512, 3, 5, 2)

# h_pool = tf.nn.avg_pool(id_5_2, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
h_flaten = tf.reshape(id_5_2, [-1, 1*1*512])

# fc layer
w_fc1 = weight([1*1*512, 10])
b_fc1 = biases([10])
prediction = tf.nn.softmax(tf.matmul(h_flaten, w_fc1)+b_fc1)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
# train step
learning_rate = tf.train.exponential_decay(0.1, global_step, staircase=True, decay_rate=0.96, decay_steps=5000)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step)

correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(5000):
    img_batch, label_batch = mnist.train.next_batch(100)
    update = tf.assign(global_step, i)
    sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch})
    sess.run(update)
    if (i+1) % 100 == 0:
        print((i+1), sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels}))

sess.close()
