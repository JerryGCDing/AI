# -*-coding: utf-8-*-
import numpy as np
import opencv
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
file_path = r'D:\Artificial_Intellegence_Project\Practical\Fruit\fruit\Training\\'

# claim variables
xs = tf.placeholder(tf.float32, [None, 100, 100, 3])
ys = tf.placeholder(tf.float32, [None, 50])
# keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0)

x_image = tf.reshape(xs, [-1, 100, 100, 3])


# weight
def weights(shape):
    init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)


# biases
def biases(shape):
    init = tf.constant(0.02, shape=shape)
    return tf.Variable(init)


# identity layer
def identity(inputs, out_size, k_size, stage, block):
    x_short_cut = inputs
    block_name = 'res'+str(stage)+str(block)
    with tf.variable_scope(block_name):

        # convolution layer 1
        conv1 = slim.conv2d(inputs, out_size, k_size, stride=1, padding='SAME', activation_fn=None)
        conv1_output = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

        # convolution layer 2
        conv2 = slim.conv2d(conv1_output, out_size, k_size, stride=1, padding='SAME', activation_fn=None)
        conv2_BN = tf.layers.batch_normalization(conv2, axis=3)
        conv2_output = tf.nn.relu(conv2_BN+x_short_cut)

    return conv2_output


# convolution layer
def conv(inputs, out_size, k_size, stage, block):
    x_short_cut = inputs
    block_name = 'res'+str(stage)+str(block)
    with tf.variable_scope(block_name):

        # convolution layer 1
        conv1 = slim.conv2d(inputs, out_size, k_size, stride=2, padding='SAME', activation_fn=None)
        conv1_output = tf.nn.relu(tf.layers.batch_normalization(conv1, axis=3))

        # convolution layer 2
        conv2 = slim.conv2d(conv1_output, out_size, k_size, stride=1, padding='SAME', activation_fn=None)
        conv2_output = tf.layers.batch_normalization(conv2, axis=3)

        # input reshape
        input_conv = slim.conv2d(x_short_cut, out_size, k_size, stride=2, padding='SAME', activation_fn=None)
        input_reshape = tf.layers.batch_normalization(input_conv, axis=3)

        # output
        output = tf.nn.relu(input_reshape+conv2_output)

    return output


# stage 1
conv1 = slim.conv2d(x_image, 64, 7, stride=2, padding='VALID')
conv1_relu = tf.nn.relu(tf.layers.batch_normalization(conv1))
h1_pool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# stage 2
id_2_1 = identity(h1_pool, 64, 3, 2, 1)
id_2_2 = identity(id_2_1, 64, 3, 2, 2)
id_2_3 = identity(id_2_2, 64, 3, 2, 3)

# stage 3
conv_3_1 = conv(id_2_3, 128, 3, 3, 1)
id_3_2 = identity(conv_3_1, 128, 3, 3, 2)
id_3_3 = identity(id_3_2, 128, 3, 3, 3)
id_3_4 = identity(id_3_3, 128, 3, 3, 4)

# stage 4
conv_4_1 = conv(id_3_4, 256, 3, 4, 1)
id_4_2 = identity(conv_4_1, 256, 3, 4, 2)
id_4_3 = identity(id_4_2, 256, 3, 4, 3)
id_4_4 = identity(id_4_3, 256, 3, 4, 4)
id_4_5 = identity(id_4_4, 256, 3, 4, 5)
id_4_6 = identity(id_4_5, 256, 3, 4, 6)


# stage 5
conv_5_1 = conv(id_4_6, 512, 3, 5, 1)
id_5_2 = identity(conv_5_1, 512, 3, 5, 2)
id_5_3 = identity(id_5_2, 512, 3, 5, 3)

# fc layer
h_pool = tf.nn.avg_pool(id_5_3, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

h_pool_flaten = tf.reshape(h_pool, [-1, 3*3*512])

# stage 6
w_fc1 = weights([3*3*512, 50])
b_fc1 = biases([50])
h_fc1 = tf.matmul(h_pool_flaten, w_fc1)+b_fc1
prediction = tf.nn.softmax(h_fc1)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

# learning rate decay
learning_rate = tf.train.exponential_decay(1e-3, global_step, staircase=True, decay_rate=0.96, decay_steps=20000)

# train step
train_step = tf.train.AdamOptimizer(learning_rate, epsilon=0.1).minimize(cross_entropy, global_step=global_step)
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
img, label = opencv.file_name(file_path)
label_op = sess.run(tf.one_hot(label, depth=50))

index = [i for i in range(len(img))]
random.shuffle(index)

print('training-----------------------------------------------')
for i in range(1500):
    update = tf.assign(global_step, i)
    if i >= 502:
        a = i % 502
    else:
        a = i
    img_batch = np.array(img)[index[a*50: a*50+50]]
    label_batch = np.array(label_op[index[a*50: a*50+50]])
    sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch})
    sess.run(update)
    if (i+1) % 10 == 0:
        print((i+1), sess.run(accuracy, feed_dict={xs: img_batch, ys: label_batch}))
        # save_path = saver_1.save(sess, check_dir, global_step=0)

file_path = r'D:\Artificial_Intellegence_Project\Practical\Fruit\fruit\Test\\'
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
img, label = opencv.file_name(file_path)
label_op = sess.run(tf.one_hot(label, depth=50))

print('testing=================================================')
for a in range(169):
    label_batch = np.array(label_op[index[a*50: a*50+50]])
    img_batch = np.array(img)[index[a*50: a*50+50]]
    if (a+1) % 10 == 0:
        print(sess.run(accuracy, feed_dict={xs: img_batch, ys: label_batch}))

sess.close()
