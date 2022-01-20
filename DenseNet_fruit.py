# -*-coding: utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import opencv
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
file_path = r'D:\Artificial_Intellegence_Project\Practical\Fruit\fruit\Training\\'

# claim variables
xs = tf.placeholder(tf.float32, [None, 100, 100, 3])
ys = tf.placeholder(tf.float32, [None, 50])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0)

x_image = tf.reshape(xs, [-1, 100, 100, 3])


def weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def biases(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def dense(stage, block, inputs, growth_rate=32, keep_prob=keep_prob):
    global list_i
    block_name = 'Des'+str(stage)+str(block)
    with tf.variable_scope(block_name):
        try:
            x_input = tf.layers.batch_normalization(inputs, axis=2)
        except Exception:
            x_input = tf.layers.batch_normalization(tf.concat(inputs, axis=3))
        x_relu = tf.nn.relu(x_input)

        # 1*1 convolution
        conv1 = slim.conv2d(x_relu, growth_rate*4, 1, stride=1, padding='SAME', activation_fn=None)
        conv1_dropout = tf.nn.dropout(conv1, keep_prob=keep_prob)
        # 3*3 convolution
        conv1_relu = tf.nn.relu(tf.layers.batch_normalization(conv1_dropout, axis=3))
        conv2 = slim.conv2d(conv1_relu, growth_rate, 3, stride=1, padding='SAME', activation_fn=None)
        # print(list_i)
        conv2_dropout = tf.nn.dropout(conv2, keep_prob=keep_prob)

        list_i.append(conv2_dropout)
        return conv2_dropout


def transition(stage, block, drop=0.5):
    global list_i
    block_name = 'Des'+str(stage)+str(block)
    with tf.variable_scope(block_name):
        x_input = tf.layers.batch_normalization(tf.concat(list_i, axis=3), axis=3)

        # 1*1 convolution
        conv = slim.conv2d(x_input, int(x_input.shape[3])*drop, 1, stride=1, padding='SAME', activation_fn=None)
        # average pooling
        h1 = tf.nn.avg_pool(conv, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        list_i = [h1]
        return h1


# stage 1
conv0 = slim.conv2d(x_image, 64, 7, stride=2, padding='SAME', activation_fn=None)
conv0_relu = tf.nn.relu(tf.layers.batch_normalization(conv0, axis=3))
h1 = tf.nn.max_pool(conv0_relu, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
list_i = [h1]
# print(h1.shape)
# print(len(list_i))

# stage 2
des_21 = dense(2, 1, list_i)
des_22 = dense(2, 2, list_i)
des_23 = dense(2, 3, list_i)
des_24 = dense(2, 4, list_i)
des_25 = dense(2, 5, list_i)
des_26 = dense(2, 6, list_i)

t1 = transition(2, 7)

# stage 3
des_31 = dense(3, 1, t1)
des_32 = dense(3, 2, list_i)
des_33 = dense(3, 3, list_i)
des_34 = dense(3, 4, list_i)
des_35 = dense(3, 5, list_i)
des_36 = dense(3, 6, list_i)
des_37 = dense(3, 7, list_i)
des_38 = dense(3, 8, list_i)
des_39 = dense(3, 9, list_i)
des_310 = dense(3, 10, list_i)
des_311 = dense(3, 11, list_i)
des_312 = dense(3, 12, list_i)

t2 = transition(3, 13)

# stage 4
des_41 = dense(4, 1, list_i)
des_42 = dense(4, 2, list_i)
des_43 = dense(4, 3, list_i)
des_44 = dense(4, 4, list_i)
des_45 = dense(4, 5, list_i)
des_46 = dense(4, 6, list_i)
des_47 = dense(4, 7, list_i)
des_48 = dense(4, 8, list_i)
des_49 = dense(4, 9, list_i)
des_410 = dense(4, 10, list_i)
des_411 = dense(4, 11, list_i)
des_412 = dense(4, 12, list_i)
des_413 = dense(4, 13, list_i)
des_414 = dense(4, 14, list_i)
des_415 = dense(4, 15, list_i)
des_416 = dense(4, 16, list_i)
des_417 = dense(4, 17, list_i)
des_418 = dense(4, 18, list_i)
des_419 = dense(4, 19, list_i)
des_420 = dense(4, 20, list_i)
des_421 = dense(4, 21, list_i)
des_422 = dense(4, 22, list_i)
des_423 = dense(4, 23, list_i)
des_424 = dense(4, 24, list_i)

t3 = transition(4, 25)

# stage 5
des_51 = dense(5, 1, list_i)
des_52 = dense(5, 2, list_i)
des_53 = dense(5, 3, list_i)
des_54 = dense(5, 4, list_i)
des_55 = dense(5, 5, list_i)
des_56 = dense(5, 6, list_i)
des_57 = dense(5, 7, list_i)
des_58 = dense(5, 8, list_i)
des_59 = dense(5, 9, list_i)
des_510 = dense(5, 10, list_i)
des_511 = dense(5, 11, list_i)
des_512 = dense(5, 12, list_i)
des_513 = dense(5, 13, list_i)
des_514 = dense(5, 14, list_i)
des_515 = dense(5, 15, list_i)
des_516 = dense(5, 16, list_i)

# stage 6
h = tf.layers.batch_normalization(des_516, axis=3)
h_pool = tf.nn.avg_pool(h, [1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
h1_flatten = tf.reshape(h_pool, [-1, 1*1*32])
# print(h1_flatten.shape)
# fc layer
w_fc1 = weight([1*1*32, 50])
b_fc1 = biases([50])

h_fc1 = tf.nn.dropout(tf.matmul(h1_flatten, w_fc1)+b_fc1, keep_prob=keep_prob)
prediction = tf.nn.softmax(h_fc1)
# print(prediction)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction+1e-3), reduction_indices=[1]))

learning_rate = tf.train.exponential_decay(1e-3, global_step, staircase=True, decay_rate=0.96, decay_steps=2000)
train_step = tf.train.AdamOptimizer(learning_rate, epsilon=0.1).minimize(cross_entropy, global_step)

# compute accuracy
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

img, label = opencv.file_name(file_path)
label_op = sess.run(tf.one_hot(label, depth=50))

index = [i for i in range(len(img))]
random.shuffle(index)

print('training-----------------------------------------------')
for i in range(2000):
    update = tf.assign(global_step, i)
    if i >= 502:
        a = i % 502
    else:
        a = i
    img_batch = np.array(img)[index[a*50: a*50+50]]
    label_batch = np.array(label_op)[index[a*50: a*50+50]]
    sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch, keep_prob: 0.5})
    sess.run(update)
    if (i+1) % 100 == 0:
        print((i+1), sess.run(accuracy, feed_dict={xs: img_batch, ys: label_batch, keep_prob: 1}))
        # save_path = saver_1.save(sess, check_dir, global_step=0)

file_path = r'D:\Artificial_Intellegence_Project\Practical\Fruit\fruit\Test\\'
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
img, label = opencv.file_name(file_path)
label_op = sess.run(tf.one_hot(label, depth=50))

print('testing=================================================')
for a in range(254):
    label_batch = np.array(label_op)[index[a*50: a*50+50]]
    img_batch = np.array(img)[index[a*50: a*50+50]]
    if (a+1) % 50 == 0:
        print(sess.run(accuracy, feed_dict={xs: img_batch, ys: label_batch, keep_prob: 1}))

sess.close()
