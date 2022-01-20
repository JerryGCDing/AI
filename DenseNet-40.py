# -*-coding: utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import opencv
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.625)
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


def dense(stage, block, inputs, growth_rate=12, internal_layer=True, keep_prob=keep_prob):
    block_name = 'Des'+str(stage)+str(block)
    with tf.variable_scope(block_name):
        x_input = tf.layers.batch_normalization(inputs, axis=3)
        x_relu = tf.nn.relu(x_input)

        # 3*3 convolution
        conv1_relu = tf.nn.relu(tf.layers.batch_normalization(x_relu, axis=3))
        conv2 = slim.conv2d(conv1_relu, growth_rate, 3, stride=1, padding='SAME', activation_fn=None)
        # print(list_i)
        conv2_dropout = tf.nn.dropout(conv2, keep_prob=keep_prob)

        if internal_layer == True:
            output = tf.concat([inputs, conv2_dropout], axis=3)
        else:
            output = conv2_dropout
        return output


def transition(inputs, stage, block, drop=0.5):
    block_name = 'Des'+str(stage)+str(block)
    with tf.variable_scope(block_name):
        x_input = tf.layers.batch_normalization(inputs, axis=3)

        # 1*1 convolution
        conv = slim.conv2d(x_input, int(x_input.shape[3])*drop, 1, stride=1, padding='SAME', activation_fn=None)
        # average pooling
        h1 = tf.nn.avg_pool(conv, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return h1


# stage 1
conv0 = slim.conv2d(x_image, 64, 7, stride=2, padding='SAME')
conv0_relu = tf.nn.relu(tf.layers.batch_normalization(conv0, axis=3))
h1 = tf.nn.max_pool(conv0_relu, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# stage 2
des_21 = dense(2, 1, h1)
des_22 = dense(2, 2, des_21)
des_23 = dense(2, 3, des_22)
des_24 = dense(2, 4, des_23)
des_25 = dense(2, 5, des_24)
des_26 = dense(2, 6, des_25)

t1 = transition(des_26, 2, 7)

# stage 3
des_31 = dense(3, 1, t1)
des_32 = dense(3, 2, des_31)
des_33 = dense(3, 3, des_32)
des_34 = dense(3, 4, des_33)
des_35 = dense(3, 5, des_34)
des_36 = dense(3, 6, des_35)
des_37 = dense(3, 7, des_36)
des_38 = dense(3, 8, des_37)
des_39 = dense(3, 9, des_38)
des_310 = dense(3, 10, des_39)
des_311 = dense(3, 11, des_310)
des_312 = dense(3, 12, des_311)

t2 = transition(des_312, 3, 13)

# stage 4
des_41 = dense(4, 1, t2)
des_42 = dense(4, 2, des_41)
des_43 = dense(4, 3, des_42)
des_44 = dense(4, 4, des_43)
des_45 = dense(4, 5, des_44)
des_46 = dense(4, 6, des_45)
des_47 = dense(4, 7, des_46)
des_48 = dense(4, 8, des_47)

t3 = transition(des_48, 4, 9)

# stage 5
des_51 = dense(5, 1, t3)
des_52 = dense(5, 2, des_51)
des_53 = dense(5, 3, des_52)
des_54 = dense(5, 4, des_53)
des_55 = dense(5, 5, des_54)
des_56 = dense(5, 6, des_55)
des_57 = dense(5, 7, des_56)
des_58 = dense(5, 8, des_57)
des_59 = dense(5, 9, des_58, internal_layer=False)

# stage 6
h = tf.layers.batch_normalization(des_59, axis=3)
h_pool = tf.nn.avg_pool(h, [1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
h1_flatten = tf.reshape(h_pool, [-1, 1*1*12])

# fc layer
w_fc1 = weight([1*1*12, 50])
b_fc1 = biases([50])

h_fc1 = tf.matmul(h1_flatten, w_fc1)+b_fc1
prediction = tf.nn.softmax(h_fc1)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

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
    sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch, keep_prob: 0.8})
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
