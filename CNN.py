# -*-coding: utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape, layer):
    initial = tf.truncated_normal(shape, stddev=0.1, name='w'+layer)
    return tf.Variable(initial)


def biases_variable(shape, layer):
    initial = tf.constant(0.1, shape=shape, name='b'+layer)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_poo_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# claim variables
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
ys = tf.placeholder(tf.float32, [None, 10])  # 10 numbers
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0)

# inputs
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# hidden convolution layer 1
w_conv1 = weight_variable([5, 5, 1, 32], '1')  # patch_size, patch_size, feature_map(i)/channel, feature_map(o)
b_conv1 = biases_variable([32], '1')  # feature_map output
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)  # convolution(inputs*weight)+ biases
# pooling operation
h_pool1 = max_poo_2x2(h_conv1)  # shape[28, 28, 1] >> [14, 14, 32]

# hidden convolution layer 2
w_conv2 = weight_variable([7, 7, 32, 64], '2')
b_conv2 = biases_variable([64], '2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
# pooling operation
h_pool2 = max_poo_2x2(h_conv2)  # shape [14, 14, 32] >> [7, 7, 64]
# [7, 7, 64] >> [7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# full-connected layer 1
w_fc1 = weight_variable([7*7*64, 1024], '2')
b_fc1 = biases_variable([1024], '2')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# full-connected layer 2(output)
w_fc2 = weight_variable([1024, 10], '2')
b_fc2 = biases_variable([10], '2')
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2)+b_fc2)
# Exponential decay
learning_rate = tf.train.exponential_decay(1e-2, global_step, staircase=True, decay_rate=0.03, decay_steps=5500)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# compute accuracy
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# saver
saver = tf.train.Saver()
try:
    saver.restore(sess, 'my_mnist_net/save_net.ckpt')
except Exception:
    pass
# train step
for i in range(5500):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
    update = tf.assign(global_step, i)
    sess.run(update)
    if (i+1) % 50 == 0:
        print(sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 0.5}))
    save_path = saver.save(sess, 'my_mnist_net/save_net.ckpt', global_step=0)

sess.close()
