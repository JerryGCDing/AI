# -*-coding: utf-8-*-
import cv2
import os
# import tensorflow as tf
import numpy as np
import copy
import random

file_path = r'/YeDisk/home/tyding/Fruit/fruit/Test/'


def file_name(file_dir):
    list0 = []
    list_label = []
    dict0 = {}
    i = -1
    for files in os.listdir(file_dir):
        # print(files)
        file = os.path.join(file_dir, files)
        # print(file)
        for root, dirs, files_ in os.walk(file):
            i = i+1
            for name in files_:
                list0.append(os.path.join(root, name))
                list_label.append(i)
    list_img = []
    # print(len(list0))
    for filename in list0:
        image = (cv2.imread(filename, 1).astype(np.float32))/255
        # image = tf.reshape(image, [100, 100, 3])
        list_img.append(image)

    return list_img, list_label


if __name__ == '__main__':
    # sess = tf.Session()
    a, b = file_name(file_path)
    # print(sess.run(tf.one_hot(b, depth=50))[np.newaxis, :])
    img = copy.copy(a)
    random.shuffle(img)
    print(img[1] in a)
