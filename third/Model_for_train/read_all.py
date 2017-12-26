# coding:utf-8

import tensorflow as tf
import numpy as np
from keras.models import *
from keras.layers import *
import csv

csv_reader = csv.reader(open(r'H:/homework3/data/captcha/labels/labels.csv', encoding='utf-8'))
labels = []  # 存放图片的标签
for row in csv_reader:
    labels += [row[1]]

def read_tfrecord(path):
    filename_queue = tf.train.string_input_producer([path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img': tf.FixedLenFeature([40, 50], tf.float32),
                                       })
    image = tf.cast(features['img'], tf.float64)
    label = tf.cast(features['label'], tf.int32)
    return image, label

def read(path):
    images_a = []
    labels_a = []
    im, la = read_tfrecord(path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(4000):
            val, l = sess.run([im, la])
            val = np.reshape(val, [2000])
            images_a += [val]
            labels_a += [int(l)]

    return images_a, labels_a


def train(input_x, input_y):
    input_x = np.reshape(input_x, [-1, 40, 50, 1])
    x = tf.placeholder("float", shape=[None, 40, 50, 1], name='x')
    y = tf.placeholder("float", shape=[None, 4, 11], name='y')
    keep_prob = tf.placeholder("float", name='keep_prob')

    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1), name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
    # 第一层卷积激活池化
    h_conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    h_relu1 = tf.nn.relu(h_conv1)
    h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [-1,20,25,32]

    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv2')
    # 第二层卷积激活池化
    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
    h_relu2 = tf.nn.relu(h_conv2)
    h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [-1, 10, 14, 32]

    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='W_conv3')
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv3')
    # 第三层卷积激活池化
    h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
    h_relu3 = tf.nn.relu(h_conv3)
    h_pool3 = tf.nn.max_pool(h_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [-1, 5, 7, 64]

    print('pool3', h_pool3.get_shape())

    W_fc1 = tf.Variable(tf.truncated_normal([5 * 7 * 64, 1024], stddev=0.1), name='W_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
    # reshape成全连接层，dropout避免过拟合
    h_fc1 = tf.matmul(tf.reshape(h_pool3, [-1, 5 * 7 * 64]), W_fc1) + b_fc1
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc21 = tf.Variable(tf.truncated_normal([1024, 11], stddev=0.1), name='W_fc21')
    b_fc21 = tf.Variable(tf.constant(0.1, shape=[11]), name='b_fc21')

    W_fc22 = tf.Variable(tf.truncated_normal([1024, 11], stddev=0.1), name='W_fc22')
    b_fc22 = tf.Variable(tf.constant(0.1, shape=[11]), name='b_fc22')

    W_fc23 = tf.Variable(tf.truncated_normal([1024, 11], stddev=0.1), name='W_fc23')
    b_fc23 = tf.Variable(tf.constant(0.1, shape=[11]), name='b_fc23')

    W_fc24 = tf.Variable(tf.truncated_normal([1024, 11], stddev=0.1), name='W_fc24')
    b_fc24 = tf.Variable(tf.constant(0.1, shape=[11]), name='b_fc24')
    # 分成四个全连接层，分别对应输出四位
    h_fc21 = tf.matmul(h_fc1, W_fc21) + b_fc21
    h_fc22 = tf.matmul(h_fc1, W_fc22) + b_fc22
    h_fc23 = tf.matmul(h_fc1, W_fc23) + b_fc23
    h_fc24 = tf.matmul(h_fc1, W_fc24) + b_fc24

    print('h_fc21', h_fc21.get_shape())
    # 整合结果
    y_conv = tf.stack([h_fc21, h_fc22, h_fc23, h_fc24], 1)

    print('y_conv', y_conv.get_shape())
    print('y', y.get_shape())
    # 求损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    # 求准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(y, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    epochs = 50
    batch_size = 200

    global_step = tf.Variable(0, trainable=False)
    #每10个迭代降一次学习率
    learning_rate = tf.train.exponential_decay(0.0001,
                                               global_step,
                                               (input_x.shape[0] / batch_size + 1) * 10,
                                               0.96,
                                               staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l_la = 0
        for i in range(1780):
            x_ = input_x[l_la:40 + l_la]
            y_ = input_y[l_la:40 + l_la]
            train_step.run(feed_dict={x: x_, y: y_, keep_prob: 0.3})
            if i % 100 == 0 and i != 0:
                accuracys = sess.run(accuracy, feed_dict={x: x_, y: y_, keep_prob: 1.0})
                print('处理%d,正确率为%g' % (i * 40, accuracys))
            l_la += 20
        saver = tf.train.Saver()
        saver.save(sess, "./save/model.ckpt")

    return y


def turn_lab(pp='0'):
    ndarr_lab = np.zeros([4000, 4, 11])
    strs=int(pp)
    la_qie = labels[4000*strs:4000*(strs+1)]
    for i in range(len(la_qie)):
        ndarr_lab[i][0][int(la_qie[i][0])] = 1
        try:
            ndarr_lab[i][1][int(la_qie[i][1])] = 1
        except:
            ndarr_lab[i][1][10] = 1
            ndarr_lab[i][2][10] = 1
            ndarr_lab[i][3][10] = 1
        try:
            ndarr_lab[i][2][int(la_qie[i][2])] = 1
        except:
            ndarr_lab[i][2][10] = 1
            ndarr_lab[i][3][10] = 1
        try:
            ndarr_lab[i][3][int(la_qie[i][3])] = 1
        except:
            ndarr_lab[i][3][10] = 1
    return ndarr_lab
# images_8, labels_8 = read('H:/flask_/tfrecords/all8.tfrecords')
# labels_8 = turn_lab(pp='8')
#
# labels_all = np.vstack((labels_all,labels_8))
# images_all += images_8