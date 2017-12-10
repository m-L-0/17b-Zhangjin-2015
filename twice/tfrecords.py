import os
import numpy as np
import tensorflow as tf
from PIL import Image

path = 'C:\\Users\\JIN\\Desktop\\machine\\homework2\\data\\车牌字符识别训练数据'
i = [];j = [];k = []
for _, dirnames, filenames in os.walk(path):
    if len(_)!=54 and _[-6:]!='letter' and _[-3:]!='num' and _[-4:]!='word':
        i+=[_]
    if dirnames!=[]:
        j+=[dirnames]
    if filenames!=[]:
        k+=[filenames]
dict_all = {}
tb = 0
for heads in j[1:]:
    for head in heads:
        dict_all.update({head:[]})
        for back in k[tb]:
            dirs = i[tb]+'\\'+back
            dict_all[head]+=[dirs]
        tb+=1
dict_dict = {'0': 0,'1': 1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'川': 11,'鄂': 12,'甘': 13,'赣': 14,'桂': 16,'贵': 17,'黑': 18,'沪': 19,'吉': 20,'冀': 21,'津': 22,'晋': 23,'京': 24,'辽': 25,'鲁': 26,'蒙': 27,'闽': 28,'宁': 29,'青': 30,'琼': 31,'陕': 32,'苏': 33,'皖': 34,'湘': 35,'渝': 37,'豫': 38,
'粤': 39,'云': 40,'浙': 41,'A': 42,'B': 43,'C': 44,'D': 45,'E': 46,'F': 47,'G': 48,'H': 49,'J': 51,'K': 52,'L': 53,'M': 54,'N': 55,'P': 57,'Q': 58,'R': 59,'S': 60,'T': 61,'U': 62,'V': 63,'W': 64,'X': 65,'Y': 66,'Z': 67}   
writer = tf.python_io.TFRecordWriter("all_s.tfrecords")
for labels in dict_all.keys():
    for imgs in dict_all[labels]:
        img = Image.open(imgs)
        #img = img.convert('1')
        img = img.resize((24,48), Image.ANTIALIAS)
        images = img.tobytes()
        try:
            label = dict_dict[labels]
        except:
            print('在字典中不能正常获得标签')
        example = tf.train.Example(features = tf.train.Features(feature = {
                         "label": tf.train.Feature(int64_list=tf.train.Int64List(value = [label])),
                         'img': tf.train.Feature(bytes_list=tf.train.BytesList(value = [images]))
                         })) 
        writer.write(example.SerializeToString())
writer.close()


dicts_dicts={}
for key,value in dict_dict.items():
    dicts_dicts.update({value:key})
	
def read_tfrecord(path):
    filename_queue = tf.train.string_input_producer([path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img' : tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img'], tf.uint8)
    image = tf.cast(image, dtype='float32')*(1/255)-0.5 
    image = tf.reshape(image, [48, 24, 3])
    image = tf.split(image, 3, 2)[0]
    #image = tf.split(image, num_or_size_splits=3, axis=1)[0]
    label = tf.cast(features['label'], tf.int32)
    return image, label

import matplotlib.pyplot as plt
image, label = read_tfrecord("all_s.tfrecords")
image_batch, label_batch = tf.train.shuffle_batch([image,label], batch_size=1, capacity=6000, min_after_dequeue=5999, num_threads=2) 
init = tf.local_variables_initializer()
ii=[]
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator() #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)   #启动QueueRunner, 此时文件名队列已经进队
    for i in range(2):  # 规定出队数量
        img, label = sess.run([image_batch, label_batch])
        print(label.shape,img.shape)
        for j in range(1):
            print(dicts_dicts[label[j]])
            im = img[j].reshape(48,24)
            plt.imshow(im)
            plt.show()
    coord.request_stop()
    coord.join(threads)
	
ii = []
jj = []
labelss = []
image, label = read_tfrecord("all_s.tfrecords")
#image_batch, label_batch = tf.train.shuffle_batch([image,label], batch_size=1, capacity=2000, min_after_dequeue=0, num_threads=2) 
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator() #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)   #启动QueueRunner, 此时文件名队列已经进队
    for i in range(18447):  # 规定出队数量
        img, labeless = sess.run([image, label])
        #for j in range(1):
        im = img.reshape(48,24)
        ii += [im]
        jj += [dicts_dicts[labeless]]
        labelss += [labeless]
    coord.request_stop()
    coord.join(threads)

label_count={}
letter_label_count = {}
for kk in dict_dict.keys():
    if kk not in ['I','O','广','新','藏']:
        label_count.update({kk:jj.count(kk)})

for kk in dicts_dicts.keys():
    if kk not in [50,56,15,10,36]:
        letter_label_count.update({kk:jj.count(dicts_dicts[kk])})
#print(label_count)

#coding:utf-8  
import matplotlib  
matplotlib.use('qt4agg')  
#指定默认字体  
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   
matplotlib.rcParams['font.family']='sans-serif'  
plt.figure(1, figsize=(20,20))    
#expl = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  
# Colors used. Recycle if not enough.  
colors  = ["blue","red","coral","green","yellow","orange"]  
# autopct: format of "percent" string;  
plt.pie(list(label_count.values()),  colors=colors, labels=list(label_count.keys()), autopct='%1.1f%%',pctdistance=0.8, shadow=True)  
plt.show()

import matplotlib.pyplot as plt
plt.figure(2, figsize=(20,20))  
plt.bar(list(letter_label_count.keys()),list(letter_label_count.values()),color='rgb')
ll = 0
for iii in dicts_dicts.keys():
    plt.text(list(letter_label_count.keys())[ll],list(letter_label_count.values())[ll],dicts_dicts[iii])
    ll+=1
plt.grid()
plt.show()

c = list(zip(jj,ii))
import random
random.shuffle(c)
jj[:],ii[:]=zip(*c)
In [10]:

xtrain = ii[:15000]
ytrain = jj[:15000]
xtest = ii[15000:18000]
ytest = jj[15000:18000]
vtrain = ii[18000:18447]
vtest = jj[18000:18447]
xtrain = [ii[k].reshape(1152) for k in range(15000)]
xtest = [ii[k].reshape(1152) for k in range(15000,18000)]
vtrain = [ii[k].reshape(1152) for k in range(18000,18447)]

def knn_simple(xtrain, ytrain, xtest, ytest):
    start = time.time()
    xtr = tf.placeholder(tf.float32, [None, 1152])
    xte = tf.placeholder(tf.float32, [1152])
    
    distance = Euclidean_distance(xtr,xte)
    
    pred = tf.argmin(distance, 0)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    right = 0
    for i in range(3000):
        if i % 100==0 and i !=0:
            print("已处理 {0}，正确率为{1}".format(i,right/i))
        #distance = sess.run(distance,{xtr: xtrain, xte: xtest[i]})
        index = sess.run(pred,{xtr: xtrain, xte: xtest[i]})
        #print(sess.run(distance[10036]))
        if np.argmax(ytest[i]) == np.argmax(ytrain[index]):	#这里的后半部分酌情更改
            right += 1.0
    print('总共处理%d 用时%f' %(3000,time.time()-start))
    print('正确数量%d 正确率为%f' %(right,right/3000))
	
def Euclidean_distance(xtr,xte):
    return tf.sqrt(tf.reduce_sum(tf.pow((xtr-xte), 2), reduction_indices=1))
import time
#knn_simple(xtrain, ytrain, xtest, ytest)


lbels = np.zeros([18447,68])
for i in range(len(jj)):
    lbels[i][dict_dict[jj[i]]]=1
for i in range(18447):
    ii[i] = ii[i].reshape(1152)
xtrain = ii[:15000]
xtest = ii[15000:18000]
vtrain = ii[18000:18447]
ytrain = lbels[:15000]
ytest = lbels[15000:18000]
vtest = lbels[18000:18447]

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
	
sess = tf.InteractiveSession()

x_ = tf.placeholder(tf.float32, [None, 1152], name='image')
y = tf.placeholder(tf.float32, [None, 68], 'label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
x = tf.reshape(x_, [-1, 48, 24, 1], name='x')


w1 = init_weights([3, 3, 1, 32])
b1 = init_bias([32])
conv1 = tf.nn.relu(tf.nn.conv2d(x, w1,  strides=[1, 1, 1, 1], padding='SAME') + b1)  # shape=(?, 48, 24, 32)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # # shape=(?, 24, 12, 32)



w2 = init_weights([3, 3, 32, 64])
b2 = init_bias([64])
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w2,  strides=[1, 1, 1, 1], padding='SAME') + b2)  # shape=(?, 24, 12, 64)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # shape=(?, 12, 6, 64)

w3 = init_weights([3, 3, 64, 96])    
b3 = init_bias([96])
conv3 = tf.nn.relu(tf.nn.conv2d(pool2, w3,  strides=[1, 1, 1, 1], padding='SAME') + b3)  # shape=(?, 12, 6, 96)
pool3_ = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # shape=(?, 6, 3, 96)


pool3 = tf.reshape(pool3_, [-1, 6*3*96])

w4 = init_weights([6 * 3 * 96, 512])
b4 = init_bias([512])
h = tf.nn.relu(tf.matmul(pool3, w4) + b4)

h = tf.nn.dropout(h, keep_prob)

w_o = init_weights([512, 68])
b_o = init_bias([68])
y_o = tf.matmul(h, w_o) + b_o


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_o, labels=y))
train_op = tf.train.AdamOptimizer().minimize(cost)
prediction = tf.equal(tf.argmax(y_o, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

#image_batch, label_batch = tf.train.shuffle_batch([xtrain,ytrain], batch_size=50, capacity=2000, min_after_dequeue=1990, num_threads=2)
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
bs = 0
saver = tf.train.Saver() 
for i in range(1500):
    train_data_x = xtrain[bs:bs+50]
    train_data_y = ytrain[bs:bs+50]
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x_: train_data_x, y: train_data_y, keep_prob: 1.0})
        print("step %d, accuracy:%g" % (i, train_accuracy))
    train_op.run(feed_dict={x_: train_data_x, y: train_data_y, keep_prob: 0.3})
    bs+=10
saver.restore(sess, r"./Model/model.ckpt")

train_data_x = xtrain
train_data_y = ytest
train_accuracy = sess.run(accuracy, feed_dict={x_: train_data_x, y: train_data_y, keep_prob: 1.0})
print("step, accuracy: %g" % (train_accuracy))
