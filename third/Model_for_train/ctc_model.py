from keras.models import *
from keras.layers import *
from keras import backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


input_tensor = Input((40, 50, 1))
x = input_tensor

#三层卷积池化、relu
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
#reshape层
conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

#dense_1
x = Dense(32, activation='relu')(x)

#gru_1
gru_1 = GRU(128, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(128, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)

#merge_1层
gru1_merged = merge([gru_1, gru_1b], mode='sum')

#gru_2
gru_2 = GRU(128, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(128, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
#merge_2
x = merge([gru_2, gru_2b], mode='concat')

#dropout_1
x = Dropout(0.25)(x)

#dense_2
x = Dense(44, init='he_normal', activation='softmax')(x)

labels = Input(name='the_labels', shape=[4, 11], dtype='float32')
input_length = Input(name='input_length', shape=[4, 11], dtype='int64')
label_length = Input(name='label_length', shape=[4, 11], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(4, 11), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')


