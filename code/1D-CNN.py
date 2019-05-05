import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Dropout,BatchNormalization
from tensorflow.keras import Sequential,utils

class CNN_1D(object):

    def __init__(self, current_data_length):
        self.current_data_length = 1194
        self.model = self.build_model()
        self.inputs = []
        self.train_labels = []
        self.test_labels = []

    def build_model(self):
        convolution_1d_layer = Conv1D(32, 5, strides=1, padding='valid', input_shape=(self.current_data_length, 1),
                                      activation="relu", name="convolution_1d_layer")
        # 定义最大化池化层
        max_pooling_layer = MaxPooling1D(pool_size=2, strides=1, padding="valid", name="max_pooling_layer")

        convolution_2d_layer = Conv1D(32, 5, strides=1, padding='valid', activation="relu", name="convolution_2d_layer")

        # 定义最大化池化层
        max_pooling2_layer = MaxPooling1D(pool_size=2, strides=1, padding="valid", name="max_pooling2_layer")

        # 平铺层，调整维度适应全链接层
        reshape_layer = Flatten(name="reshape_layer")

        # 定义全链接层
        full_connect_layer = Dense(200, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                                   bias_initializer="random_normal", use_bias=True, name="full_connect_layer")

        full_connect_layer2 = Dense(200, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                                    bias_initializer="random_normal", use_bias=True, name="full_connect_layer2")

        full_connect_layer3 = Dense(20, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                                    bias_initializer="random_normal", activation='softmax', use_bias=True,
                                    name="full_connect_layer3")

        # 编译模型
        model = Sequential()
        model.add(convolution_1d_layer)
        model.add(BatchNormalization())
        model.add(max_pooling_layer)
        model.add(Dropout(0.25))
        model.add(convolution_2d_layer)
        model.add(BatchNormalization())
        model.add(max_pooling2_layer)
        model.add(Dropout(0.25))

        model.add(reshape_layer)
        model.add(full_connect_layer)
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(full_connect_layer2)
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(full_connect_layer3)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def model_train(self):
        root_dir = './test_verify_10'
        file_starts = 'Split_inputs_'
        for ii in range(10):
            file = file_starts + str(ii) + '.npy'
            split = np.load(os.path.join(root_dir, file))
            self.inputs.append(split)

        self.train_inputs = self.inputs[0][:, :self.current_data_length, :]
        for jj in range(1, 9):
            temp_input = self.inputs[jj][:, :self.current_data_length, :]
            self.train_inputs = np.concatenate([self.train_inputs, temp_input])

        total_labels = np.load('./test_verify_10/total_labels.npy').tolist()

        self.test_inputs = self.inputs[-1][:, :self.current_data_length, :]
        self.test_labels = total_labels[-1]

        for kk in range(0, 9):
            self.train_labels.extend(total_labels[kk][:])

        self.train_labels = utils.to_categorical(self.train_labels, num_classes=20)
        self.test_labels = utils.to_categorical(self.test_labels, num_classes=20)

        self.model.fit(self.train_inputs, self.train_labels, epochs=10, shuffle=True, batch_size=32)

    def model_test(self):
        starttime = time.time()
        scores = self.model.evaluate(self.test_inputs, self.test_labels, batch_size=128, verbose=0)
        endtime = time.time()

        print("duration per example:", (endtime - starttime) / len(self.test_labels))
        print(self.model.summary())

        return scores