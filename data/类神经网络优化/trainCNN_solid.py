# -*- coding: utf-8 -*-
"""
Created on Wed May  9 18:21:54 2018

@author: TomatoSir
"""

import pickle
from keras.regularizers import l2
import time
import numpy as np
from sklearn import preprocessing 
import keras
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers import Bidirectional
import os
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers import Bidirectional
from datetime import datetime
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')


def trainCNN_solid(inputsize):
    # inputsize = 36  #指标数

    dataX = np.empty(shape=[0,20,inputsize])
    dataY = np.empty(shape=[0,1])
    # 保证X和Y读取顺序一样，按名称排序
    for maindir, subdir, file_name_list in os.walk("./resultX"):
            for filename in file_name_list:
                eachpath = os.path.join(maindir, filename)
                try:
                    with open(eachpath, 'rb') as f:
                        dataX_temp = pickle.load(f)
                    dataX = np.vstack((dataX, dataX_temp))
                    print(filename + " is successfully load")
                except:
                    # 引发错误的原因是因为数据量不够
                     print(filename + " has some mistakes")

    for maindir, subdir, file_name_list in os.walk("./resultY"):
            for filename in file_name_list:
                eachpath = os.path.join(maindir, filename)
                try:
                    with open(eachpath, 'rb') as f:
                        dataY_temp = pickle.load(f)
                    dataY = np.vstack((dataY, dataY_temp))
                    print(filename + " is successfully load")
                except:
                    # 引发错误的原因是因为数据量不够
                     print(filename + " has some mistakes")

    # 数据进行归一化
    for i in range(dataX.shape[0]):
        dataX[i] = preprocessing.scale(dataX[i], axis=0)
    # 变换格式
    dataX=dataX.reshape(-1,20,inputsize,1)

    # print(dataY[0:5,:])
    # y_all1 = keras.utils.to_categorical(dataY, num_classes=3)
    # print(y_all1[0:5,:])

    # 随机排列数据
    permutation = np.random.permutation(dataX.shape[0])
    dataX = dataX[permutation, :, :]
    dataY = dataY[permutation,:]

    y_all = keras.utils.to_categorical(dataY, num_classes=3)

    SplitIndex = round(0.8*dataX.shape[0])
    x_train = dataX[:SplitIndex,:,:]
    y_train = y_all[:SplitIndex,:]
    x_test = dataX[SplitIndex:,:,:]
    y_test = y_all[SplitIndex:,:]

    #x_train = dataX
    #y_train = y_all

    # 构建模型 这里选择双向GRU模型
    model = Sequential()
    model.add(Convolution2D(
        input_shape = (20,inputsize,1),
        filters = 32,
        kernel_size = (20,inputsize),
        strides = 1,
        padding = 'same',
        activation = 'relu',
        W_regularizer=l2(0.0003)
    ))
    # 第一个池化层
    model.add(MaxPooling2D(
        pool_size = 2,
        strides = 2,
        padding = 'same',
    ))
    # 第二个卷积层
    model.add(Convolution2D(64,(20,inputsize),strides=1,padding='same',activation = 'relu',W_regularizer=l2(0.0003)))
    # 第二个池化层
    model.add(MaxPooling2D(2,2,'same'))
    # 把第二个池化层的输出扁平化为1维
    model.add(Flatten())
    # 第一个全连接层
    model.add(Dense(1024,activation = 'relu'))
    # Dropout
    model.add(Dropout(0.5))
    # 第二个全连接层
    model.add(Dense(3,activation='softmax'))

    # 定义优化器
    adam = Adam(lr=1e-4)
    # 定义优化器，loss function，训练过程中计算准确率
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


    # 训练模型
    start = time.clock()
    model.fit(x_train, y_train,  validation_split=0.1, batch_size=128, epochs=20)
    elapsed = (time.clock() - start)
    print("耗时:", elapsed)
    ## 保存训练好的模型
    print("saving model")
    version = datetime.now().strftime("%m-%d-%H-%M")
    model_name = "CNN"
    result_file = "model_solid/{}_time{}.h5"
    model.save(result_file.format(model_name, version))
    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)
    print("model,name:", result_file.format(model_name, version))
    return result_file.format(model_name, version)

#trainCNN_solid(15)