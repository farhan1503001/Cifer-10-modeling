# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:32:53 2018

@author: LENOVO
"""

import keras as k
(x_train,y_train),(x_test,y_test)=k.datasets.cifar10.load_data()
labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#plt.imshow(x_train[3])
#plt.show()
x_train=k.utils.normalize(x_train)
x_test=k.utils.normalize(x_test)
x_train_temp=x_train[0:20000,:,:,:]
y_train_temp=y_train[0:20000,:]
from sklearn.model_selection import train_test_split
x_train1,x_val,y_train1,y_val=train_test_split(x_test,y_test,test_size=0.1)
input_image=k.Input((32,32,3))
x=k.layers.Conv2D(64,(3,3),strides=(1,1),activation='relu',name='conv1')(input_image)
x=k.layers.MaxPool2D(pool_size=(2,2),name='max1')(x)
x=k.layers.BatchNormalization()(x)
#2nd_layer 
x=k.layers.Conv2D(128,(4,4),strides=(1,1),padding='valid',activation='relu',name='conv2')(x)
x=k.layers.MaxPool2D(pool_size=(2,2),name='max2')(x)
x=k.layers.BatchNormalization()(x)
#3rd layer
"""
x=k.layers.Conv2D(256,(3,3),strides=(1,1),padding='valid',activation='relu',name='conv3')(x)
x=k.layers.MaxPool2D(pool_size=(2,2))(x)
x=k.layers.BatchNormalization()(x)
"""
#4th layer
x=k.layers.Flatten()(x)
"""
x=k.layers.Dense(256,activation='relu',name='dense1',kernel_regularizer=k.regularizers.l2())(x)
x=k.layers.BatchNormalization()(x)
x=k.layers.Dropout(rate=0.2)(x)
"""
#5th layer
"""
x=k.layers.Dense(512,activation='relu',name='dense2',kernel_regularizer=k.regularizers.l2())(x)
x=k.layers.BatchNormalization()(x)
x=k.layers.Dropout(rate=0.3)(x)
"""
#Final layer
out=k.layers.Dense(10,activation='softmax',name='dense3')(x)

model=k.models.Model(inputs=input_image,outputs=out)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_temp,y_train_temp,batch_size=64,epochs=5,validation_data=(x_val,y_val))
val_error,val_accuracy=model.evaluate(x_test[0:2000,:,:,:],y_test[0:2000,:])