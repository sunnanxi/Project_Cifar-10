########cifar10数据集##########
###########保存模型############
########卷积神经网络##########
#train_x:(50000, 32, 32, 3), train_y:(50000, 1), test_x:(10000, 32, 32, 3), test_y:(10000, 1)
#60000条训练数据和10000条测试数据，32x32像素的RGB图像
#第一层两个卷积层16个3*3卷积核，一个池化层：最大池化法2*2卷积核，激活函数：ReLU
#第二层两个卷积层32个3*3卷积核，一个池化层：最大池化法2*2卷积核，激活函数：ReLU
#隐含层激活函数：ReLU函数
#输出层激活函数：softmax函数（实现多分类）
#损失函数：稀疏交叉熵损失函数
#隐含层有128个神经元，输出层有10个节点
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import classification_report
from keras import optimizers

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)

#指定GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0],True)
#初始化
plt.rcParams['font.sans-serif'] = ['SimHei']

#加载数据
cifar10 = tf.keras.datasets.cifar10
(train_x,train_y),(test_x,test_y) = cifar10.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape))

#数据预处理
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)


#建立模型
model = tf.keras.Sequential()
##特征提取阶段
#第一层
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu,data_format='channels_last',input_shape=X_train.shape[1:]))  #卷积层，16个卷积核，大小（3，3），保持原图像大小，relu激活函数，输入形状（28，28，1）
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))   #池化层，最大值池化，卷积核（2，2）
#第二层
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
##分类识别阶段
#第三层
model.add(tf.keras.layers.Flatten())    #改变输入形状
#第四层
model.add(tf.keras.layers.Dense(128,activation='relu'))     #全连接网络层，128个神经元，relu激活函数
model.add(tf.keras.layers.Dense(10,activation='softmax'))   #输出层，10个节点
print(model.summary())      #查看网络结构和参数信息

#配置模型训练方法
#损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
sgd = optimizers.SGD(lr=0.05, clipvalue=0.5) #20次
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0) #200次
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)#5次
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)#5次
model.compile(optimizer='adagrad',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#训练模型
#批量训练大小为64，迭代5/20/50/100/200次，测试集比例0.2（48000条训练集数据，12000条验证集数据）
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练前时刻：'+str(nowtime))

history = model.fit(X_train,y_train,batch_size=64,epochs=200,validation_split=0.2)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练后时刻：'+str(nowtime))

#评估模型
eval=model.evaluate(X_test,y_test,verbose=2)     #每次迭代输出一条记录，来评价该模型是否有比较好的泛化能力
print('Test test_loss:', eval[0])  #测试集损失
print('Test test_accuracy:', eval[1])  #测试集准确率
y_pred=np.argmax(model.predict(X_test), axis=1)
t = classification_report(y_test, y_pred)
print(t) #输出评估指标
#保存整个模型
model.save('CIFAR10_CNN_weights.h5')

#结果可视化
print(history.history)
loss = history.history['loss']          #训练集损失
val_loss = history.history['val_loss']  #验证集损失
acc = history.history['sparse_categorical_accuracy']            #训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy']    #验证集准确率


plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(loss,color='b',label='train')
plt.plot(val_loss,color='r',label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc,color='b',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1,10000)

    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    demo = tf.reshape(X_test[num],(1,32,32,3))
    y_pred = np.argmax(model.predict(demo))
    plt.title('标签值：'+str(test_y[num])+'\n预测值：'+str(y_pred))


#plt.ion()       #打开交互式操作模式
plt.show()
#plt.pause(5)
#plt.close()
