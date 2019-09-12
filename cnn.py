#__author:
import numpy as np
import  os
import tensorflow as tf
from read import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#读取数据
trX, trY, teX, teY = readdata()

#超参数
batch_size = 150
test_size=2500

g_CNN = tf.Graph()
with g_CNN.as_default():
    # Y_all = tf.constant(loady[:], dtype=tf.float32)
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape,stddev=0.01))
    def Model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden):
        lla  = tf.nn.relu(tf.nn.conv2d(X,w,strides=[1,1,1,1],padding="SAME"))
        ll = tf.nn.max_pool(lla,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        ll = tf.nn.dropout(ll,p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(ll, w2, strides=[1, 1, 1, 1], padding="SAME"))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        l2 = tf.nn.dropout(l2, p_keep_conv)

        l3a  = tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding="SAME"))
        l3 = tf.nn.max_pool(l3a,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        l3 = tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
        l3 = tf.nn.dropout(l3,p_keep_conv)

        l4  =  tf.nn.relu(tf.matmul(l3,w4))
        l4 = tf.nn.dropout(l4,p_keep_hidden)

        pyx = tf.matmul(l4,w_o)
        return pyx
    def loss(Y_Model,Y_ture):
        return tf.square(Y_Model-Y_ture)
    def Verify(tey,pre_y,i,test_size = 2500):
        num = 0
        for j in range(test_size):
            if tey[j] ==pre_y[j]:
                num +=1
        return print("第%d次:精度为%.2f%%" %(i ,float(num*100/test_size)))

    X = tf.placeholder('float',[None,28,28,1])
    Y = tf.placeholder("float",[None,10])

    w = init_weights([3,3,1,16])
    w2 = init_weights([3,3,16,32])
    w3 = init_weights([3,3,32,64])
    w4 = init_weights([64*4*4,256])
    w_o = init_weights([256,10])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    py_x = Model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden)#预测值

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x,labels=Y))
    COST = loss(py_x,Y)
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    predict_op = tf.argmax(py_x,1)

def CNN():
    with tf.Session(graph = g_CNN) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(200):
    # 训练 先打包 再一次次训练（填数据，自动校正 W 参数）
            training_batch = zip(range(0,len(teX),batch_size),
                                       range(batch_size,len(teX)+1,batch_size))
            for start ,end in training_batch:
                sess.run(train_op,feed_dict={
                    X:trX[start:end],Y:trY[start:end],p_keep_hidden : 0.7,p_keep_conv : 0.9})
    #验证
            pre_y = onehot(sess.run(predict_op, feed_dict={X: teX[:],p_keep_hidden:0.9,p_keep_conv :1}))
            Verify(teY,pre_y,i,test_size)
    #测试
    #略
    
    #模型保存
        tf.model_variables()

if __name__ == '__main__':
    CNN()
