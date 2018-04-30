import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist data: Train data & Test data.
mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

'''Step0: Define functions: add layers.'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return

def compute_accuracy(v_xs, v_ys): #sValidation_xs & validation_ys. Return a percentage.
    global prediction
    y_pre=sess.run(prediction, feed_dict={xs:v_xs})# Train by validation data to get prediction value.
    # Compare prediction y & validation y:
    # prediction_y: percentage of possibility of each output.
    # validation_y: the '1' position is the real result.
    # tf.argmax(): 返回的是vector中的最大值的索引号;
    # tf.argmax(y_pre,1)返回的是模型对于任一输入x预测到的标签值
    # tf.argmax(ys,1) 代表正确的标签；
    # tf.equal() 来检测g预测是否真实标签匹配,这行代码返回的是匹配的布尔值，成功1，失败0
    print("y_pre: ", y_pre) # y_pre: is a 10*sample numbers matrix: possibility percentage matrix.
    print("v_ys:", v_ys) #v_ys: is a 2D 10*sample numbers matrix: one-hot.

    correct_prediction=tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) #tf.argmax(x, axis): is a tensor: Tensor("ArgMax_2:0", shape=(10000,), dtype=int64)
    print("tf.argmax(y_pre,1):\n", tf.argmax(y_pre,1))
    print("tf.argmax(v_ys,1):\n", tf.argmax(v_ys,1))
    print("correct_prediction:\n", correct_prediction)

    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#accuracy: Tensor("Mean_1:0", shape=(), dtype=float32)
    print("accuracy:\n", accuracy)

    result=sess.run(accuracy, feed_dict={xs: v_xs, ys:v_ys})
    print("accuracy In running session:\n", accuracy) #accuracy: Tensor("Mean_1:0", shape=(), dtype=float32)
    return result #a float percentage.

'''Step1: Define placeholders for input to network'''
xs=tf.placeholder(tf.float32, [None, 784]) # [A, B]: A: Sample number; B: pixel number of each sample.
ys=tf.placeholder(tf.float32, [None,10]) #10:possible number of outputs

'''Step2: Add output layer'''
# prediction=add_layer(xs, 784, 10, activation_function=tf.nn.softmax())#Classification: Use softmax typically.
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

'''Step3: The error between prediction and real data'''
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) #Loss in clasification: cross_entropy
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)# Train step.

'''Step4: Initialize.'''
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)#Retrive 100 records to be learned for time reduction.
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys}) #Train it with 'train data'.

    if i%50==0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels)) #Accurate it with 'test data'.
