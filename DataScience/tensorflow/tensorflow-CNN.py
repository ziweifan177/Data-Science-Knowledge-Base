import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

'''Define Variables'''
xs=tf.placeholder(tf.float32, [None, 784])# 28*28
ys=tf.placeholder(tf.float32, [None,10])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1])#-1表示任意数量的样本数,大小为28x28深度为一的张量;1代表Channel: 黑白为1，RGB为3.

sess=tf.Session()

'''Define Functions:'''
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre=sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1), tf.argmax(ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result=sess.run(accuracy, feed_dict={xs:v_xs, ys: v_ys, keep_prob:1})
    return result

def weight_variable(shape):
    #stddev: Deviation: 标准差能反映一个数据集的离散程度.
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    # stride [1, x_movement, y_movement, 1]: the two 1 is required.
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    #ksize: The size of the window for each dimension of the input tensor.
    #ksize: 池化窗口的大小，参数为四维向量，通常取[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1。
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

'''Define convolutional layers:'''
#Convolutional Layer1:
W_conv1=weight_variable([5,5,1,32]) #patch: 5*5, in size:1(image的厚度), out size: 32（新image的高度）.
b_conv1=bias_variable([32])#=Output size
h_conv1= tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)#hiden layer：output size 28x28x32
h_pool1=max_pool_2x2(h_conv1)# output size 14x14x32, because stride=2, so:28/2

#Convolutional Layer2:
W_conv2=weight_variable([5,5,32,64]) #in size=last output size, out size=64
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2) # output size 7x7x64

#Full connected Layer1:
W_fc1=weight_variable([7*7*64, 1024]) #More thicker
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1, 7*7*64]) #把池化层2的输出扁平化为1维
#[n_samples,7,7,64]-->[n_samples,7*7*64], -1: 先不管sample数量。
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)

#Full connected Layer2:
W_fc2=weight_variable([1024, 10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

#Loss:
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss

#Train:
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#Initialize:
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i%50==0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))