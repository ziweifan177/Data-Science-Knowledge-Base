from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input, in_size, out_size, activation_function=None):
    with tf.name_scope('Layer'):
        with tf.name_scope('Weight'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='inner_weight')
        with tf.name_scope('Biases'):
            biases=tf.Variable(tf.zeros([1, out_size])+0.1 ,name='inner_bias')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.add(tf.matmul(input, Weights),biases)

        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs


'''Create a small neural network:'''

#Define input, output,noise
x_data=np.linspace(-1,1,300)[:,np.newaxis]
# linspace:Return evenly spaced numbers over a specified interval(numbers).
#[:,np.newaxis]: create an axis of length one.
noise=np.random.normal(0,0.05,x_data.shape) #Add some noise.
y_data=np.square(x_data)-0.5+noise #Actuall value.

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    x_placeholder=tf.placeholder(tf.float32, [None,1],name='x_input')
    y_placeholder=tf.placeholder(tf.float32, [None,1],name='y_input')

#add hiden layer:layer1,is a 10 outputs layer.
lay1=add_layer(x_placeholder,1,10,activation_function=tf.nn.relu)

#add output layer:
prediction=add_layer(lay1, 10,1,activation_function=None) #output: 1, input: 10

#Define the loss(error between prediction and real y):
with tf.name_scope('Loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_placeholder), reduction_indices=[1]), name='inner_loss')
#reduction_indices: The old (deprecated) name for axis.
#tf.reduce_sum adds the elements in the second dimension, due to the reduction_indices=[1]

#Define how to minimize the loss to improve:
with tf.name_scope('Train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#Initialize all variables:
init=tf.global_variables_initializer()

sess=tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)
#Check the plot: To the specific path, use:
#cd C:\Learning\Python\python\
# tensorboard --logdir='logs'

sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # Plot real x and y.
plt.ion()
plt.show()

#Train:
import time
time.sleep(2)

'''Visualize the progress of learning:'''
for i in range(1000):
    sess.run(train_step, feed_dict={x_placeholder: x_data, y_placeholder: y_data})

    if i%50==0:
        try:
            ax.lines.remove(lines[0]) #Remove the lines firstly.
        except Exception:
            pass

        prediction_value=sess.run(prediction, feed_dict={x_placeholder: x_data})
        lines=ax.plot(x_data, prediction_value,'r-', lw=5) # plot by red line and weight of line is 5.
        plt.pause(0.1)

sess.close()





