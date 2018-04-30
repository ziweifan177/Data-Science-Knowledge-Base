from __future__ import print_function
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

def add_layer(input, in_size, out_size, name_layer, activation_function=None):

    layer_name='layer%s' % name_layer

    with tf.name_scope(layer_name):
        with tf.name_scope('Weight'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='inner_weight')
            tf.summary.histogram(layer_name+'/Weights', Weights)

        with tf.name_scope('Biases'):
            biases=tf.Variable(tf.zeros([1, out_size])+0.1 ,name='inner_bias')
            tf.summary.histogram(layer_name+'/Biases', biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.add(tf.matmul(input, Weights),biases)

        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', outputs)

        return outputs

'''Make up some real data'''
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
lay1=add_layer(x_placeholder,1,10,name_layer=1, activation_function=tf.nn.relu)

#add output layer:
prediction=add_layer(lay1, 10,1,name_layer=2, activation_function=None) #output: 1, input: 10

#Define the loss(error between prediction and real y):
with tf.name_scope('Loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_placeholder), reduction_indices=[1]), name='inner_loss')
#reduction_indices: The old (deprecated) name for axis.
#tf.reduce_sum adds the elements in the second dimension, due to the reduction_indices=[1]
    tf.summary.scalar('loss',loss) #Generate loss into 'Event' of Tensorboard.

#Define how to minimize the loss to improve:
with tf.name_scope('Train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.Session()

merged=tf.summary.merge_all() #Merge all the histogram and scalar(events) definition into graph.
writer = tf.summary.FileWriter("logs/", sess.graph)
#Check the plot: To the specific path, use:
#cd C:\Learning\Python\python\
# tensorboard --logdir='logs'

#Initialize all variables:
sess.run(tf.global_variables_initializer())

'''Visualize the progress of learning:'''
for i in range(1000):
    sess.run(train_step, feed_dict={x_placeholder: x_data, y_placeholder: y_data})

    if i%50==0:
        result=sess.run(merged, feed_dict={x_placeholder: x_data, y_placeholder: y_data})
        writer.add_summary(result, i)
        #write all the summaries into file every 50 steps.

sess.close()





