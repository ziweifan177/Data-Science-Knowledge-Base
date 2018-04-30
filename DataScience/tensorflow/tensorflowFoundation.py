import tensorflow as tf
import numpy as np


#Step1: Create data
x_data=np.random.rand(100).astype(np.float32)#type in TF is always float.
y_data=x_data*0.1+0.3

#Step2: Create tensorflow structure
Weights=tf.Variable(tf.random_uniform([1],-0.1,0.1))#Generate weight vector in 1 dimension.
#(Outputs random values from a uniform distribution.)
biases = tf.Variable(tf.zeros([1]))#Generate Biases in zeros.

y=Weights*x_data+biases

init=tf.global_variables_initializer()

loss=tf.reduce_mean(tf.square(y-y_data))

initializer=tf.train.GradientDescentOptimizer(0.5) # Kind of Optimizer here.

train=initializer.minimize(loss)

#Initialize

sess=tf.Session()

sess.run(init) #!!!!!!!!!!!!!!!

for step in range(200):
    sess.run(train) #!!!!!!!!!!!!!!!!!!!!!
    if step%20==0:
        print(step, sess.run(Weights), sess.run(biases))
#Print weight, bias trained every 20 steps.
sess.close()