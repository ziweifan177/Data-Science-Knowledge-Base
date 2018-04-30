import tensorflow as tf
import numpy as np

# W=tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
# b=tf.Variable([[1,2,3]], dtype=tf.float32, name='biases') # Create a rank 2 array
# #dtype=tf.float32, is required!
# #names is optional.
#
# saver=tf.train.Saver()
#
# #Save:
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     save_path=saver.save(sess, 'logs/saver/saver.ckpt')
#     print('Save to path:', save_path)

############################Restore##################################
#!!!! Must define the same dtype & shape when restore.
#Define a empty variables for loading:

W=tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
b=tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases') #Rank2 Array, not rank 1.

saver=tf.train.Saver()

#No need to initialize.
with tf.Session() as sess:
    saver.restore(sess, 'logs/saver/saver.ckpt')
    print('Weights:', sess.run(W))
    print('Biases:', sess.run(b))