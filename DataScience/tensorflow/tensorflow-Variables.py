import tensorflow as tf

#Variables & Operation:
state=tf.Variable(0,name='counter') #Define a variable in this way.
one=tf.constant(1) #Define a constant in this way.

new_value=tf.add(state,one)#Operate add: in this way.
update=tf.assign(state, new_value) #Assign: in this way.

init=tf.global_variables_initializer() #!!!!!!!!!!!!!!!!!Have to initialize all the variables.

#RUN:
with tf.Session() as sess:
    sess.run(init) #!!!!!!!!!!!!!!!!!!!! Have to: Activate the init
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
