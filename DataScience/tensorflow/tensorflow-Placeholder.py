import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

result=tf.multiply(input1, input2)

with tf.Session() as sess:
    printedResult=sess.run(result, feed_dict={input1:[7.2], input2:[3.4]})
    print(printedResult)
    #OR: print(sess.run(result, feed_dict={input1:[7.2], input2:[3.4]}))

#____________:
# tf.Variable：主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
# 1. 声明时，必须提供初始值；
# 2. 名称的真实含义，在于变量，也即在真实训练时，其值是会改变的，自然事先需要指定初始值

# tf.placeholder(dtype, shape=None, name=None)：用于得到传递进来的真实的训练样本：
# 1. 不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定；
# 2. 这也是其命名的原因所在，仅仅作为一种占位符；
# 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值