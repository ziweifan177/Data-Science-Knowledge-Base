import tensorflow as tf
from sklearn.datasets import load_digits #0-9 digit picture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#LabelBinarizer: one-hot

'''Step 0.1: Function definition: Add layer'''
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1,)
    Wx_plus_b=tf.matmul(inputs, Weights)+biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

'''Step 0.2: Placeholder definition: For inputs'''
xs=tf.placeholder(tf.float32, [None, 64]) #8*8
ys=tf.placeholder(tf.float32, [None, 10]) # output: 10
keep_prob=tf.placeholder(tf.float32) #Placeholder for the percentage will be kept

'''Step 1: Define: data to be loaded'''
digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y) # If y=1: LabelBinarizer is to put 1 in the second position.
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=3) #Split X--> X_train, X_test

'''Step 2: Define: output Layer'''
l1=add_layer(xs, 64, 50, 'Layer1', activation_function=tf.nn.tanh)
prediction=add_layer(l1, 50, 10, 'Layer2', activation_function=tf.nn.softmax)

'''Step 3: Define: Cross Entropy(loss in classification)'''
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

'''Step 4: Define: Train step'''
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''Step 5: Define: Session'''
sess=tf.Session()

'''Step 6: Define: Component in tensorboard'''
tf.summary.scalar('loss', cross_entropy)
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter("logs/train", sess.graph)
test_writer=tf.summary.FileWriter("logs/test", sess.graph)

'''Step 7: Initialize'''
sess.run(tf.global_variables_initializer())

'''Step 8: Dropout & Train'''
for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob: 0.3}) #ADD: Keep percentage

    if i%50==0:
        train_result=sess.run(merged, feed_dict={xs: X_train, ys:y_train, keep_prob:0.3}) #For comparation:
        test_result=sess.run(merged, feed_dict={xs: X_test, ys:y_test, keep_prob:1})

        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

sess.close()




