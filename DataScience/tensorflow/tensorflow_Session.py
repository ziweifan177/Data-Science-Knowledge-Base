import tensorflow as tf
import numpy as np

matrix1=tf.constant([[3,2]]) #Initialize a matrix with 1 row 2 columns.
matrix2=tf.constant([[4],
                     [5]]) #Initialize another matrix with 2 rows 1 column.

product=tf.matmul(matrix1, matrix2) # matmul(): Matrix multiply.

with tf.Session() as sess: #Initialize a session and close automatically.
    res=sess.run(product)
    print(res)

#Same:
# sess=tf.Session()
# res=sess.run(product)
# print(res)
# sess.close()
