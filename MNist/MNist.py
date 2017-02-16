#from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Parameter
logs_path = 'c:/board/example_mnist'
x = tf.placeholder(tf.float32, [None, 784], name='Input')
y_ = tf.placeholder(tf.float32, [None, 10], name='Label')

W = tf.Variable(tf.zeros([784, 10]), name='Weight')
b = tf.Variable(tf.zeros([10]), name='Bias')

# Model
with tf.name_scope('Model'):
    y = tf.nn.softmax(tf.matmul(x, W) + b)
with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
with tf.name_scope('SGD'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
 
tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, c, a, summary = sess.run([train_step, cross_entropy, correct_prediction, merged_summary_op], feed_dict={x:batch_xs, y_:batch_ys})
        summary_writer.add_summary(summary, i)

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
 