import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# construction phase
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('fc_1'):
    W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
    h = tf.sigmoid(tf.matmul(x, W1) + b1)
    tf.histogram_summary('layer1_weights', W1)

with tf.name_scope('fc_2'):
    W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y_predict = tf.nn.softmax(tf.matmul(h, W2) + b2)

with tf.name_scope('eval'):
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_predict), reduction_indices=[1]))
        tf.scalar_summary('loss', cross_entropy)
    backprop = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# execution phase
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("graphs", sess.graph)
sess.run(tf.initialize_all_variables())

train_steps = 20000
batch_size = 50
for i in range(train_steps):
    batch_x, batch_y = data.train.next_batch(batch_size)
    if i % 10 == 0:
        summary, _ = sess.run([merged, backprop], feed_dict={x: batch_x, y: batch_y})
        writer.add_summary(summary, i)
    else:
        sess.run(backprop, feed_dict={x: batch_x, y: batch_y})

print(sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels}))
