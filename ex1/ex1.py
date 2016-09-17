import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

filename = "data/ex1data1.txt"
data = np.loadtxt(filename, delimiter = ",", usecols=(0,1), unpack = True)

iterations = 1500
alfa = 0.01

# Define X and y arrays
Xtr = np.transpose(np.asarray(data[:-1]))
ytr = np.transpose(np.asarray(data[-1:]))
theta = tf.Variable(0.0, name = "theta")
m = Xtr.size

def h(X,theta):
    return tf.mul(X,theta)

# Define placeholders
X = tf.placeholder('float')
Y = tf.placeholder('float')
h = h(X,theta)

cost = tf.square(h-Y)

init = tf.initialize_all_variables()
# train_op = tf.train.GradientDescentOptimizer(alfa).minimize(cost)

with tf.Session() as sess:
	sess.run(init)

	for _ in range(iterations):
		for(x, y) in zip(Xtr, ytr):
			sess.run(cost, feed_dict={X:x, Y: y})

	print(sess.run(cost))