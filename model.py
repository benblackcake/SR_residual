
import tensorflow as tf

import numpy as np



class SRresidual:
	def __init__(self, learning_rate):
		self.learning_rate = learning_rate

	def forward(self,x,n_layer):

		for i in range(n_layer-1):
			# w = self._weight(shape=[3,3,3,64])
			x = self._conv_layer(x)
			print(x)
		
		x = tf.layers.conv2d(x, kernel_size=3, filters=1, strides=1, padding='same', use_bias=True)
		x = tf.keras.layers.ReLU()(x)

		return x


	def _conv_layer(self,x):
		# x = tf.nn.conv2d(x,weight,stride=[1,1,1,1],padding='SAME')
		x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same',use_bias=True)
		x = tf.keras.layers.ReLU()(x)
		return x


	def loss(self,y,y_pred):
		'''
		y is residual 
		'''
		return tf.reduce_mean(tf.square(y - y_pred))
		# return tf.nn.l2_loss(y - y_pred)


	def optmizer(self,loss_function):

		opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

		gvs = opt.compute_gradients(loss_function)
		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		train_op = opt.apply_gradients(capped_gvs)


		# global_step = tf.Variable(0, trainable=False)

		# initial_learning_rate = 1e-5

		# learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=10,decay_rate=0.9)
		# opt = tf.train.GradientDescentOptimizer(learning_rate)
		# add_global = global_step.assign_add(1)

		# with tf.control_dependencies([add_global]):
		# 	train_op = opt.minimize(loss_function)


		return train_op