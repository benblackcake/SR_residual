
import tensorflow as tf

import numpy as np



class SRresidual:
	def __init__(self, learning_rate):
		self.learning_rate = learning_rate

	def forward(self,x,n_layer):
		with tf.variable_scope('residual') as scope:
			x = tf.layers.conv2d(x, kernel_size=3, filters=1, strides=1, padding='same', use_bias=False)
			x = tf.nn.relu(x)
			skip = x
			for i in range(n_layer-1):
				# w = self._weight(shape=[3,3,3,64])
				x = self._conv_layer(x)
				print(x)
			x = tf.layers.conv2d(x, kernel_size=3, filters=1, strides=1, padding='same', use_bias=False)
			x = tf.nn.relu(x)
			x = x + skip
			
			return x


	def _conv_layer(self,x):
		# x = tf.nn.conv2d(x,weight,stride=[1,1,1,1],padding='SAME')
		# skip = x
		x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same',use_bias=False)
		x = tf.nn.relu(x)
		# x = x + skip
		return x


	def loss(self,y,y_pred):
		'''
		y is residual 
		'''
		return tf.reduce_mean(tf.square(y - y_pred))
		# return tf.nn.l2_loss(y - y_pred)


	def optmizer(self,loss_function):

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='residual')
		with tf.control_dependencies(update_ops):
			return tf.train.AdamOptimizer(self.learning_rate).minimize(loss_function, 
				var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='residual'))