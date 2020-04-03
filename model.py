
import tensorflow as tf
import numpy as np



class SRresidual:
	def __init__(self):
		pass

	def forward(self,x,n_layer):
		x_ =x
		for i in range(n_layer-1):
			# w = self._weight(shape=[3,3,3,64])
			x = self._conv_layer(x)
			print(x)
		
		x = tf.layers.conv2d(x, kernel_size=3, filters=1, strides=1, padding='same', use_bias=True)
		x = tf.keras.layers.ReLU()(x)
		x = x-x_
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


	def optmizer(self,loss_function):

		opt = tf.train.AdamOptimizer(learning_rate=1e-5)

		gvs = opt.compute_gradients(loss_function)
		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		train_op = opt.apply_gradients(capped_gvs)

		return train_op