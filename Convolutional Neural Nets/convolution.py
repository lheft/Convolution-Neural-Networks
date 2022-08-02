from __future__ import absolute_import

import os
from re import L
from unittest.mock import patch
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_out_channels = filters.shape[3]

	strideY = 1
	strideX = 1


	
	# Cleaning padding input
	if padding == 'SAME':
		padY=(filter_height-1)/2
		padX= (filter_width-1)/2
		padY=int(padY)
		padX=int(padX)
	else:
		padY=0
		padX=0


	output_height=(in_height + 2*padY - filter_height)/strideY + 1
	output_width=(in_width + 2*padX - filter_width)/strideX + 1
	output_height=math.floor(output_height)
	output_width=math.floor(output_width)
	output_shape=(num_examples,output_height,output_width,filter_out_channels)
	output=np.zeros(output_shape)

	for h in range (num_examples):
		padded_i= np.pad(inputs[h], ((padY,),(padX,),(0,)), mode='constant', constant_values=0.0)
	
		for e in range (output_height):		
		
			for f in range(output_width):
				
				for t in range(filter_out_channels):
					current_filter = filters[:,:,:,t]
					total = 0
					sum_filter_h= e + filter_height
					sum_filter_w= f + filter_width
					patch_img=padded_i[e : sum_filter_h, f : sum_filter_w]
					patch_new= np.reshape(patch_img, current_filter.shape)
					conv_patch_filter= patch_new * current_filter
					total= np.sum(conv_patch_filter)
					output[h][e][f][t]=total

					#imgPatch = padded_i[e:(e+filter_height), f:(filter_width+f)]
					#reshaped_patch = np.reshape(imgPatch, current_filter.shape)
					#total = np.sum(np.multiply(reshaped_patch,current_filter))
					#output[h][e][f][t] = total
	
	return tf.convert_to_tensor(output, dtype = tf.float32)


	# Calculate output dimensions

	# PLEASE RETURN A TENSOR. HINT: tf.convert_to_tensor(your_array, dtype = tf.float32)

def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output

	return

if __name__ == '__main__':
	main()
