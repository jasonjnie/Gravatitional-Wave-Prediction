"""
@Date: 11/07/2017

@Rui Lan
"""

import tensorflow as tf 
import h5py 
import numpy as np 
import random 

def prepare_data(inputs, labels):

	idx = np.random.permutation(inputs.shape[0])
	input_epoch = inputs[idx, :]
	label_epoch = labels[idx, :]

	return input_epoch, label_epoch 

def convert_to_tensor_float(inputs):
	return tf.convert_to_tensor(inputs, dtype=tf.float32)

def get_a_batch(inputs, labels, batch_size, num_gpus, step):
	
	#idx = np.random.randint(0, inputs.shape[0], batch_size)
	input_batch = np.zeros([batch_size, num_gpus, 8192])
	label_batch = np.zeros([batch_size, num_gpus, 2])
	

	for gpu in range(num_gpus):
		#idx = np.random.randint(0, inputs.shape[0], batch_size)
		for i in range(batch_size):
			input_batch[i, gpu, :] = inputs[step*num_gpus*batch_size+gpu*batch_size+i]
			label_batch[i, gpu, :] = labels[step*num_gpus*batch_size+gpu*batch_size+i]
	
	return input_batch, label_batch

def generator_f(data, size, snr):
    signal = np.zeros([size, 8192])
    label = np.zeros([size, 2])
    data = random.sample(data, int(size/2))
    data = np.array(data)

    for i in range(size):
        if i % 2 == 0:
            shift = random.randint(0, 1638)
            label[i] = [0, 1]
            signal[i, 0:8191-shift:1] = data[int(i/2), shift:8191:1] * snr
            signal[i] += np.random.normal(0, 1, 8192)
            signal[i] = signal[i] / np.std(signal[i])

        else:
            label[i] = [1, 0]
            signal[i] = np.random.normal(0, 1, 8192)

    return signal, label   

def generator_r(data, size, snr):
    signal = np.zeros([size, 8192])
    label = np.zeros([size, 2])
    data = random.sample(data, int(size/2))
    data = np.array(data)

    for i in range(size):
        if i % 2 == 0:
            shift = random.randint(0, 1638)
            label[i] = [0, 1]
            signal[i, 0:8191-shift:1] = data[int(i/2), shift:8191:1] * random.uniform(snr, 2)
            signal[i] += np.random.normal(0, 1, 8192)
            signal[i] = signal[i] / np.std(signal[i])

        else:
            label[i] = [1, 0]
            signal[i] = np.random.normal(0, 1, 8192)
    return signal, label  


def read_dataset(phase):
	print("<<<Loading datasets>>>")

	file = h5py.File('/projects/ncsa/grav/cs598_final_proj/split_dataset.h5', 'r')

	if phase == 'train':
		
		sig = file.get('train')
		sig = list(sig)

	else: 
		sig = file.get('test')
		sig = list(sig)

	return sig

def generate_batch_input(data, phase, snr, size):

	if phase == 'train':
		inputs, labels = generator_r(data, size, snr)
	else:
		inputs, labels = generator_f(data, size, snr)

	"""
	num_preprocess_threads = 16

	inputs_batch, labels_batch = tf.train.shuffle_batch(
		[inputs, labels],
		batch_size=size,
		num_threads=num_preprocess_threads,
		capacity=2*N, 
		min_after_dequeue=100)

	print(inputs_batch.shape)
	"""
	return inputs, labels 

def weight_variable(name, shape):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape=shape, 
			initializer=tf.contrib.layers.xavier_initializer())

	return var 

def bias_variable(name, shape):
	with tf.device('/cpu:0'):
	    var =  tf.get_variable(name, shape=shape, 
	    	initializer=tf.constant_initializer(0.1))

	return var

def inference(inputs):

	with tf.name_scope('Reshape_layer'):
		XX = tf.reshape(inputs, [-1, 8192, 1, 1])

	with tf.name_scope('Convolution_layer1'):
		w_conv1 = weight_variable('w_conv11', [16, 1, 1, 32])
		b_conv1 = bias_variable('b_conv11', [32])
		conv1 = tf.nn.conv2d(XX, w_conv1, strides=[1,1,1,1], padding='VALID')

	with tf.name_scope('Relu_layer1'):
		h_conv1 = tf.nn.relu(conv1 + b_conv1)

	with tf.name_scope('Pooling_layer1'):
		h_pool1 = tf.nn.max_pool(h_conv1, [1, 4, 1, 1], strides=[1, 4, 1, 1], 
			padding='SAME')

	with tf.name_scope('Convolution_layer2'):
		w_conv2 = weight_variable('w_conv2', [8, 1, 32, 64])
		b_conv2 = bias_variable('b_conv2', [64])
		conv2 = tf.nn.atrous_conv2d(h_pool1, w_conv2, rate=4, padding='VALID')

	with tf.name_scope('Relu_layer2'):
		h_conv2 = tf.nn.relu(conv2 + b_conv2)

	with tf.name_scope('Pooling_layer2'):
		h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], 
			padding='SAME')

	with tf.name_scope('Convolution_layer3'):
		w_conv3 = weight_variable('w_conv3', [8, 1, 64, 128])
		b_conv3 = bias_variable('b_conv3', [128])
		conv3=tf.nn.atrous_conv2d(h_pool2, w_conv3, rate=4, padding='VALID')

	with tf.name_scope('Relu_layer3'):
		h_conv3 = tf.nn.relu(conv3 + b_conv3)

	with tf.name_scope('Pooling_layer3'):
		h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], 
			padding='SAME')

	with tf.name_scope('Flatten_layer'):
		h_flatten = tf.reshape(h_pool3, [-1, 7680*2])

	with tf.name_scope('Fully_conn_1'):
		w_linear1 = weight_variable('w_linear1', [7680*2, 64])
		b_linear1 = bias_variable('b_linear1', [64])
		h_linear1 = tf.nn.relu(tf.matmul(h_flatten, w_linear1) + b_linear1)

	with tf.name_scope('Fully_conn_2'):
		w_linear2 = weight_variable('w_linear2', [64, 2])
		b_linear2 = bias_variable('b_linear2', [2])
		Y_logits = tf.matmul(h_linear1, w_linear2) + b_linear2

	with tf.name_scope('Output'):
		Y = tf.nn.softmax(Y_logits)

	return Y, Y_logits

def loss(logits, labels):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,
		name='cross_entropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
	tf.add_to_collection('losses', cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def accuracy(results, labels):
	is_correct = tf.equal(tf.argmax(results, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) * 100
	tf.add_to_collection('accuracies', accuracy)
	return tf.add_n(tf.get_collection('accuracies'), name='total_accuracy')






