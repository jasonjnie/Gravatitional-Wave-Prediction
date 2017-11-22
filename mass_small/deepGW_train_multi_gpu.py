"""
@Date: 11/07/2017
@Update: 11/11/2017
@Rui Lan
"""

import tensorflow as tf 
import deepGW
import numpy as np 
import time 
import scipy.io as sio

"""
<<<Hyperparameters>>>
"""
lr = 0.0001 			#not sure whether this matters 
snr = 0.6
train_step_size = 50
#num_epoch = 300
num_gpus = 1
log_device_placement = True
num_step = 15000

def tower_loss(scope, inputs, labels):

	inputs = deepGW.convert_to_tensor_float(inputs)
	labels = deepGW.convert_to_tensor_float(labels)

	"""
	pred, logits = deepGW.inference_mass_predictor(inputs)

	_ = deepGW.loss(logits, labels)

	_ = deepGW.accuracy(pred, labels)
	"""

	pred = deepGW.inference_mass_estimator_4conv(inputs)

	_ = deepGW.loss_estimator(pred, labels)

	_ = deepGW.accuracy_estimator(pred, labels)

	losses = tf.get_collection('losses', scope)
	accuracies = tf.get_collection('accuracies', scope)

	total_loss = tf.add_n(losses, name='total_loss')

	total_accuracy = tf.add_n(accuracies, name='total_accuracy')

	return total_loss, total_accuracy

def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):

		grads = []
		for g, _ in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)

			grads.append(expanded_g)

		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads 


def train(inputs, labels):
	with tf.device('/cpu:0'):
		global_step = tf.Variable(0, name='global_step', trainable=False)
		
		opt = tf.train.AdamOptimizer(lr)
		
		with tf.name_scope('Input'):
			X = tf.placeholder(tf.float32, [None, num_gpus, 8192])
			
		with tf.name_scope('Label'):
			Y_ = tf.placeholder(tf.float32, [None, num_gpus, 2])
			
		tower_grads = []

		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(num_gpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('tower_%d' % i) as scope:
					
						loss, accuracy = tower_loss(scope, X[:, i, :], Y_[:, i, :])
						tf.get_variable_scope().reuse_variables()
						
						grads = opt.compute_gradients(loss)
						
						tower_grads.append(grads)

		grads = average_gradients(tower_grads)
		
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		saver = tf.train.Saver()

		init = tf.global_variables_initializer()
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement))
    	
		sess.run(init)

		tf.train.start_queue_runners(sess=sess)

		# for epoch in range(num_epoch):
			
			
		# 	inputs, labels = deepGW.generate_batch_input(data=inputs, phase='train', snr=snr, size=2*len(inputs), num_epoch, epoch)

		# 	inputs, labels = deepGW.generate_batch_input_estimator(inputs, labels, 'train', snr, len(inputs), num_epoch, epoch)
		# 	num_step = inputs.shape[0] // (train_step_size * num_gpus)
		# 	input_epoch, label_epoch = deepGW.prepare_data(inputs, labels)
			
		for step in range(num_step):
			
			input_epoch, label_epoch = deepGW.generate_batch_input_estimator(inputs, labels, 'train', snr, num_gpus*train_step_size, 0, 0)
			input_batch, label_batch = deepGW.get_a_batch(input_epoch, label_epoch, train_step_size, num_gpus, 0)

			start_time = time.time()

			_, loss_value, acc_value = sess.run([apply_gradient_op, loss, accuracy], feed_dict={X: input_batch, Y_: label_batch})

			duration = time.time() - start_time

			if step % 10 == 0:

				num_examples_per_step = train_step_size * num_gpus
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration / num_gpus

				format_str = ('step %d, mse = %.5f, relative_error = %.2f (%.1f examples/sec; %.3f sec/batch)')

				print(format_str % (step, loss_value, acc_value, examples_per_sec, sec_per_batch))

				"""
				if (step + 1) == num_step:
					saver.save(sess, '/home/ruilan2/multi_gpu/ver3.ckpt', global_step=epoch)
				"""

	
	# inputs, labels = deepGW.read_dataset(phase='val')
	# test_loss = []
	# test_acc = []
	# test_snr = np.linspace(0.2, 3, 29)

	# for i in range(29):
	# 	print("SNR = ", test_snr[i])

	# 	testsig, testlabel = deepGW.generate_batch_input_estimator(inputs, labels, 'test', snr, 1000, 0, 0)

	# 	m, r = sess.run([loss, accuracy], feed_dict={X: testsig, Y_: testlabel})

	# 	print('test:' + ' mse:' + str(m) + ' relative_error:' + str(r))

	# test_loss.append(m)
	# test_acc.append(r)

	# sio.savemat('/home/ruilan2/multi_gpu/loss.mat', {'mse': test_loss})
	# sio.savemat('/home/ruilan2/multi_gpu/acc.mat', {'ree': test_acc})	
	pass

"""
def test(inputs, labels):

	test_loss = []
	test_acc = []
	test_snr = np.linspace(0.2, 3, 29)

	for i in range(29):
		print("SNR = ", test_snr[i])

		testsig, testlabel = deepGW.generate_batch_input_estimator(inputs, labels, 'test', snr, 1000, 0, 0)

		m, r = sess.run([loss, accuracy], feed_dict={X: testsig, Y_: testlabel})

		print('test:' + ' mse:' + str(m) + ' relative_error:' + str(r))

	test_loss.append(m)
	test_acc.append(r)

	sio.savemat('/home/ruilan2/multi_gpu/ver3/loss.mat', {'mse': test_loss})
	sio.savemat('/home/ruilan2/multi_gpu/ver3/acc.mat', {'ree': test_acc})
	pass
"""


inputs, labels = deepGW.read_dataset(phase='train')
train(inputs, labels)

#inputs, labels = deepGW.read_dataset(phase='val')
#test(inputs, labels)