import numpy as np 
import random 
import h5py
import tensorflow as tf 
from scipy import interpolate
import matplotlib.pyplot as plt 
import copy
import os
import scipy.io as sio


def add_noise(signal_, label_, size, snr, X, Y_):
    signal = np.zeros([size, 8192])
    label = np.zeros([size, 2])
    
    idx = np.random.permutation(signal_.shape[0])
    signal_arr = signal_[idx, :]
    label_arr = label_[idx, :]  
    signal = signal_arr[0:size, :]
    label = label_arr[0:size, :]
    
    for i in range(size):
        r = random.randint(0, 1638)
        signal[i, 0:8191-r:1] = signal[i, r:8191:1] * snr
        signal[i] += np.random.normal(0, 1, 8192)
        signal[i] = signal[i] / np.std(signal[i])

    feed_dict={
        X:signal,
        Y_:label,
    }
    return feed_dict  

print("---------------------------------------------------")
print("LOADING DATASET")
print("---------------------------------------------------")

#... loading datasets ...
ftrain = h5py.File('/projects/ncsa/grav/cs598_final_proj/Dataset/ProperWhitenGW/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
trainsig = ftrain.get('WhitenedSignals')
trainsig = np.array(trainsig)
trainlabel = ftrain.get('m1m2')
trainlabel = np.array(trainlabel)

ftest = h5py.File('/projects/ncsa/grav/cs598_final_proj/Dataset/ProperWhitenGW/ValEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
testsig = ftest.get('WhitenedSignals')
testsig = np.array(testsig)
testlabel = ftest.get('m1m2')
testlabel = np.array(testlabel)

print("---------------------------------------------------")
print("CONSTURCTING NERUAL NETWORK")
print("---------------------------------------------------")

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))     #########???????????

def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.1))

with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, [None, 8192])
    
with tf.name_scope('Label'):
    Y_ = tf.placeholder(tf.float32, [None, 2])

with tf.name_scope('Reshape_layer'):
    XX = tf.reshape(X, [-1, 8192, 1, 1])

#... Convolution Layer 1 ... 
w_conv1 = weight_variable('w_conv1', [16, 1, 1, 64])
b_conv1 = bias_variable('b_conv1', [64])

with tf.name_scope('Convolution_layer1'):
    conv1 = tf.nn.conv2d(XX, w_conv1, strides=[1, 1, 1, 1], padding='VALID')
    h_conv1 = tf.nn.bias_add(conv1, b_conv1)
print(conv1.shape)

#... Pooling Layer 1 ... 
with tf.name_scope('Pooling_layer1'):
    h_pool1 = tf.nn.max_pool(h_conv1, [1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')
print(h_pool1.shape)

#... Activation Layer 1 ...
with tf.name_scope('Relu_layer1'):
    h_pool1 = tf.nn.relu(h_pool1)

#... Convolution Layer 2 ... 
w_conv2 = weight_variable('w_conv2', [31, 1, 64, 128])
b_conv2 = bias_variable('b_conv2', [128])

with tf.name_scope('Convolution_layer2'):
    conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='VALID')
    h_conv2 = tf.nn.bias_add(conv2, b_conv2)
print(h_conv2.shape)

#... Pooling Layer 2 ... 
with tf.name_scope('Pooling_layer2'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')
print(h_pool2.shape)

#... Activation Layer 2 ...
with tf.name_scope('Relu_layer2'):
    h_pool2 = tf.nn.relu(h_pool2)

#... Convolution Layer 3 ... 
w_conv3 = weight_variable('w_conv3', [31, 1, 128, 256])
b_conv3 = bias_variable('b_conv3', [256])

with tf.name_scope('Convolution_layer3'):
    conv3=tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='VALID')
    h_conv3 = tf.nn.bias_add(conv3, b_conv3)
print(h_conv3.shape)

#... Pooling Layer 3 ... 
with tf.name_scope('Pooling_layer3'):
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')
print(h_pool3.shape)

#... Activation Layer 3 ...
with tf.name_scope('Relu_layer3'):
    h_pool3 = tf.nn.relu(h_pool3)

#... Convolution Layer 4 ... 
w_conv4 = weight_variable('w_conv4', [63, 1, 256, 512])
b_conv4 = bias_variable('b_conv4', [512])

with tf.name_scope('Convolution_layer4'):
    conv4 = tf.nn.conv2d(h_pool3, w_conv4, strides=[1, 1, 1, 1], padding='VALID')
    h_conv4 = tf.nn.bias_add(conv4, b_conv4)
print(h_conv4.shape)

#... Pooling Layer 4 ... 
with tf.name_scope('Pooling_layer4'):
    h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')
print(h_pool4.shape)

#... Activation Layer 4 ...
with tf.name_scope('Relu_layer4'):
    h_pool4 = tf.nn.relu(h_pool4)

#... Flatten Layer ... 
with tf.name_scope('Flatten_layer'):
    h_flatten = tf.reshape(h_pool4, [-1, 7168])
print(h_flatten.shape)

#... Fully Connect Layer ... 
w_linear1 = weight_variable('w_linear1', [7168, 128])
b_linear1 = bias_variable('b_linear1', [128])

with tf.name_scope('Relu_layer5'):
    h_linear1 = tf.nn.relu(tf.matmul(h_flatten, w_linear1) + b_linear1)
print(h_linear1.shape)

#... Fully Connect Layer ... 
w_linear2 = weight_variable('w_linear2', [128, 64])
b_linear2 = bias_variable('b_linear2', [64])

with tf.name_scope('Relu_layer6'):
    h_linear2 = tf.nn.relu(tf.matmul(h_linear1, w_linear2) + b_linear2)
print(h_linear2.shape)

#... Fully Connect Layer ... 
w_linear3 = weight_variable('w_linear3', [64, 2])
b_linear3 = bias_variable('b_linear3', [2])

with tf.name_scope('Relu_layer7'):
    Y = tf.matmul(h_linear2, w_linear3) + b_linear3
print(Y.shape)

#... LOSS FUNCTION... 
mse = tf.losses.mean_squared_error(Y_, Y)

#... Relative Error...
relative_error = tf.reduce_mean(tf.cast(tf.divide(tf.abs(tf.subtract(Y, Y_)), Y_), dtype=tf.float32)) * 100

#... OPTIMIZATION ... 
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = optimizer.minimize(mse)



saver = tf.train.Saver()
tf.summary.scalar('relative_error', relative_error)
tf.summary.scalar('mse', mse)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/ruilan2/Project/save/mass_ver3/train/', sess.graph)
test_writer = tf.summary.FileWriter('/home/ruilan2/Project/save/mass_ver3/test/')


#... INITIALIZER ... 

epoch = 15000
snr = 0.6
test = add_noise(testsig, testlabel, 100, snr, X, Y_)

print("---------------------------------------------------")
print("Hyperparameter:")
print("Learning rate:", learning_rate)
print("SNR:", snr)
print("Number of epoch", epoch)
print("---------------------------------------------------")


print("---------------------------------------------------")
print("TRAINING")
print("---------------------------------------------------")

#... Training and Testing ...

error_train = []
error_test = []
mse_train = []
mse_test = []

sess.run(tf.global_variables_initializer())
for i in range(epoch):

    if (i + 1) % 100 == 0:
        summary, m, r = sess.run([merged, mse, relative_error], feed_dict=test)
        print('test:'+' mse:' + str(m) + ' relative_error:' + str(r))
        error_test.append(r)
        mse_test.append(m)
        test_writer.add_summary(summary, tf.train.global_step(sess, global_step))

    feed_dict = add_noise(trainsig, trainlabel, 50, snr, X, Y_)
    m, r = sess.run([mse, relative_error], feed_dict=feed_dict)
    _, summary = sess.run([train_step, merged], feed_dict=feed_dict)

    print('epoch:' + str(i) + ' mse:' + str(m) + ' relative_error:' + str(r))
    error_train.append(r)
    mse_train.append(m)
    train_writer.add_summary(summary, tf.train.global_step(sess, global_step))

saver.save(sess, '/home/ruilan2/Project/save/mass_ver3/mass_ver3.ckpt', global_step=epoch)
print('Model Saved')

sio.savemat('/home/ruilan2/Project/save/mass_ver3/mse_train.mat', {'mse': mse_train})
sio.savemat('/home/ruilan2/Project/save/mass_ver3/error_train.mat', {'ree': error_train})
sio.savemat('/home/ruilan2/Project/save/mass_ver3/mse_test.mat', {'mse': mse_test})
sio.savemat('/home/ruilan2/Project/save/mass_ver3/error_test.mat', {'ree': error_test})


print("---------------------------------------------------")
print("TESTING")
print("---------------------------------------------------")

test_snr = np.linspace(0.2, 3, 29)

test_mse = []
test_ree = []

for i in range(29):
    print("SNR = ", test_snr[i])

    test = add_noise(testsig, testlabel, 100, test_snr[i], X, Y_)
    m, r = sess.run([mse, relative_error], feed_dict=test)
    print('test:' + ' mse:' + str(m) + ' relative_error:' + str(r))

    test_mse.append(m)
    test_ree.append(r)

sio.savemat('/home/ruilan2/Project/save/mass_ver3/mse.mat', {'mse': test_mse})
sio.savemat('/home/ruilan2/Project/save/mass_ver3/ree.mat', {'ree': test_ree})



















