import tensorflow as tf
import h5py 
import numpy as np 
import random
import pickle

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

"""
def generator_fix_snr(data, size, snr):
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
"""

def generator_fix_snr_estimator(signal_, label_, size, snr):
    signal = np.zeros([size, 8192])
    label = np.zeros([size, 2])

    signal_ = np.array(signal_)
    label_ = np.array(label_)
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

    return signal, label

"""
def generator_random(data, size, snr):
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
"""

def read_dataset(phase):
    print("<<<Loading datasets>>>")

    if phase == 'train':
        #ftrain = h5py.File('/projects/ncsa/grav/cs598_final_proj/Dataset/ProperWhitenGW/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
        ftrain = h5py.File('../Dataset/ProperWhitenGW/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
        sig = ftrain.get('WhitenedSignals')
        sig = list(sig)
        label = ftrain.get('m1m2')
        label = list(label)

    elif phase == 'test':
        #ftest = h5py.File('/projects/ncsa/grav/cs598_final_proj/Dataset/ProperWhitenGW/TestEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
        ftest = h5py.File('../Dataset/ProperWhitenGW/TestEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
        sig = ftest.get('WhitenedSignals')
        sig = list(sig)
        label = ftest.get('m1m2')
        label = list(label)

    else:
        #fval = h5py.File('/projects/ncsa/grav/cs598_final_proj/Dataset/ProperWhitenGW/ValEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
        fval = h5py.File('../Dataset/ProperWhitenGW/ValEOB_q-1-10-0.02_ProperWhitenZ.h5', 'r')
        sig = fval.get('WhitenedSignals')
        sig = list(sig)
        label = fval.get('m1m2')
        label = list(label)

    return sig, label

"""
def computer_batch_snr(snr, num_epoch, epoch):
    return 3 - ((3 - snr) * (epoch / num_epoch))
"""

'''
def generate_batch_input(data, phase, snr, size, num_epoch, epoch):
    """
    For testing
    """
    if phase == 'train':
        snr_ = computer_batch_snr(snr, num_epoch, epoch)

        inputs, labels = generator_fix_snr(data, size, snr_)
    else:

        inputs, labels = generator_fix_snr(data, size, snr)

    return inputs, labels 
'''

def generate_batch_input_estimator(data, label, phase, snr, size, num_epoch, epoch):
    """
    For training
    """
    if phase == 'train':
        #snr_ = computer_batch_snr(snr, num_epoch, epoch)

        #DEBUG::Use fix snr
        snr_ = snr
        inputs, labels = generator_fix_snr_estimator(data, label, size, snr_)
    else:

        inputs, labels = generator_fix_snr_estimator(data, label, size, snr)

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

"""
def inference_mass_predictor(inputs):

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
"""


def inference_mass_estimator_4conv(inputs):
    with tf.name_scope('Reshape_layer'):
        XX = tf.reshape(inputs, [-1, 8192, 1, 1])

    with tf.name_scope('Convolution_layer1'):
        w_conv1 = weight_variable('w_conv1', [16, 1, 1, 64])
        b_conv1 = bias_variable('b_conv1', [64])
        conv1 = tf.nn.conv2d(XX, w_conv1, strides=[1, 1, 1, 1], padding='VALID')
        h_conv1 = tf.nn.bias_add(conv1, b_conv1)

    with tf.name_scope('Pooling_layer1'):
        h_pool1 = tf.nn.max_pool(h_conv1, [1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')

    with tf.name_scope('Relu_layer1'):
        h_pool1 = tf.nn.relu(h_pool1)

    with tf.name_scope('Convolution_layer2'):
        w_conv2 = weight_variable('w_conv2', [31, 1, 64, 128])
        b_conv2 = bias_variable('b_conv2', [128])
        conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='VALID')
        h_conv2 = tf.nn.bias_add(conv2, b_conv2)

    with tf.name_scope('Pooling_layer2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')

    with tf.name_scope('Relu_layer2'):
        h_pool2 = tf.nn.relu(h_pool2)

    with tf.name_scope('Convolution_layer3'):
        w_conv3 = weight_variable('w_conv3', [31, 1, 128, 256])
        b_conv3 = bias_variable('b_conv3', [256])
        conv3=tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='VALID')
        h_conv3 = tf.nn.bias_add(conv3, b_conv3)

    with tf.name_scope('Pooling_layer3'):
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')

    with tf.name_scope('Relu_layer3'):
        h_pool3 = tf.nn.relu(h_pool3)

    with tf.name_scope('Convolution_layer4'):
        w_conv4 = weight_variable('w_conv4', [63, 1, 256, 512])
        b_conv4 = bias_variable('b_conv4', [512])
        conv4 = tf.nn.conv2d(h_pool3, w_conv4, strides=[1, 1, 1, 1], padding='VALID')
        h_conv4 = tf.nn.bias_add(conv4, b_conv4)


    with tf.name_scope('Pooling_layer4'):
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='VALID')

    with tf.name_scope('Relu_layer4'):
        h_pool4 = tf.nn.relu(h_pool4)

    with tf.name_scope('Flatten_layer'):
        h_flatten = tf.reshape(h_pool4, [-1, 7168])

    with tf.name_scope('Relu_layer5'):
        w_linear1 = weight_variable('w_linear1', [7168, 128])
        b_linear1 = bias_variable('b_linear1', [128])
        h_linear1 = tf.nn.relu(tf.matmul(h_flatten, w_linear1) + b_linear1)

    with tf.name_scope('Relu_layer6'):
        w_linear2 = weight_variable('w_linear2', [128, 64])
        b_linear2 = bias_variable('b_linear2', [64])
        h_linear2 = tf.nn.relu(tf.matmul(h_linear1, w_linear2) + b_linear2)

    with tf.name_scope('Relu_layer7'):
        w_linear3 = weight_variable('w_linear3', [64, 2])
        b_linear3 = bias_variable('b_linear3', [2])
        Y = tf.matmul(h_linear2, w_linear3) + b_linear3

    return  Y


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
        Y = tf.matmul(h_linear1, w_linear2) + b_linear2

    # with tf.name_scope('Output'):
    # 	Y = tf.nn.softmax(Y_logits)

    return Y



"""
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
"""


def loss_estimator(pred, labels):
    mse = tf.losses.mean_squared_error(labels, pred)

    tf.add_to_collection('losses', mse)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def accuracy_estimator(results, labels):

    relative_error = tf.cast(tf.divide(tf.abs(tf.subtract(results, labels)), labels), dtype=tf.float32)

    relative_error_mean = tf.reduce_mean(relative_error) * 100

    tf.add_to_collection('accuracies', relative_error_mean)
    return tf.add_n(tf.get_collection('accuracies'), name='total_accuracy')


