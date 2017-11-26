import tensorflow as tf
import deepGW
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')


test_step_size = 50
all_num_gpus = [1]    # testing      ########################
log_device_placement = False
SNR_max = 16
SNR_min = 0.06
SNR_num = 40    # 16 / 0.2


def make_plot(loss, acc):
    """
    Test step: plot SNR vs. MSE, SNR vs. realative_error
    :param loss(list): list of MSEs
    :param acc(list): list of relative error
    :return: None
    """
    snr = np.linspace(SNR_min, SNR_max, SNR_num)
    for i in range(len(all_num_gpus)):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(snr, loss[i])
        plt.ylabel("MSE")
        ax2 = fig.add_subplot(212)
        ax2.plot(snr, acc[i])
        plt.xlabel("SNR")
        plt.ylabel("Relative Error")
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.suptitle("Test Prediction on " + str(all_num_gpus[i]) + " GPUs")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("result_img/Test_" + str(all_num_gpus[i]) + "_GPUs")


def test(inputs, labels, num_gpus):
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, [None, 1, 8192])
    with tf.name_scope('Label'):
        Y_ = tf.placeholder(tf.float32, [None, 1, 2])

    inputs_tensor = deepGW.convert_to_tensor_float(X[:, 0, :])
    labels_tensor = deepGW.convert_to_tensor_float(Y_[:, 0, :])

    pred = deepGW.inference_mass_estimator_4conv(inputs_tensor)    # same model for inference

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
    # cross_entropy = tf.reduce_mean(cross_entropy)
    # is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) * 100

    loss = deepGW.loss_estimator(pred, labels_tensor)
    acc = deepGW.accuracy_estimator(pred, labels_tensor)

    saver = tf.train.Saver()
    #model_path = "/home/nie9/Gravatitional-Wave-Prediction/mass_small/Model/Model_" + str(num_gpus) + "_GPU.ckpt"
    model_path = "/home/abc99lr/Gravatitional-Wave-Prediction/mass_small/Model/Model_" + str(num_gpus) + "_GPU.ckpt"
    saver.restore(sess, model_path)

    test_snr = np.linspace(SNR_min, SNR_max, SNR_num)
    test_loss = []
    test_acc = []

    for i in range(SNR_num):
        snr = test_snr[i]
        # print("SNR = ", snr)
        input_epoch, label_epoch = deepGW.generate_batch_input_estimator(inputs, labels, 'test', snr,
                                                                         num_gpus*test_step_size, 0, 0)
        input_batch, label_batch = deepGW.get_a_batch(input_epoch, label_epoch, test_step_size, num_gpus, 0)

        # test = add_noise(testsig, testlabel, 100, test_snr[i], X, Y_)

        cur_loss, cur_acc = sess.run([loss, acc], feed_dict={X: input_batch, Y_: label_batch})
        print('test: SNR =' + str(snr) + ' cross_entropy:' + str(cur_loss) + ' accuracy:' + str(cur_acc))

        test_loss.append(cur_loss)
        test_acc.append(cur_acc)

    return test_loss, test_acc

    #sio.savemat('/home/ruilan2/multi-gpu/wave_large/save_results/cross_entropy.mat', {'cross_entropy': test_error})
    #sio.savemat('/home/ruilan2/multi-gpu/wave_large/save_results/accuracy.mat', {'accuracy': test_acc})



if __name__ == "__main__":
    inputs, labels = deepGW.read_dataset(phase='test')
    loss, acc = [], []
    for i in range(len(all_num_gpus)):
        ret_value = test(inputs, labels, all_num_gpus[i])
        loss.append(ret_value[0])
        acc.append(ret_value[1])

    make_plot(loss, acc)


