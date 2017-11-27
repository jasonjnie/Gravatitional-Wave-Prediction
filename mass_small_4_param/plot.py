import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


test_snr = np.linspace(0.2, 3, 29) 
mat1 = sio.loadmat('./save_results/result_1123/1gpu/test_accuracy_4gpus.mat')
mat2 = sio.loadmat('./save_results/result_1123/1gpu/test_cross_entropy_4gpus.mat')

mat3 = sio.loadmat('./save_results/result_1123/2gpus/test_accuracy_4gpus.mat')
mat4 = sio.loadmat('./save_results/result_1123/2gpus/test_cross_entropy_4gpus.mat')


acc_0001 = mat1['accuracy']
err_0001 = mat2['cross_entropy']

acc_0002 = mat3['accuracy']
err_0002 = mat4['cross_entropy']


acc1 = []
acc2 = []

acc_0001 = np.array(acc_0001).T
acc_0002 = np.array(acc_0002).T

"""
for i in range(1600):
	#print(i*20)
	acc1.append(acc_0001[i*10])
for i in range(800):	
	acc2.append(acc_0002[i*10])



x1 = np.arange(1600)
x2 = np.arange(800)
"""
fig, ax = plt.subplots()
#plt.xticks(np.linspace(0, 8000, 200))
ax.plot(test_snr, np.array(acc_0001).tolist(), 'b', label="Using 1 GPU")
ax.plot(test_snr, np.array(acc_0002).tolist(), 'r', label="Using 2 GPUs")
legend = ax.legend(loc='lower right', shadow=False)
plt.xlabel('SNR')
plt.ylabel('Accuracy')
plt.title('SNR - Accuracy Plot')
plt.show()


"""
err1 = []
err2 = []

err_0001 = np.array(err_0001).T
err_0002 = np.array(err_0002).T


for i in range(1600):
	#print(i*20)
	err1.append(err_0001[i*10])
for i in range(800):	
	err2.append(err_0002[i*10])

x1 = np.arange(1600)
x2 = np.arange(800)
fig, ax = plt.subplots()
#plt.xticks(np.linspace(0, 8000, 200))
ax.plot(x1*10, np.array(err1).tolist(), 'b', label="Using 1 GPU")
ax.plot(x2*10, np.array(err2).tolist(), 'r', label="Using 2 GPUs")
legend = ax.legend(loc='upper right', shadow=False)
plt.xlabel('Step')
plt.ylabel('Cross-entropy')
plt.title('Step - Cross-entropy Plot')
plt.show()
"""












