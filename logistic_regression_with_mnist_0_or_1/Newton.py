#coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print('start read dataset...')
mnist = input_data.read_data_sets("../datasets/")
train_images = [mnist.train.images[mnist.train.labels == 0], mnist.train.images[mnist.train.labels == 1]]
train_labels = [mnist.train.labels[mnist.train.labels == 0], mnist.train.labels[mnist.train.labels == 1]]
val_images = [mnist.validation.images[mnist.validation.labels == 0], mnist.validation.images[mnist.validation.labels == 1]]
val_labels = [mnist.validation.labels[mnist.validation.labels == 0], mnist.validation.labels[mnist.validation.labels == 1]]
print("train_dataset: '0': {},{}  '1': {},{}".format(train_images[0].shape, train_labels[0].shape, train_images[1].shape, train_labels[1].shape))
print("val_dataset: '0': {},{}  '1': {},{}".format(val_images[0].shape, val_labels[0].shape, val_images[1].shape, val_labels[1].shape))
print('read finished!')

N = train_labels[0].shape[0] + train_labels[1].shape[0]  # train number
M = 28 * 28 + 1  # feature dimension   M + 1 = 785    + 1 is bias
epsilon = 1e-8
lr = 0.01
epoch = 1000

time = 0.05  # show time span


X = np.concatenate([train_images[0], train_images[1]], axis=0)  # (?, 784)
print('norm...')
x0 = np.ones(shape=(N, 1), dtype=np.float32)
X = np.concatenate([x0, X], axis=-1)  # input_x (?, 785)
print('train_x: ', X.shape)
Y = np.concatenate([train_labels[0], train_labels[1]], axis=0)
Y = np.expand_dims(Y, axis=-1)
print('train_y: ', Y.shape)


X_val = np.concatenate([val_images[0], val_images[1]], axis=0)  # (?, 784)
N_val = N = val_labels[0].shape[0] + val_labels[1].shape[0]  # val number
x0_val = np.ones(shape=(N_val, 1), dtype=np.float32)
X_val = np.concatenate([x0_val, X_val], axis=-1)
print('val_x: ', X_val.shape)
Y_val = np.concatenate([val_labels[0], val_labels[1]], axis=0)
Y_val = np.expand_dims(Y_val, axis=-1)
print('val_y: ', Y_val.shape)


theta = np.ndarray(shape=[M, 1], dtype=np.float32)
for i in range(M):
    theta[i][0] = np.random.uniform(0.0, 1.0)


def sigmoid(theta, x):
    h = - np.dot(theta.T, x)
    res = 1 + np.power(np.e, h)
    return 1.0 / (res + epsilon)


def GD(theta, lr, epoch):
    plt.figure()
    theta = np.zeros(shape=[M, 1], dtype=np.float32)  # theta init = 0
    grad = np.zeros(shape=[M, 1], dtype=np.float32)
    H = np.zeros(shape=[M, M], dtype=np.float32)  # Hessian  matrix
    loss = 0
    for count in range(epoch + 1):
        for i in range(N):
            xi = np.expand_dims(X[i], axis=-1)
            sig = sigmoid(theta, xi)
            loss += ((Y[i][0] * np.log(sig[0][0] + 1e-8)) + (1 - Y[i][0]) * np.log(1 - sig[0][0] + 1e-8))

            grad += ((sig[0][0] - Y[i][0]) * xi)

            H += sig * sig * np.dot(xi, xi.T)

        grad /= N  # d j(theta) / d theta
        loss /= -N
        H /= N

        # compute validation dataset accuracy
        correct = 0
        temp00, temp01, temp10, temp11 = 0, 0, 0, 0
        for number in range(N_val):
            xi = np.expand_dims(X_val[number], axis=-1)
            h = np.dot(theta.T, xi)
            if h[0][0] > 0:
                pred = 1
                if Y_val[number][0] == pred:
                    correct += 1
                    temp11 += 1
                else:
                    temp01 += 1

            elif h[0][0] < 0:
                pred = 0
                if Y_val[number][0] == pred:
                    correct += 1
                    temp00 += 1
                else:
                    temp10 += 1

        acc = float(correct)/ N_val

        if count % 1 == 0:
            plt.ion()
            plt.subplot(121)
            plt.title('train_loss')
            plt.xlim(0, epoch)
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.scatter(count, loss, color='r', label='loss', s=2)

            plt.subplot(122)
            plt.title('validation_accuracy')
            plt.xlim(0, epoch)
            plt.ylim(0, 1)
            plt.xlabel('step')
            plt.ylabel('accuracy')
            plt.scatter(count, acc, color='r', label='accuracy', s=2)

            plt.pause(time)

        if count % 1 == 0:
            print('loss: ', loss, 'count: ', count, end=' ')
            print('acc = {:.3f}'.format(acc))
            print('************************************************')
            print('----------------------')
            print(' label   predict num')
            print('   0      0      %d' % temp00)
            print('   0      1      %d' % temp01)
            print('   1      0      %d' % temp10)
            print('   1      1      %d' % temp11)
            print('----------------------')

        # update
        theta = theta - np.dot(np.linalg.pinv(H), grad)  # 广义逆矩阵

    plt.ioff()
    plt.show()


def main():
    GD(theta, lr, epoch)


if __name__ == '__main__':
    main()
