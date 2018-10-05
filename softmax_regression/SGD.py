#coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random

print('start read dataset...')
mnist = input_data.read_data_sets("./")
train_images = mnist.train.images
train_labels = mnist.train.labels
val_images = mnist.validation.images
val_labels = mnist.validation.labels
print("train_dataset:  {},{}".format(train_images.shape, train_labels.shape))
print("val_dataset:  {},{}".format(val_images.shape, val_labels.shape))
print('read finished!')

N = train_labels.shape[0]  # train number
M = 28 * 28 + 1  # feature dimension   M + 1 = 785    + 1 is bias
C = 10  # theta = C * M
epsilon = 1e-8
lr = 0.01
epoch = 3000
batch = 16

time = 0.05  # show time span


X = train_images  # (?, 784)

#X /= 255.0

x0 = np.ones(shape=(N, 1), dtype=np.float32)
X = np.concatenate([x0, X], axis=-1)  # input_x (?, 785)
print('train_x: ', X.shape)
Y = train_labels
Y = np.expand_dims(Y, axis=-1)
print('train_y: ', Y.shape)


X_val = val_images  # (?, 784)

#X_val /= 255.0

N_val = val_labels.shape[0]  # val number
x0_val = np.ones(shape=(N_val, 1), dtype=np.float32)
X_val = np.concatenate([x0_val, X_val], axis=-1)
print('val_x: ', X_val.shape)
Y_val = val_labels
Y_val = np.expand_dims(Y_val, axis=-1)
print('val_y: ', Y_val.shape)


theta = np.zeros(shape=[C, M], dtype=np.float32)


def softmax(xi, thetaj):
    h = np.dot(thetaj.T, xi)
    eh = np.exp(h)
    sum = 0
    for i in range(C):
        theta_sum = np.expand_dims(theta[i], axis=-1)
        sum += np.exp(np.dot(theta_sum.T, xi))

    return float(eh) / sum


def SGD(theta, lr, epoch):
    #plt.figure(figsize=(1920, 1080))
    plt.figure()
    grad = np.zeros(shape=[C, M], dtype=np.float32)

    for count in range(epoch + 1):

        loss = 0
        rdm = random.sample(range(0, N), batch)

        #print(rdm)

        for i in range(batch):
            i = rdm[i]
            xi = np.expand_dims(X[i], axis=-1)
            thetaj = np.expand_dims(theta[Y[i][0]], axis=-1)

            h = softmax(xi, thetaj)

            loss += (np.log(h + epsilon))

            for k in range(C):
                thetak = np.expand_dims(theta[k], axis=-1)
                hk = softmax(xi, thetak)
                if Y[i][0] != k:
                    tmp = np.squeeze(- hk[0][0] * xi)
                    grad[k] += tmp
                else:
                    tmp = np.squeeze((1 - hk[0][0]) * xi)
                    grad[k] += tmp


        #loss /= -N
        loss /= -batch
        grad /= batch


        if count % 100 == 0:

            # compute validation dataset accuracy

            correct = 0
            for number in range(N_val):
                xi = np.expand_dims(X_val[number], axis=-1)
                # thetaj = np.expand_dims(theta[Y[number][0]], axis=-1)
                list_h = []
                for i in range(C):
                    list_h.append(softmax(xi, np.expand_dims(theta[i], axis=-1)))

                classes = np.argmax(list_h)

                #if number % 500 == 0:
                #    print('{}, {}, {}'.format(number, classes, Y_val[number][0]))

                if classes == Y_val[number][0]:
                    correct += 1

            acc = float(correct) / N_val

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

            print('loss: ', loss[0][0], 'count: ', count, end=' ')
            print('val_acc = {:.3f}'.format(acc))
            print('************************************************')

        # update
        theta = theta + lr * grad

    plt.ioff()
    plt.show()


def main():
    SGD(theta, lr, epoch)


if __name__ == '__main__':
    main()
