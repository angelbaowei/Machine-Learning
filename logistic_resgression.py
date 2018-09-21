import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


N = 80
M = 2  # M + 1 = 3


filename_x = './ex4x.dat'
filename_y = './ex4y.dat'

X = pd.read_table(filename_x, names=['score1', 'score2'], sep='\s+')


x0 = np.ones(shape=[N, 1], dtype=np.float32)
x = np.array(X, dtype=np.float32)

X_trans = np.transpose(x, [1, 0])

X = np.concatenate([x0, x], axis=1)
print('shape X: ', X.shape, type(X))

y = pd.read_table(filename_y, names=['classes'], sep='\s+')
y = np.array(y, dtype=np.float32)
print('shape y: ', y.shape, type(y))

x_1, x_0 = np.split(X_trans, 2, axis=1)


sita = np.zeros(shape=[M + 1, 1], dtype=np.float32)

count_list = []
loss_list = []
sita0_list = []
sita1_list = []
sita2_list = []


def sigmoid(sita, x):
    h = - np.dot(sita.T, x)
    res = 1 + np.power(np.e, h)
    if res == 0:
        res += 1e-8
    return 1.0 / res


def GD(sita):
    epoch = 3000
    grad = np.zeros(shape=[M + 1, 1], dtype=np.float32)
    lr = 2e-3
    for count in range(epoch + 1):

        loss = 0
        for i in range(N):
            xi = np.expand_dims(X[i], axis=-1)
            sig = sigmoid(sita, xi)
            grad += ((y[i][0] - sig[0][0]) * xi)
            loss += ( (y[i][0] * np.log(sig[0][0] + 1e-8)) + (1-y[i][0]) * np.log(1-sig[0][0] + 1e-8) )

        loss /= -N
        grad /= N

        if count % 10 == 0:
            count_list.append(count)
            loss_list.append(loss)
            sita0_list.append(sita[0][0])
            sita1_list.append(sita[1][0])
            sita2_list.append(sita[2][0])

        if count % 500 == 0:

            print('loss: ', loss, 'count: ', count, 'sita: ', sita[0][0], sita[1][0], sita[2][0])
            print('************************************************************')

        # bias
        sita[0][0] = sita[0][0] + lr * grad[0][0] * 1e3
        # w
        sita[1][0] = sita[1][0] + lr * grad[1][0]
        sita[2][0] = sita[2][0] + lr * grad[2][0]

    return sita


def SGD(sita):
    epoch = 300000
    grad = np.zeros(shape=[M + 1, 1], dtype=np.float32)
    lr = 4e-4
    for count in range(epoch + 1):
        i = np.random.random_integers(0, N - 1)  # random
        xi = np.expand_dims(X[i], axis=-1)
        sig = sigmoid(sita, xi)
        grad = ((y[i][0] - sig[0][0]) * xi)
        #print(grad[0][0])
        loss = - ((y[i][0] * np.log(sig[0][0] + 1e-8)) + (1 - y[i][0]) * np.log(1 - sig[0][0] + 1e-8))

        if count % 500 == 0:
            count_list.append(count)
            loss_list.append(loss)
            sita0_list.append(sita[0][0])
            sita1_list.append(sita[1][0])
            sita2_list.append(sita[2][0])
        if count % 1000 == 0:
            print('loss: ', loss, 'count: ', count, 'sita: ', sita[0][0], sita[1][0], sita[2][0])
            print('************************************************************')

        sita[0][0] = sita[0][0] + lr * grad[0][0] * 4e1
        sita[1][0] = sita[1][0] + lr * grad[1][0]
        sita[2][0] = sita[2][0] + lr * grad[2][0]

    return sita


def Newton(sita):
    epoch = 80
    sita = np.zeros(shape=[M + 1, 1], dtype=np.float32)  # sita init = 0
    grad = np.zeros(shape=[M + 1, 1], dtype=np.float32)
    H = np.zeros(shape=[M + 1, M + 1], dtype=np.float32)  # Hessian  matrix
    loss = 0
    for count in range(epoch + 1):
        for i in range(N):
            xi = np.expand_dims(X[i], axis=-1)
            sig = sigmoid(sita, xi)
            loss += ((y[i][0] * np.log(sig[0][0] + 1e-8)) + (1 - y[i][0]) * np.log(1 - sig[0][0] + 1e-8))

            grad += ( (sig[0][0] - y[i][0]) * xi )

            H += sig * sig * np.dot(xi, xi.T)

        grad /= N   #  d j(sita) / d sita
        loss /= -N
        H /= N

        count_list.append(count)
        loss_list.append(loss)
        sita0_list.append(sita[0][0])
        sita1_list.append(sita[1][0])
        sita2_list.append(sita[2][0])

        if count % 1 == 0:
            print('loss: ', loss, 'count: ', count, 'sita: ', sita[0][0], sita[1][0], sita[2][0])
            print('************************************************************')

        sita = sita - np.dot(np.linalg.inv(H), grad)



    return sita


def result(sita):
    sita0 = sita[0][0]
    sita1 = sita[1][0]
    sita2 = sita[2][0]
    print('sita0: ', sita0, 'sita1: ', sita1, 'sita2: ', sita2)
    # sita0 = -16.38
    # sita1 = 0.1483
    # sita2 = 0.1589
    plt.figure()
    fg2, axes = plt.subplots(2, 2)
    fg_loss = axes[0][0]
    fg_sita0 = axes[0][1]
    fg_sita1 = axes[1][0]
    fg_sita2 = axes[1][1]
    fg_loss.set_xlabel('step')
    fg_loss.set_ylabel('loss')
    fg_sita0.set_xlabel('step')
    fg_sita0.set_ylabel('sita0')
    fg_sita1.set_xlabel('step')
    fg_sita1.set_ylabel('sita1')
    fg_sita2.set_xlabel('step')
    fg_sita2.set_ylabel('sita2')
    fg_loss.plot(count_list, loss_list)
    fg_sita0.plot(count_list, sita0_list)
    fg_sita1.plot(count_list, sita1_list)
    fg_sita2.plot(count_list, sita2_list)

    plt.figure()
    plt.scatter(x_1[0], x_1[1], marker='+', c='red')  # y = 1 pass exam
    plt.scatter(x_0[0], x_0[1], marker='o', c='white', edgecolors='green')  # y = 0 net pass exam
    # 分界线应为 h(X;sita) = 0.5  即 sita^T * x(i) = 0

    plt.xlabel('Exam score1')
    plt.ylabel('Exam score2')
    plt.plot(X_trans[0], (-sita0 - sita1 * X_trans[0]) / sita2)  # X * sita = 0 是分界线
    plt.show()


def main():
    gd = GD(sita); result(gd)
    #sgd = SGD(sita); result(sgd)
    #newton = Newton(sita); result(newton)


if __name__ == '__main__':
    main()
