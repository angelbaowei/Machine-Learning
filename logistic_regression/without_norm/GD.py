import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


N = 80
M = 2  # M + 1 = 3    1 is bias
epsilon = 1e-8
lr = 2e-3
lr = 0.05
epoch = 3000
time = 1e-6

filename_x = '../ex4x.dat'
filename_y = '../ex4y.dat'

X = pd.read_table(filename_x, names=['score1', 'score2'], sep='\s+')


x0 = np.ones(shape=[N, 1], dtype=np.float32)
x = np.array(X, dtype=np.float32)

X_trans = np.transpose(x, [1, 0])

#print('norm...')
#mean_X = np.mean(X_trans, axis=1)
#mean_X = np.expand_dims(mean_X, axis=-1)
#std_X = np.std(X_trans, axis=1)
#std_X = np.expand_dims(std_X, axis=1)
#X_trans = (X_trans - mean_X) / std_X

X = np.transpose(X_trans, [1, 0])
X = np.concatenate([x0, X], axis=1)
print('shape X: ', X.shape, type(X))


y = pd.read_table(filename_y, names=['classes'], sep='\s+')
y = np.array(y, dtype=np.float32)
print('shape y: ', y.shape, type(y))

x_1, x_0 = np.split(X_trans, 2, axis=1)


theta = np.ndarray(shape=[M + 1, 1], dtype=np.float32)
for i in range(M + 1):
    theta[i][0] = np.random.uniform(0.0, 1.0)


def sigmoid(theta, x):
    h = - np.dot(theta.T, x)
    res = 1 + np.power(np.e, h)
    return 1.0 / (res + epsilon)


def GD(theta, lr, epoch):
    plt.figure()
    grad = np.zeros(shape=[M + 1, 1], dtype=np.float32)
    for count in range(epoch + 1):

        loss = 0
        for i in range(N):
            xi = np.expand_dims(X[i], axis=-1)
            sig = sigmoid(theta, xi)
            grad += ((y[i][0] - sig[0][0]) * xi)
            loss += ( (y[i][0] * np.log(sig[0][0] + epsilon)) + (1-y[i][0]) * np.log(1-sig[0][0] + epsilon) )

        loss /= -N
        grad /= N

        if count % 20 == 0:
            plt.ion()
            plt.subplot(232)
            plt.cla()
            plt.axis('equal')
            plt.scatter(x_1[0], x_1[1], marker='+', c='red')  # y = 1 pass exam
            plt.scatter(x_0[0], x_0[1], marker='o', c='white', edgecolors='green')  # y = 0 net pass exam
            # 分界线应为 h(X;theta) = 0.5  即 theta^T * x(i) = 0

            plt.xlabel('Exam score1')
            plt.ylabel('Exam score2')
            #plt.xlim(-2.0, 2.0)
            #plt.ylim(-2.0, 2.0)
            plt.plot(X_trans[0], (-theta[0][0] - theta[1][0] * X_trans[0]) / theta[2][0])  # X * theta = 0 是分界线
            plt.title('step = {}    set {}s / step\ntheta0 = {:.3f}, theta1 = {:.3f}, theta2 = {:.3f}'.format(count, time, theta[0][0], theta[1][0], theta[2][0]))

            plt.subplot(223)
            plt.xlim(0, epoch)
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.scatter(count, loss, color='r', label='loss', s=2)
            plt.subplot(224)
            plt.xlim(0, epoch)
            #plt.ylim(-1, 2)
            plt.xlabel('step')
            plt.ylabel('theta')
            plt.title('red : theta0, blue : theta1, green : theta2')
            plt.scatter(count, theta[0][0], color='r', label='theta0', s=2)
            plt.scatter(count, theta[1][0], color='b', label='theta1', s=2)
            plt.scatter(count, theta[2][0], color='g', label='theta2', s=2)

            plt.pause(time)

        if count % 100 == 0:

            print('loss: ', loss, 'count: ', count, 'theta: ', theta[0][0], theta[1][0], theta[2][0])
            print('************************************************************')

        #theta = theta + lr * grad
        theta[0][0] = theta[0][0] + lr * grad[0][0] #* 1e3  # bias with differenet lr
        theta[1][0] = theta[1][0] + lr * grad[1][0]
        theta[2][0] = theta[2][0] + lr * grad[2][0]

    plt.ioff()
    plt.show()


def main():
    GD(theta, lr, epoch)


if __name__ == '__main__':
    main()
