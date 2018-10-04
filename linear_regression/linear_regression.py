import numpy as np
from matplotlib import pyplot as plt

N = 14
M = 1

l = []
for i in range(2000, 2014):
    l.append(i)

x = np.array(l, dtype=np.float32)
x = np.expand_dims(x, axis=-1)
x0 = np.ones(shape=[N, 1], dtype=np.float32)

mean_x, std_x = np.mean(x), np.std(x)
print('x mean, std ', mean_x, std_x)

print('norm')
x = (x - mean_x) / std_x

X = np.concatenate([x0, x], axis=-1)  # shpe = N * (M+1)
print('X shape: ', X.shape)
y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900], dtype=np.float32)
y = np.expand_dims(y, axis=-1)
print('y shape:', y.shape)
mean_y, std_y = np.mean(y), np.std(y)
print('y mean, std ', mean_y, std_y)

y = (y - mean_y) / std_y

# h(w) = X * theta
theta = np.zeros(shape=[M + 1, 1], dtype=np.float32)

count_list = []
loss_list = []
theta0_list = []
theta1_list = []


def GD(theta):
    epoch = 1000
    lr = 0.5
    for i in range(epoch + 1):
        h = np.dot(X, theta)
        loss = 0
        grad = np.zeros(shape=[M + 1, 1], dtype=np.float32)

        loss = 0.5 * np.dot((h - y).T, h - y)[0][0]
        grad = (np.dot(X.T, h - y))

        loss /= N
        grad /= N

        if i % 100 == 0:
            count_list.append(i)
            loss_list.append(loss)
            theta0_list.append(theta[0][0])
            theta1_list.append(theta[1][0])
            print('step = %d, loss = %f, theta0 = %f, theta1 = %f' % (i, loss, theta[0][0], theta[1][0]))

        theta = theta - lr * grad

    #theta[0][0] = -1599
    #theta[1][0] = 0.8
    return theta

def formula(theta):
    #  正规方程
    XTX = np.dot(X.T, X)
    ni = np.linalg.inv(XTX)
    theta = np.dot(np.dot(ni, X.T), y)
    return theta


def result(theta):

    plt.scatter(x, y)
    plt.xlabel('Year')
    plt.ylabel('Price')
    print(theta)
    plt.plot(x, theta[1][0] * x + theta[0][0])

    fg2, axes = plt.subplots(2, 2)
    fg_loss = axes[0][0]
    fg_theta0 = axes[1][0]
    fg_theta1 = axes[1][1]
    fg_loss.set_xlabel('step')
    fg_loss.set_ylabel('loss')
    fg_theta0.set_xlabel('step')
    fg_theta0.set_ylabel('theta0')
    fg_theta1.set_xlabel('step')
    fg_theta1.set_ylabel('theta1')
    fg_loss.plot(count_list, loss_list)
    fg_theta0.plot(count_list, theta0_list)
    fg_theta1.plot(count_list, theta1_list)

    plt.show()


def main():
    gd = GD(theta); result(gd)
    #form = formula(theta); result(form)


if __name__ == '__main__':
    main()
