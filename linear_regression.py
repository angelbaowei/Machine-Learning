import numpy as np
from matplotlib import pyplot as plt

N = 14  # sample number
M = 1  # feature number

l = []
for i in range(2000, 2014):
    l.append(i)

x = np.array(l, dtype=np.float32)
x = np.expand_dims(x, axis=-1)
x0 = np.ones(shape=[N, 1], dtype=np.float32)
X = np.concatenate([x0, x], axis=-1)  # shpe = N * (M+1)
print('X shape: ', X.shape)
y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900], dtype=np.float32)
y = np.expand_dims(y, axis=-1)
print('y shape:', y.shape)
mean_x, std_x = np.mean(x), np.std(x)
print('x mean, std ', mean_x, std_x)
mean_y, std_y = np.mean(y), np.std(y)
print('y mean, std ', mean_y, std_y)

# h(w) = X * sita
sita = np.zeros(shape=[M + 1, 1], dtype=np.float32)

count_list = []
loss_list = []
sita0_list = []
sita1_list = []


def GD(sita):
    epoch = 3500000
    lr = 4e-7
    for i in range(epoch + 1):
        h = np.dot(X, sita)
        loss = 0
        grad = np.zeros(shape=[M + 1, 1], dtype=np.float32)

        loss = 0.5 * np.dot((h - y).T, h - y)[0][0]
        grad = (np.dot(X.T, h - y))

        loss /= N
        grad /= N

        if i % 2000 == 0:
            count_list.append(i)
            loss_list.append(loss)
            sita0_list.append(sita[0][0])
            sita1_list.append(sita[1][0])
            print('step = %d, loss = %f, sita0 = %f, sita1 = %f' % (i, loss, sita[0][0], sita[1][0]))

        sita[0][0] = sita[0][0] - lr * grad[0][0] * 9e5
        sita[1][0] = sita[1][0] - lr * grad[1][0]

    #sita[0][0] = -1599
    #sita[1][0] = 0.8
    return sita

def formula(sita):
    #  正规方程
    XTX = np.dot(X.T, X)
    ni = np.linalg.inv(XTX)
    sita = np.dot(np.dot(ni, X.T), y)
    return sita


def result(sita):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('Year')
    plt.ylabel('Price')
    print(sita)
    plt.plot(x, sita[1][0] * x + sita[0][0])

    plt.figure()
    fg2, axes = plt.subplots(2, 2)
    fg_loss = axes[0][0]
    fg_sita0 = axes[1][0]
    fg_sita1 = axes[1][1]
    fg_loss.set_xlabel('step')
    fg_loss.set_ylabel('loss')
    fg_sita0.set_xlabel('step')
    fg_sita0.set_ylabel('sita0')
    fg_sita1.set_xlabel('step')
    fg_sita1.set_ylabel('sita1')
    fg_loss.plot(count_list, loss_list)
    fg_sita0.plot(count_list, sita0_list)
    fg_sita1.plot(count_list, sita1_list)

    plt.show()


def main():
    gd = GD(sita); result(gd)
    #form = formula(sita); result(form)


if __name__ == '__main__':
    main()




