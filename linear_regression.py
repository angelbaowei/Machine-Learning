# 单特征 线性回归

import numpy as np
from matplotlib import pyplot as plt

l = []
for i in range(2000, 2014):
    l.append(i)

x = np.array(l, dtype=np.float32)
y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900], dtype=np.float32)
mean_x, std_x = np.mean(x), np.std(x)
mean_y, std_y = np.mean(y), np.std(y)
print(0.838/std_x)

old_x = x
old_y = y
x = (x - mean_x) / std_x
y = (y - mean_y) / std_y
m = 14
gt_w = 1.1
gt_b = 0

plt.scatter(x, y)
plt.plot(x, 1.1 * x)  # 假设 的 ground_truth
plt.xlabel('Year')
plt.ylabel('Price')
# plt.show()

# h(w) = w * x + b
# J(w, b) = 1/2 * (h(w) - y_)^T * (h(w) - y_)
w = np.random.uniform(0, 2)
w = np.array(w, dtype=np.float32)
b = np.random.uniform(-3, 3)
b = np.array(b, dtype=np.float32)
print(w, b)


def compute_loss(x, w, b, y_):
    loss = 0
    h = np.dot(np.transpose(w), x) + b
    # print(h)
    for i in range(m):
        loss += 1/2 * np.dot(np.transpose(h-y_), h-y_)

    loss /= m
    return loss


lr_w = lr_b = 8e-1
for i in range(500):
    loss = compute_loss(x, w, b, y)
    print('step = %d, loss = %f' % (i+1, loss))

    h = np.dot(np.transpose(w), x) + b

    grid_w = 0
    grid_b = 0
    for i in range(m):
        grid_w += ((h[i]-y[i]) * x[i])
        grid_w /= m
        grid_b += (h[i]-y[i])
        grid_b /= m
        grid_w = np.fabs(grid_w)
        grid_b = np.fabs(grid_b)

    print('grid_w = %f grid_b = %f' % (grid_w, grid_b))

    if w < gt_w:
        grid_w *= -1
    if b < gt_b:
        grid_b *= -1

    w = w - lr_w * grid_w
    b = b - lr_b * grid_b
    print('w = %f b = %f' % (w, b))



input = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
input = (input - mean_x) / std_x
output = (w * input + b)
output = (output * std_y + mean_y)
print(output)


plt.plot(x, w * x + b)
plt.show()

