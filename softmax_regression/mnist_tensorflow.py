#coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print('start read dataset...')
mnist = input_data.read_data_sets("../datasets/", one_hot=True)

N = mnist.train.num_examples  # train number
M = 28 * 28  # feature dimension   784
C = 10  # theta = M * C
lr = 1e-6
epoch = 10000
#batch = 256


X = tf.placeholder(shape=(None, M), dtype=tf.float32)
Y = tf.placeholder(shape=(None, C), dtype=tf.float32)

w = tf.get_variable(
    'w',
    shape=[M, C],
    dtype=tf.float32,
    initializer=tf.glorot_uniform_initializer()
)

b = tf.get_variable(
    'b',
    shape=[1, C],
    dtype=tf.float32,
    initializer=tf.zeros_initializer()
)

Y_pred = tf.nn.softmax(tf.matmul(X, w) + b)

loss = -tf.reduce_sum(Y * tf.log(Y_pred))

tloss = tf.summary.scalar('train_loss', loss)
vloss = tf.summary.scalar('val_loss', loss)

global_steps = tf.train.get_or_create_global_step()

decay_lr = tf.train.polynomial_decay(
    learning_rate=lr,
    global_step=global_steps,
    decay_steps=epoch,
    end_learning_rate=1e-6,
    power=0.9
)

slr = tf.summary.scalar('lr', decay_lr)

train_step = tf.train.GradientDescentOptimizer(decay_lr).minimize(loss, global_step=global_steps)

with tf.control_dependencies([train_step]):
    train_op = tf.no_op(name='train')

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

vacc = tf.summary.scalar('val_acc', acc)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./', graph=sess.graph)

    for count in range(epoch + 1):
        #train_images, train_labels = mnist.train.next_batch(batch, shuffle=True)
        train_images, train_labels = mnist.train.images, mnist.train.labels
        _, train_loss, step, summary = sess.run([train_op, loss, global_steps, merged], feed_dict={X: train_images, Y: train_labels})
        if count % 10 == 0:
            summary_writer.add_summary(summary, global_step=step)
        if count % 100 == 0:
            print('train_loss: {:.3f}, step: {}'.format(train_loss, step - 1))
            val_images, val_labels = mnist.validation.images, mnist.validation.labels
            val_acc = sess.run(acc, feed_dict={X: val_images, Y: val_labels})
            print('val_acc: {:.3f}, step: {}'.format(val_acc, step - 1))

