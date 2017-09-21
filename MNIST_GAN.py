from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import nntools


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist)

LOAD_OLD = False

LoGdIr = "tensorboard/large_net"
LOAD_DIR = LoGdIr


def layer_dict(layer_func, out_size, activation_function, name, use_batchnorm, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    return {
        'layer': layer_func,
        'use_batchnorm': use_batchnorm,
        'name': name,
        'kwargs': {
            'out_size': out_size,
            'name': name,
            'activation_function': activation_function,
            **kwargs
        }
    }


def generator(z, is_training):
    layers = [
        layer_dict(
            layer_func=nntools.fc_layer,
            out_size=128,
            activation_function=nntools.leaky_relu,
            name='fc1',
            use_batchnorm=True
        ),
        layer_dict(
            layer_func=nntools.fc_layer,
            out_size=28*28*16,
            activation_function=nntools.leaky_relu,
            name='fc2',
            use_batchnorm=True
        ),
        {
            'layer': nntools.reshape,
            'kwargs': {'out_size': [-1, 28, 28, 16]},
            'use_batchnorm': None,
            'name': 'Reshape'
        },
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=16,
            activation_function=nntools.leaky_relu,
            name='conv1',
            use_batchnorm=True
        ),
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=32,
            activation_function=nntools.leaky_relu,
            name='conv2',
            use_batchnorm=True
        ),
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=1,
            activation_function=tf.nn.sigmoid,
            name='conv3',
            use_batchnorm=True
        )
    ]
    out, params = nntools.create_nn(z, layers, is_training=is_training, name='Generator')
    return tf.reshape(out, [-1, 28, 28, 1]), params


def discriminator(z, is_training, reuse_vars=None):
    layers = [
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=32,
            activation_function=nntools.leaky_relu,
            name='conv1',
            use_batchnorm=True
        ),
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=32,
            activation_function=nntools.leaky_relu,
            name='conv2',
            use_batchnorm=False
        ),
        layer_dict(
            layer_func=nntools.fc_layer,
            out_size=128,
            activation_function=nntools.leaky_relu,
            name='fc1',
            use_batchnorm=False
        ),
        layer_dict(
            layer_func=nntools.fc_layer,
            out_size=1,
            activation_function=tf.nn.sigmoid,
            name='fc2',
            use_batchnorm=False
        )
    ]
    out, params = nntools.create_nn(z, layers, is_training=is_training, name='Discriminator', reuse_vars=reuse_vars)
    return out, params


def sample_Z(m, n):
    return np.random.randn(m, n)


def plot_batch(batch):
    N, img_width, img_height = batch.shape

    num_cols = int(np.sqrt(N))
    num_rows = int(N/num_cols)

    grid = np.zeros([img_height*num_rows, img_width*num_cols])

    for r in range(num_rows):
        for c in range(num_cols):
            grid[r*img_width:r*img_width + img_width, c*img_height:c*img_height + img_height] = batch[r*num_rows + c]

    plt.imshow(grid, interpolation='None', cmap='gray')
    plt.pause(1e-2)


is_training = tf.placeholder(tf.bool)


with tf.variable_scope('G'):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
    G_sample, G_vars = generator(Z, is_training=is_training)


with tf.variable_scope('D') as scope:
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X')


    D_real, D_vars = discriminator(X, reuse_vars=None, is_training=is_training)
    D_fake, D_fake_vars = discriminator(G_sample, reuse_vars=True, is_training=is_training)


D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))  # TODO: somethigg
G_loss = -tf.reduce_mean(tf.log(D_fake))

tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('D_loss', D_loss)

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=D_vars)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=G_vars)


mb_size = 64
Z_dim = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def load_model_from_file(load_dir, sess):
    step = 1
    if tf.gfile.Exists(load_dir):
        restorer = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(load_dir)
        abs_ckpt_path = os.path.abspath(ckpt.model_checkpoint_path)
        restorer.restore(sess, abs_ckpt_path)

        step_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step = %s.' % (ckpt.model_checkpoint_path, step_str))

        step = int(step_str) + 1
    else:
        print('Could not load model from %s' % load_dir)
    return step


with tf.Session(config=config) as sess:
    model_file_name = os.path.join(LoGdIr, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=2)

    train_writer = tf.summary.FileWriter('%s/train' % LoGdIr, sess.graph)
    test_writer = tf.summary.FileWriter('%s/test' % LoGdIr)

    init = tf.global_variables_initializer()

    sess.run(init)

    step = load_model_from_file(LOAD_DIR, sess) if LOAD_OLD else 1
    # TODO: PRINT SOMETHING WITH AcCuRAZY

    for it in range(step, 100001):
        if it % 5 == 0:
            print('%s' % str(it))
        X_mb, _ = mnist.train.next_batch(mb_size)
        X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

        if it % 5 == 0:
            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
            )

        summary_G, _, G_loss_curr = sess.run(
            [merged, G_solver, G_loss],
            feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
        )
        if it % 100 == 0:
            # train_writer.add_summary(summary_D, it)
            train_writer.add_summary(summary_G, it)
            train_writer.flush()
        if it % 230 == 0:
            img = sess.run(
                [G_sample],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: False}
            )
            print(img)
            img = img[0].reshape([-1, 28, 28])
            plot_batch(img)
            if it % 1000 == 0:
                plt.savefig(LoGdIr + '/images/{}.png'.format(it))

            # test_writer.add_summary(summary_D, it)

            X_mb, _ = mnist.test.next_batch(mb_size)
            X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

            _, D_loss_curr = sess.run(
                [D_solver, D_loss],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
            )
            summary_G, _, G_loss_curr = sess.run(
                [merged, G_solver, G_loss],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
            )

            test_writer.add_summary(summary_G, it)
            test_writer.flush()

            saver.save(sess, model_file_name, it)
    plt.show()
