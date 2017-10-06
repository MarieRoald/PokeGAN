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

LoGdIr = "tensorboard/large_net9999"
LOAD_DIR = LoGdIr

if not os.path.exists(os.path.join(LoGdIr, 'images')):
    os.makedirs(os.path.join(LoGdIr, 'images'))

mb_size = 64
Z_dim = 100

W1_dim = 128
W2_dim = 1024
W3_dim = 64
W4_dim = 32

weight_decay = 1


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
            out_size=W1_dim*7*7,
            activation_function=tf.nn.relu,
            name='fc1',
            use_batchnorm=True
        ),
        {
            'layer': nntools.reshape,
            'kwargs': {'out_size': [mb_size, 7, 7, W1_dim]},
            'use_batchnorm': None,
            'name': 'Reshape'
        },
        layer_dict(
            layer_func=nntools.upconv_layer,
            out_size=W2_dim,
            activation_function=tf.nn.relu,
            name='conv1',
            use_batchnorm=True,
            kwargs={
                'batch_size': mb_size,
                'k_size': 5,
                'stride': 4
            }
        ),
        layer_dict(
            layer_func=nntools.upconv_layer,
            out_size=1,
            activation_function=tf.nn.relu,
            name='conv2',
            use_batchnorm=False,
            kwargs={
                'batch_size': mb_size,
                'k_size': 5,
                'stride': 1
            }
        )
    ]
    out, params = nntools.create_nn(
        z,
        layers,
        is_training=is_training,
        name='Generator'
    )
    return tf.reshape(out, [-1, 28, 28, 1]), params


def discriminator(z, is_training, reuse_vars=None):
    layers = [
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=W2_dim,
            activation_function=nntools.leaky_relu,
            name='conv1',
            use_batchnorm=True,
            kwargs={'stride': 1,
                    'k_size': 5}
        ),
        layer_dict(
            layer_func=nntools.conv_layer,
            out_size=W1_dim,
            activation_function=nntools.leaky_relu,
            name='conv2',
            use_batchnorm=True,
            kwargs={'stride': 4,
                    'k_size': 5}
        ),
        layer_dict(
            layer_func=nntools.fc_layer,
            out_size=1,
            activation_function=tf.nn.sigmoid,
            name='fc1',
            use_batchnorm=False
        )
    ]
    out, params = nntools.create_nn(
        z,
        layers,
        is_training=is_training,
        name='Discriminator',
        reuse_vars=reuse_vars
    )
    return out, params


def sample_Z(m, n):
    return np.random.randn(m, n)


def plot_batch(batch):
    N, img_width, img_height = batch.shape

    plt.close()

    num_cols = int(5)
    num_rows = int(5)

    grid = np.zeros([img_height*num_rows, img_width*num_cols])

    for r in range(num_rows):
        for c in range(num_cols):
            grid[
                r*img_width:r*img_width + img_width,
                c*img_height:c*img_height + img_height
            ] = batch[r*num_rows + c]

    plt.imshow(grid, interpolation='None', cmap='gray')
    plt.pause(1e-2)


is_training = tf.placeholder(tf.bool)
print('Creating generator network')
with tf.variable_scope('G'):
    Z = tf.placeholder(tf.float32, shape=[mb_size, Z_dim], name='Z')
    G_sample, G_vars = generator(Z, is_training=is_training)
    G_vars_sq = [tf.nn.l2_loss(var)/np.prod(var.get_shape().as_list()) for var in G_vars.values()]
    G_reg = weight_decay*tf.add_n(G_vars_sq)


print('Creating discriminator network')
with tf.variable_scope('D') as scope:
    X = tf.placeholder(tf.float32, shape=[mb_size, 28, 28, 1], name='X')

    D_real, D_vars = discriminator(X, reuse_vars=None, is_training=is_training)
    D_fake, D_fake_vars = discriminator(G_sample, reuse_vars=True, is_training=is_training)

    D_vars_sq = [tf.nn.l2_loss(var)/np.prod(var.get_shape().as_list()) for var in D_vars.values()]
    D_reg = weight_decay*tf.add_n(D_vars_sq)

print('Creating losses:')
G_loss = -tf.reduce_mean(tf.log(D_fake))
D_loss = -tf.reduce_mean(0.9*tf.log(D_real) + 0.1*tf.log(D_real) + tf.log(1. - D_fake))  # TODO: somethigg

# Create summaries:
fake_acc = tf.reduce_mean(tf.cast(tf.less(D_fake, 0.5), tf.float32))
real_acc = tf.reduce_mean(tf.cast(tf.less(-D_real, -0.5), tf.float32))

tf.summary.scalar('Fake acc', fake_acc)
tf.summary.scalar('Real acc', real_acc)

tf.summary.scalar('D_real', tf.reduce_mean(D_real))
tf.summary.scalar('D_fake', tf.reduce_mean(D_fake))

tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('D_loss', D_loss)

merged = tf.summary.merge_all()

# Define train step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(
        D_loss + D_reg,
        var_list=D_vars
    )
    G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(
        G_loss + G_reg,
        var_list=G_vars
    )


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

        _, _, D_loss_curr, G_loss_curr, summary_G = sess.run(
            [D_solver, G_solver, D_loss, G_loss, merged],
            feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
        )
        if it % 100 == 0:
            # train_writer.add_summary(summary_D, it)
            train_writer.add_summary(summary_G, it)
            train_writer.flush()
        if it % 250 == 0:
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

            _, D_loss_curr, D_fake_curr, D_real_curr = sess.run(
                [D_solver, D_loss, D_fake, D_real],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
            )
            print("d_fake", D_fake_curr)
            print("d_real", D_real_curr)
            summary_G, _, G_loss_curr = sess.run(
                [merged, G_solver, G_loss],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim), is_training: True}
            )

            test_writer.add_summary(summary_G, it)
            test_writer.flush()

            if it % 1000 == 0:
                saver.save(sess, model_file_name, it)
    plt.show()
