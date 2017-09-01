from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist)

LOAD_OLD = False

LoGdIr = "noe_bullshit"
LOAD_DIR = LoGdIr


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('Summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('Mean', mean)
        with tf.name_scope('Stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('Stddev', stddev)
        tf.summary.scalar('Max', tf.reduce_max(var))
        tf.summary.scalar('Min', tf.reduce_min(var))
        tf.summary.histogram('Histogram', var)


def fc_layer(x, out_size, activation_function=lambda x: x, name=None):
    with tf.name_scope(name):
        params = {'W': None, 'b': None}
        shape = x.get_shape().as_list()

        input_is_vector = len(shape) == 2
        last_dim = shape[-1] if input_is_vector else np.prod(shape[1:])

        params['W'] = tf.get_variable(
            "W",
            [last_dim, out_size],
            initializer=tf.glorot_normal_initializer()
        )

        params['b'] = tf.get_variable(
            "b",
            [out_size],
            initializer=tf.zeros_initializer()
        )

        x = x if input_is_vector else tf.reshape(x, (-1, last_dim))
        out = tf.matmul(x, params['W']) + params['b']

        return activation_function(out), params


def conv_layer(x, out_size, activation_function=lambda x: x, name=None, k_size=3, stride=1):
    with tf.name_scope(name):
        params = {'W': None, 'b': None}
        shape = x.get_shape().as_list()

        params['W'] = tf.get_variable(
            'W',
            [k_size, k_size, shape[-1], out_size],
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )

        params['b'] = tf.get_variable(
            'b',
            initializer=tf.zeros([shape[-1], out_size])
        )

        out = params['b'] + tf.nn.conv2d(
            input=x,
            filter=params['W'],
            strides=(1, stride, stride, 1),
            padding='SAME'
        )

        variable_summaries(params['W'])
        # variable_summaries(params['b'])

        return activation_function(out), params


def create_nn(in_var, layers, is_training, name=None):
    """A function that creates a custom neural network.

    Parameters:
    -----------
    in_var : tf.Variable
        The input variable of the network,
    layers : list
        A list of dictionaries describing the shape of each layer.
        The keys of these dictionaries should be:
            `layer`: A function that creates nodes for each layer. This
            function should have the keyword arguments
                     `x` - the input variable to this layer.
                     `out_size` - the no. of channels out of this layer.
                     `activation_function` - the activation function to use.
                     `name` - The name of this layer.
            `out_size`: The no. of channels out of the corresponding layer.
            `activation_function`: The activation function the corresponding
            layer should use, e.g. tf.nn.relu.
            `name`: The name- and variable scope of the corresponding layer.
            `use_batchnorm`: Boolean, whether or not to use batchnorm after
            this layer.
            `kwargs`: Other keyword arguments for the layer function as a
            dictionary (optional).
    is_training : tf.Placeholder
        This should be true when the network is training and false otherwise.
    name : str
        The name scope of the network

    Returns:
    --------
    out : tf.Variable
        The output variable of the network.
    params : The weights used in this network
    """
    out = in_var
    params = {}
    with tf.name_scope(name):
        for layer in layers:
            layer['kwargs'] = {} if 'kwargs' not in layer else layer['kwargs']
            assert 'out_size' in layer
            assert 'activation_function' in layer
            assert 'name' in layer
            assert 'use_batchnorm' in layer

            with tf.variable_scope(layer['name']):
                out, curr_params = layer['layer'](
                    x=out,
                    out_size=layer['out_size'],
                    activation_function=layer['activation_function'],
                    name=layer['name'],
                    **layer['kwargs']
                )

                if layer['use_batchnorm']:
                    out = tf.layers.batch_normalization(out, training=is_training)

            for key, val in curr_params.items():
                params['{}/{}'.format(layer['name'], key)] = val

    return out, params


def layer_dict(layer_func, out_size, activation_function, name, use_batchnorm, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    return {
        'layer': layer_func,
        'out_size': out_size,
        'activation_function': activation_function,
        'name': name,
        'use_batchnorm': use_batchnorm,
        'kwargs': kwargs
    }


def generator(z):
    layers = [
        layer_dict(
            layer_func=fc_layer,
            out_size=128,
            activation_function=tf.nn.relu,
            name='fc1',
            use_batchnorm=False
        ),
        layer_dict(
            layer_func=fc_layer,
            out_size=28*28,
            activation_function=tf.nn.sigmoid,
            name='fc2',
            use_batchnorm=False
        )
    ]
    out, params = create_nn(z, layers, is_training=True, name='Generator')
    print(tf.reshape(out, [-1, 28, 28, 1]), params)
    return tf.reshape(out, [-1, 28, 28, 1]), params
    """
    with tf.variable_scope("fc1"):
        gen_fc, params_fc = fc_layer(z, 128, tf.nn.relu)

    with tf.variable_scope("fc2"):
        gen_prob, params_prob = fc_layer(gen_fc, 28*28, tf.nn.sigmoid)

    gen_vars = []
    for key_fc, key_prob in zip(params_fc, params_prob):
        gen_vars += [params_fc[key_fc], params_prob[key_prob]]

    return tf.reshape(gen_prob, [-1, 28, 28, 1]), gen_vars
    """


def discriminator(z):
    with tf.variable_scope("conv1"):
        disc_conv1, params_conv1 = conv_layer(z, 32, activation_function=tf.nn.relu)

    with tf.variable_scope("fc1"):
        disc_fc, params_fc = fc_layer(disc_conv1, 128, tf.nn.relu)

    with tf.variable_scope("fc2"):
        disc_prob, params_prob = fc_layer(disc_fc, 1, activation_function=tf.nn.sigmoid)

    disc_vars = []
    for key_fc, key_prob in zip(params_fc, params_prob):
        disc_vars += [params_fc[key_fc], params_prob[key_prob]]

    return disc_prob, disc_vars


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


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


with tf.variable_scope('G'):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
    G_sample, G_vars = generator(Z)


with tf.variable_scope('D') as scope:
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X')

    D_real, D_vars = discriminator(X)
    scope.reuse_variables()
    D_fake, _ = discriminator(G_sample)


D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))  # TODO: somethigg
G_loss = -tf.reduce_mean(tf.log(D_fake))

tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('D_loss', D_loss)

merged = tf.summary.merge_all()


D_solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(D_loss, var_list=D_vars)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars)


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

    for it in range(step, 10001):
        print('%s' % str(it))
        X_mb, _ = mnist.train.next_batch(mb_size)
        X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)}
        )

        summary_G, _, G_loss_curr = sess.run(
            [merged, G_solver, G_loss],
            feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)}
        )

        # train_writer.add_summary(summary_D, it)
        train_writer.add_summary(summary_G, it)
        train_writer.flush()
        if it % 500 == 0:
            img = sess.run(
                [G_sample],
                feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)}
            )
            print(img)
            img = img[0].reshape([-1, 28, 28])
            plot_batch(img)

            # test_writer.add_summary(summary_D, it)

            X_mb, _ = mnist.test.next_batch(mb_size)
            X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
            summary_G, _, G_loss_curr = sess.run([merged, G_solver, G_loss],
                                                 feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})

            test_writer.add_summary(summary_G, it)
            test_writer.flush()

            saver.save(sess, model_file_name, it)
    plt.show()
