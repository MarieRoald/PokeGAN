from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
print(mnist)



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def fc_layer(x, out_size, activation_function=lambda x: x):
    params = {'W': None, 'b': None}
    shape = x.get_shape().as_list()

    input_is_vector = len(shape) == 2
    last_dim = shape[-1] if input_is_vector else np.prod(shape[1:])

    params['W'] = tf.get_variable(
        "W",
        [last_dim, out_size],
        initializer=tf.random_normal_initializer(stddev=tf.cast(tf.sqrt(1./(out_size*last_dim)), tf.float32))
    )

    initializer_b = tf.constant_initializer(0.0)
    params['b'] = tf.get_variable("b", [out_size], initializer=initializer_b)

    out = tf.matmul(x, params['W']) if input_is_vector else tf.matmul(tf.reshape(x, (-1, last_dim)), params['W']) + params['b']

    return activation_function(out), params


def conv_layer(x, out_size, k_size=3, stride=1, activation_function=lambda x: x):
    params = {'W': None, 'b': None}
    shape = x.get_shape().as_list()

    params['W'] = tf.get_variable(
        'W',
        [k_size, k_size, shape[-1], out_size],
        initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
    )

    params['b'] = tf.get_variable(
        'b',
        initializer=tf.zeros([shape[-1], out_size])
    )

    out = tf.nn.conv2d(x,params['W'], (1, stride,stride, 1), padding='SAME') + params['b']

    variable_summaries(params['W'])
    #variable_summaries(params['b'])

    return activation_function(out), params



def generator(z):
    with tf.variable_scope("fc1"):
        gen_fc, params_fc = fc_layer(z, 128, tf.nn.relu)

    with tf.variable_scope("fc2"):
        gen_prob, params_prob = fc_layer(gen_fc, 28*28, tf.nn.sigmoid)

    gen_vars = []
    for key_fc, key_prob in zip(params_fc, params_prob):
        gen_vars += [params_fc[key_fc], params_prob[key_prob]]

    return tf.reshape(gen_prob, [-1, 28, 28, 1]), gen_vars


def discriminator(z):
    with tf.variable_scope("conv1"):
        disc_conv1, params_conv1 = conv_layer(z, 32, activation_function=tf.nn.relu)

    with tf.variable_scope("fc1"):
        disc_fc, params_fc = fc_layer(disc_conv1, 128, tf.nn.relu)

    with tf.variable_scope("fc2"):
        disc_logit, params_prob = fc_layer(disc_fc, 1)
        disc_prob = tf.nn.sigmoid(disc_logit)

    disc_vars = []
    for key_fc, key_prob in zip(params_fc, params_prob):
        disc_vars += [params_fc[key_fc], params_prob[key_prob]]

    return disc_prob, disc_logit, disc_vars


def sample_Z(m,n):
    return np.random.uniform(-1., 1., size=[m,n])


def plot_batch(batch):
    N,img_width,img_height = batch.shape

    num_cols = int(np.sqrt(N))
    num_rows = N/num_cols

    grid = np.zeros([img_height*num_rows,img_width*num_cols])

    for r in range(num_rows):
        for c in range(num_cols):
            grid[r*img_width:r*img_width + img_width,c*img_height:c*img_height + img_height] = batch[r*num_rows + c]

    plt.imshow(grid,interpolation='None', cmap='gray')
    plt.pause(1e-2)


with tf.variable_scope('G'):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
    G_sample, G_vars = generator(Z)


with tf.variable_scope('D') as scope:
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X')

    D_real, D_logit_real, D_vars = discriminator(X)
    scope.reuse_variables()
    D_fake, D_logit_fake, _  = discriminator(G_sample)


D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake)) #TODO: somethigg
G_loss =  - tf.reduce_mean(tf.log(D_fake))

tf.summary.scalar('G_loss',G_loss)
#tf.summary.scalar('D_loss', D_loss)

merged = tf.summary.merge_all()



D_solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(D_loss, var_list=D_vars)
G_solver =  tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars)


mb_size = 64
Z_dim = 100
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./train',sess.graph)
    test_writer = tf.summary.FileWriter('./test')

    init = tf.global_variables_initializer()


    sess.run(init)
    for it in range(1001):
        print('%s'%str(it))
        X_mb, _  = mnist.train.next_batch(mb_size)
        X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

        summary_D,_, D_loss_curr = sess.run([merged,D_solver, D_loss], feed_dict={X:X_mb , Z: sample_Z(mb_size, Z_dim)})
        summary_G,_, G_loss_curr = sess.run([merged,G_solver, G_loss], feed_dict={Z:sample_Z(mb_size, Z_dim)})

        #train_writer.add_summary(summary_D, it)
        #train_writer.add_summary(summary_G, it)
        if it % 500 == 0:
            img = sess.run([G_sample], feed_dict={X:X_mb, Z: sample_Z(mb_size, Z_dim)})
            print(img)
            img = img[0].reshape([-1, 28, 28])
            plot_batch(img)

            #test_writer.add_summary(summary_D, it)
            #test_writer.add_summary(summary_G, it)

    plt.show()