from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

print(mnist)

def fc_layer(x, out_size, activation_function=lambda x: x):
    params = {'W': None, 'b': None}

    shape = x.get_shape().as_list()

    input_is_vector = len(shape) == 2
    last_dim = shape[-1] if input_is_vector else np.prod(shape[1:])

    params['W'] = tf.get_variable("W", [last_dim, out_size],initializer=tf.random_normal_initializer(stddev=tf.cast(tf.sqrt(1./(out_size*last_dim)), tf.float32)));params['b'] = tf.get_variable("b", [out_size], initializer=tf.constant_initializer(0.0))

    out = tf.matmul(x, params['W']) if input_is_vector else tf.matmul(tf.reshape(x, (-1, last_dim)), params['W']) + params['b']

    return activation_function(out), params

def generator(z):
    with tf.variable_scope("fc1"):
        gen_fc, params_fc = fc_layer(z, 128, tf.nn.relu)

    with tf.variable_scope("fc2"):
        gen_prob, params_prob = fc_layer(gen_fc, 28*28, tf.nn.sigmoid)

    gen_vars = []
    for key_fc, key_prob in zip(params_fc, params_prob):
        gen_vars += [params_fc[key_fc], params_prob[key_prob]]

    return gen_prob, gen_vars

def discriminator(z )  :
    with tf.variable_scope("fc1"):
        disc_fc, params_fc = fc_layer(z, 128, tf.nn.relu)

    with tf.variable_scope("fc2"):
        disc_logit, params_prob = fc_layer(disc_fc, 1)
        disc_prob = tf.nn.sigmoid(disc_logit)

    disc_vars = []
    for key_fc, key_prob in zip(params_fc, params_prob):
        disc_vars += [params_fc[key_fc], params_prob[key_prob]]

    return disc_prob, disc_logit, disc_vars

with tf.variable_scope('G'):
    Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
    G_sample ,G_vars   = generator(Z  )

with tf.variable_scope('D') as scope:
    X = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='X')

    D_real, D_logit_real, D_vars = discriminator(X)
    scope.reuse_variables()
    D_fake, D_logit_fake, _  = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake)) #TODO: somethigg
G_loss =  - tf.reduce_mean(tf.log(D_fake))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_vars )
G_solver =  tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars)


def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])

mb_size = 64
Z_dim = 100


def plot_batch(batch):
    N,img_width,img_height = batch.shape

    num_cols = int(np.sqrt(N))
    num_rows = N/num_cols

    grid = np.zeros([img_height*num_rows,img_width*num_cols])

    for r in range(num_rows):
        for c in range(num_cols):
            grid[r*img_width:r*img_width + img_width,c*img_height:c*img_height + img_height] = batch[r*num_rows + c]

    plt.imshow(grid,interpolation='None', cmap='gray')
    plt.show()


import matplotlib.pyplot as plt
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for it in range(1000000):
        print('%s'%str(it))
        X_mb, _  = mnist.train.next_batch(64)

        _, D_loss_curr = sess.run([D_solver, D_loss ], feed_dict={X:X_mb , Z: sample_Z(mb_size,100)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z:sample_Z(mb_size,100)})

        if it % 5000 == 0:
            img = sess.run([G_sample], feed_dict={X:X_mb , Z: sample_Z(mb_size,100)})
            print(img)
            img = img[0].reshape([-1,28,28])
            plot_batch(img)


















































