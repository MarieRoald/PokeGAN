from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def leaky_relu(x):
    return tf.nn.relu(x) - 0.001*tf.nn.relu(-x)


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
            initializer=tf.zeros([out_size])  # TODO: FIX STUFF
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


def upconv_layer(x, out_size, activation_function=lambda x: x, name=None, k_size=3, stride=1):
    with tf.name_scope(name):
        params = {'W': None, 'b': None}
        shape = x.get_shape().as_list()

        params['W'] = tf.get_variable(
            'W',
            [k_size, k_size, out_size, shape[-1]],
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )

        params['b'] = tf.get_variable(
            'b',
            initializer=tf.zeros([out_size])  # TODO: FIX STUFF
        )

        out = params['b'] + tf.nn.conv2d_backprop_input(
            input_sizes=x.get_shape(),
            filter=params['W'],
            out_backprop=x,
            strides=(1, stride, stride, 1),
            padding='SAME'
        )
        print(shape[-1])

        variable_summaries(params['W'])
        # variable_summaries(params['b'])

        return activation_function(out), params


def reshape(x, out_size, **kwargs):
    return tf.reshape(x, out_size), {}


def create_nn(in_var, layers, is_training, reuse_vars=None, name=None):
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
                     `name` - The name of this layer.
                     Optionally, the function can take other arguments,
                     passed as **kwargs
            `out_size`: The no. of channels out of the corresponding layer.
            `activation_function`: The activation function the corresponding
            layer should use, e.g. tf.nn.relu, if this is None it is not
            passed.
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
            assert 'name' in layer
            assert 'use_batchnorm' in layer
            if 'activation_function' in layer:
                layer['kwargs']['activation_function'] = layer['activation_function']

            with tf.variable_scope(layer['name'], reuse=reuse_vars) as var_scope:
                out, curr_params = layer['layer'](
                    x=out,
                    **layer['kwargs']
                )

                if layer['use_batchnorm']:
                    out = tf.contrib.layers.batch_norm(out, is_training=is_training)

            for key, val in curr_params.items():
                params['{}/{}'.format(layer['name'], key)] = val

    return out, params
