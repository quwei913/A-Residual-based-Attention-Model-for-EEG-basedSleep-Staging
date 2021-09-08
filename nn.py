import tensorflow as tf
from tensorflow.python.training import moving_averages
from utils import SEED


def _create_variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, wd=None):
    initializer = tf.contrib.layers.variance_scaling_initializer(seed=SEED[tf.app.flags.FLAGS.dataset])
    # Create or get the existing variable
    var = _create_variable(
        name,
        shape,
        initializer
    )

    # L2 weight decay
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)

    return var


def conv_1d(name, input_var, filter_shape, stride, padding="SAME", 
            bias=None, wd=None):
    with tf.variable_scope(name) as scope:
        # Trainable parameters
        kernel = variable_with_weight_decay(
            "weights",
            shape=filter_shape,
            wd=wd
        )

        # Convolution
        output_var = tf.nn.conv2d(
            input_var,
            kernel,
            [1, stride, 1, 1],
            padding=padding
        )

        # Bias
        if bias is not None:
            biases = _create_variable(
                "biases",
                [filter_shape[-1]],
                tf.constant_initializer(bias)
            )
            output_var = tf.nn.bias_add(output_var, biases)

        return output_var

def fc(name, input_var, n_hiddens, bias=None, wd=None):
    with tf.variable_scope(name) as scope:
        # Get input dimension
        input_dim = input_var.get_shape()[-1].value

        # Trainable parameters
        weights = variable_with_weight_decay(
            "weights",
            shape=[input_dim, n_hiddens],
            wd=wd
        )

        # Multiply weights
        output_var = tf.matmul(input_var, weights)

        # Bias
        if bias is not None:
            biases = _create_variable(
                "biases",
                [n_hiddens],
                tf.constant_initializer(bias)
            )
            output_var = tf.add(output_var, biases)

        return output_var


def batch_norm(name, input_var, is_train, decay=0.999, epsilon=1e-5):
    """Batch normalization modified from BatchNormLayer in Tensorlayer.
    Source: <https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py#L2190>
    """

    inputs_shape = input_var.get_shape()
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]

    with tf.variable_scope(name) as scope:
        # Trainable beta and gamma variables
        beta = tf.get_variable('beta',
                                shape=params_shape,
                                initializer=tf.zeros_initializer(), trainable=is_train)
        gamma = tf.get_variable('gamma',
                                shape=params_shape,
                                initializer=tf.random_normal_initializer(mean=1.0, stddev=0.002),
                                trainable=is_train)
        
        # Moving mean and variance updated during training
        moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False)
        
        # Compute mean and variance along axis
        batch_mean, batch_variance = tf.nn.moments(input_var, axis, name='moments')

        # Define ops to update moving_mean and moving_variance
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=False)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=False)

        # Define a function that :
        # 1. Update moving_mean & moving_variance with batch_mean & batch_variance
        # 2. Then return the batch_mean & batch_variance
        def mean_var_with_update():
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        # Perform different ops for training and testing
        if is_train:
            mean, variance = mean_var_with_update()
            normed = tf.nn.batch_normalization(input_var, mean, variance, beta, gamma, epsilon)
        
        else:
            normed = tf.nn.batch_normalization(input_var, moving_mean, moving_variance, beta, gamma, epsilon)

        return normed

def adam(loss, lr, train_vars, beta1=0.9, beta2=0.999, epsilon=1e-8):
    opt = tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        name="Adam"
    )
    grads_and_vars = opt.compute_gradients(loss, train_vars)
    apply_gradient_op = opt.apply_gradients(grads_and_vars)
    return apply_gradient_op, grads_and_vars

