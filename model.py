from attention import Attention, positional_encoding
from nn import *
import tensorflow as tf


class Model(object):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            seq_length,
            is_train,
            reuse_params,
            name='staging'
    ):

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.name = name

        self.activations = []
        self.layer_idx = 1

    def residual_layer(self, input_x, out_dim, depth=64):
        input_dim = int(input_x.get_shape()[-1])

        if input_dim * 2 == out_dim:
            stride = 2
            flag = True
            channel = input_dim // 2
        else:
            stride = 1
            flag = False
            
        x = self._conv1d_layer(input_var=input_x, filter_size=1, n_filters=depth, stride=stride)
        x = self._conv1d_layer(input_var=x, filter_size=3, n_filters=depth, stride=1)
        
        x = self._conv1d_layer(x, n_filters=out_dim, filter_size=1, stride=1, activate=False)

        if flag is True :
            pad_input_x = tf.nn.avg_pool(input_x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
            pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [input_dim// 2, input_dim// 2]])
        else:
            pad_input_x = input_x

        input_x = tf.nn.relu(tf.contrib.layers.layer_norm(x + pad_input_x))

        return input_x

    def build_encoder(self, input_var):
        input_dim = input_var.get_shape().as_list()[1]
        filter_size = input_dim // 100 if input_dim % 100 == 0 else input_dim // 100 + 1
        stride = pool1_size = filter_size // 10
        pool2_size = int(3.14 * pool1_size) + 1
        print("filter_size: {}, stride: {}, pool1_size: {}, pool2_size: {}".format(filter_size, stride, pool1_size, pool2_size))

        network = self._conv1d_layer(input_var=input_var, filter_size=filter_size, n_filters=64, stride=stride, wd=1e-3)
        network1 = self.max_pool_1d(pool_size=pool1_size, stride=stride, input_var=network)
        network2 = self.max_pool_1d(pool_size=pool2_size, stride=stride, input_var=network)
        #network = tf.contrib.layers.layer_norm(tf.concat([network2, network1], 1))
        network = tf.contrib.layers.layer_norm(network2 + network1)

        network = self.residual_layer(network, 128)
        network = self.residual_layer(network, 128)

        network = self.global_avg_pool(network, [1, 2])
        network = self.drop_out(network, 0.5)

        return network

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0, activate=True):
        input_shape = input_var.get_shape()
        n_in_filters = input_shape[3].value
        name = "l{}_conv".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output = conv_1d(name="conv1d", input_var=input_var, filter_shape=[filter_size, 1, n_in_filters, n_filters],
                             stride=stride, bias=None, wd=wd)
            output = batch_norm(name="bn", input_var=output, is_train=self.is_train)

            if activate:
                output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def drop_out(self, network, keep_prob=0.5):
        name = "l{}_dropout".format(self.layer_idx)
        self.layer_idx += 1
        if self.is_train:
            network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            self.activations.append((name, network))
        return network

    def max_pool_1d(self, pool_size, stride, input_var, padding="SAME"):
        name = "l{}_max_pool".format(self.layer_idx)
        network = tf.nn.max_pool(input_var, ksize=[1, pool_size, 1, 1],
                                 strides=[1, stride, 1, 1], padding=padding,
                                 name=name)
        self.activations.append((name, network))
        self.layer_idx += 1
        return network

    def avg_pool_1d(self, pool_size, stride, input_var, padding="SAME"):
        name = "l{}_avg_pool".format(self.layer_idx)
        network = tf.nn.avg_pool(input_var, ksize=[1, pool_size, 1, 1],
                                 strides=[1, stride, 1, 1], padding=padding,
                                 name=name)
        self.activations.append((name, network))
        self.layer_idx += 1
        return network

    def global_avg_pool(self, input_var, reduction_indices):
        name = "l{}_global_avg_pool".format(self.layer_idx)
        network = tf.reduce_mean(input_var, reduction_indices=reduction_indices, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1
        return network

    def _build_placeholder(self):
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size * self.seq_length, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        self.target_var = tf.placeholder(
            tf.int32,
            shape=[self.batch_size * self.seq_length, ],
            name=name + "_targets"
        )
        self.Y = tf.one_hot(self.target_var, depth=self.n_classes, axis=1, dtype=tf.float32)
        self.lr = tf.placeholder(
            tf.float32,
            shape=[]
        )

    def build_model(self):
        network = self.build_encoder(input_var=self.input_var)
        self.cls_logits = fc(name='fc1', input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)

        output_conns = [network]
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1].value
        seq_input = tf.reshape(network,
                               shape=[-1, self.seq_length, input_dim],
                               name=name)
        assert self.batch_size == seq_input.get_shape()[0].value
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        positional_encoded = positional_encoding(input_dim, self.seq_length)
        position_inputs = tf.tile(tf.range(0, self.seq_length), [self.batch_size])
        position_inputs = tf.reshape(position_inputs, [self.batch_size, self.seq_length])
        seq_input = tf.contrib.layers.layer_norm(tf.add(seq_input,
                        tf.nn.embedding_lookup(positional_encoded, position_inputs)))

        network = Attention(num_heads=8,
                            linear_key_dim=512,
                            linear_value_dim=512,
                            model_dim=input_dim,
                            is_train=self.is_train,
                            dropout=0.2).multi_head(seq_input, seq_input, seq_input)
        network = tf.contrib.layers.layer_norm(tf.add(network, seq_input))

        seq_input = network
        network = tf.layers.dense(network, 2048, activation=tf.nn.relu)
        network = tf.layers.dense(network, input_dim)
        network = tf.contrib.layers.layer_norm(tf.add(network, seq_input))

        network = tf.reshape(network, shape=[-1, input_dim])
        output_conns.append(network)
        network = tf.contrib.layers.layer_norm(tf.add_n(output_conns))
        network = self.drop_out(network, 0.5)
      
        return network


    def init_ops(self, cls_wt=1.0):
        self._build_placeholder()

        with tf.variable_scope(self.name) as scope:
            if self.reuse_params:
                scope.reuse_variables()

            network = self.build_model()

            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            self.logits = network

            class_weights = tf.constant(cls_wt)
            weights = tf.reduce_sum(class_weights * self.Y, axis=1)
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [self.Y],
                [1.],
                softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits_v2,
                name="sequence_loss_by_example"
            )
            seq_loss = tf.reduce_sum(loss) / self.batch_size / self.seq_length
            
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.cls_logits,
                labels=self.Y,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            weighted_losses = loss * weights
            loss = tf.reduce_mean(weighted_losses, name="cross_entropy")
            
            regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )
            self.loss_op = tf.add_n([loss, regular_loss, seq_loss])
            self.seq_loss = seq_loss

            self.pred_op = tf.argmax(self.logits, 1)

