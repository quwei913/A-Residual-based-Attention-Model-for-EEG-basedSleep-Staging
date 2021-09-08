import os
import time
import sys
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score
from data_loader import SeqDataLoader
from utils import get_cls_weight, iterate_batch_seq_minibatches
import numpy as np
import tensorflow as tf
from model import Model
from nn import adam


class Trainer(object):

    def __init__(
            self,
            data_dir,
            output_dir,
            fold_idx,
            batch_size,
            n_classes,
            seq_length,
            input_dims,
            dataset,
            interval_print_cm=10
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.input_dims = input_dims
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.seq_length = seq_length
        self.dataset = dataset
        self.interval_print_cm = interval_print_cm

    def print_performance(self, sess, network_name,
                          n_train_examples, n_valid_examples,
                          train_cm, valid_cm, epoch, n_epochs,
                          train_duration, train_loss, train_acc, train_f1,
                          valid_duration, valid_loss, valid_acc, valid_f1,
                          lr, train_seq_loss, valid_seq_loss):
        
        losses = tf.get_collection("losses", scope=network_name + "\/")
        train_reg_loss = tf.add_n(losses)
        train_reg_loss_value = sess.run(train_reg_loss)
        valid_reg_loss_value = train_reg_loss_value

        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print
            print("[{}] epoch {}/{}:".format(
                datetime.now(), epoch + 1, n_epochs
            ))
            print(
                "train ({:.3f} sec): n={}, loss={:.3f} ({:.3f}|{:.3f}|{:.3f}), acc={:.3f}, "
                "f1={:.3f}, lr={:.8f}".format(
                    train_duration, n_train_examples,
                    train_loss, train_loss - train_seq_loss - train_reg_loss_value,
                    train_seq_loss, train_reg_loss_value, 
                    train_acc, train_f1, lr
                )
            )
            print(train_cm)
            print(
                "valid ({:.3f} sec): n={}, loss={:.3f} ({:.3f}|{:.3f}|{:.3f}), acc={:.3f}, "
                "f1={:.3f}, lr={:.8f}".format(
                    valid_duration, n_valid_examples,
                    valid_loss, valid_loss - valid_seq_loss - valid_reg_loss_value,
                    valid_seq_loss, valid_reg_loss_value,
                    valid_acc, valid_f1, lr
                )
            )
            print(valid_cm)
            print
        else:
            print(
                "epoch {}/{}: "
                "train ({:.2f} sec): n={}, loss={:.3f} ({:.3f}|{:.3f}|{:.3f}), "
                "acc={:.3f}, f1={:.3f} | "
                "valid ({:.2f} sec): n={}, loss={:.3f} ({:.3f}|{:.3f}|{:.3f}), "
                "acc={:.3f}, f1={:.3f}, lr={:.8f}".format(
                    epoch + 1, n_epochs,
                    train_duration, n_train_examples,
                    train_loss, train_loss - train_seq_loss - train_reg_loss_value,
                    train_seq_loss, train_reg_loss_value,
                    train_acc, train_f1,
                    valid_duration, n_valid_examples,
                    valid_loss, valid_loss - valid_seq_loss - valid_reg_loss_value,
                    valid_seq_loss, valid_reg_loss_value,
                    valid_acc, valid_f1, lr
                )
            )
        sys.stdout.flush()

    def _run_epoch(self, sess, network, inputs, targets, train_op, is_train, lr):
        start_time = time.time()
        y = []
        y_true = []
        total_loss, total_seq_loss, n_batches = 0.0, 0.0, 0
        for each_x, each_y in zip(inputs, targets):
            for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                                  targets=each_y,
                                                                  batch_size=self.batch_size,
                                                                  seq_length=self.seq_length):
                feed_dict = {
                    network.input_var: x_batch,
                    network.target_var: y_batch,
                    network.lr: lr
                }

                _, loss_value, seq_loss_value, y_pred = sess.run(
                    [train_op, network.loss_op, network.seq_loss, network.pred_op],
                    feed_dict=feed_dict
                )

                total_loss += loss_value
                total_seq_loss += seq_loss_value
                n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)

                assert not np.isnan(loss_value), \
                    "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss /= n_batches
        total_seq_loss /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss, duration, total_seq_loss

    def train(self, n_epochs):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            
            train_net = Model(
                batch_size=self.batch_size,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                is_train=True,
                input_dims=self.input_dims,
                reuse_params=False
            )
            valid_net = Model(
                batch_size=self.batch_size,
                n_classes=self.n_classes,
                seq_length=self.seq_length,
                is_train=False,
                input_dims=self.input_dims,
                reuse_params=True
            )

            data_loader = SeqDataLoader(
                data_dir=self.data_dir,
                dataset=self.dataset,
                fold_idx=self.fold_idx
            )
            x_train, y_train, x_valid, y_valid = data_loader.load_train_data()

            all_train_loss = np.zeros(n_epochs)
            all_train_seq_loss = np.zeros(n_epochs)
            all_train_acc = np.zeros(n_epochs)
            all_train_f1 = np.zeros(n_epochs)
            all_valid_loss = np.zeros(n_epochs)
            all_valid_acc = np.zeros(n_epochs)
            all_valid_f1 = np.zeros(n_epochs)

            train_net.init_ops(get_cls_weight(np.hstack(y_train)))
            valid_net.init_ops(get_cls_weight(np.hstack(y_valid)))

            print("Network (layers={})".format(len(train_net.activations)))
            print("inputs ({}): {}".format(
                train_net.input_var.name, train_net.input_var.get_shape()
            ))
            print("targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            ))
            for name, act in train_net.activations:
                print("{} ({}): {}".format(name, act.name, act.get_shape()))
            print

            train_op, grads_and_vars_op = adam(
                loss=train_net.loss_op,
                lr=train_net.lr,
                train_vars=tf.trainable_variables()
            )
            
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with tf.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

            sess.run(tf.global_variables_initializer())

            print("[{}] Start training ...\n".format(datetime.now()))

            for epoch in range(sess.run(global_step), n_epochs):
                lr = 1e-4
                y_true_train, y_pred_train, train_loss, train_duration, train_seq_loss = \
                    self._run_epoch(
                        sess=sess, network=train_net,
                        inputs=x_train, targets=y_train,
                        train_op=train_op,
                        is_train=True, lr=lr
                    )
                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train)
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro")

                y_true_val, y_pred_val, valid_loss, valid_duration, valid_seq_loss = \
                    self._run_epoch(
                        sess=sess, network=valid_net,
                        inputs=x_valid, targets=y_valid,
                        train_op=tf.no_op(),
                        is_train=False, lr=lr
                    )
                n_valid_examples = len(y_true_val)
                valid_cm = confusion_matrix(y_true_val, y_pred_val)
                valid_acc = np.mean(y_true_val == y_pred_val)
                valid_f1 = f1_score(y_true_val, y_pred_val, average="macro")

                all_train_loss[epoch] = train_loss
                all_train_seq_loss[epoch] = train_seq_loss
                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_valid_loss[epoch] = valid_loss
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1

                self.print_performance(
                    sess, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss, train_acc, train_f1,
                    valid_duration, valid_loss, valid_acc, valid_f1,
                    lr, train_seq_loss, valid_seq_loss
                )

                sess.run(tf.assign(global_step, epoch + 1))
                if (epoch + 1) == n_epochs:
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_fold{}.ckpt".format(self.fold_idx)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print("Saved model checkpoint ({:.3f} sec)".format(duration))

                    start_time = time.time()
                    save_dict = {}
                    for v in tf.global_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_fold{}.npz".format(self.fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print("Saved trained parameters ({:.3f} sec)".format(duration))

        print("Finish training")
