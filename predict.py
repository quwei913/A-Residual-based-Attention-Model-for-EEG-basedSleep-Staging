#! /usr/bin/python
# -*- coding: utf8 -*-
import os
import time
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from data_loader import SeqDataLoader
from model import Model
from utils import iterate_batch_seq_minibatches, NUM_CLASSES, EPOCH_SEC_LEN


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('model_dir', 'output',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save outputs.""")
tf.app.flags.DEFINE_string('dataset', 'sleep-EDF',
                           """Dataset name.""")


def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1, ck):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
    reg_loss_value = sess.run(reg_loss)

    # Print performance
    print(
        "duration={:.3f} sec, n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
        "f1={:.3f}, ck={:.3f}".format(
            duration, n_examples, loss, reg_loss_value, acc, f1, ck
        )
    )
    print(cm)
    print


def run_epoch(
    sess, 
    network, 
    inputs, 
    targets, 
    train_op,
    output_dir, 
    subject_idx
):
    start_time = time.time()
    y = []
    y_true = []
    total_loss, n_batches = 0.0, 0
    for sub_f_idx, each_data in enumerate(zip(inputs, targets)):
        each_x, each_y = each_data

        # Store prediction and actual stages of each patient
        each_y_true = []
        each_y_pred = []

        for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                              targets=each_y,
                                                              batch_size=network.batch_size,
                                                              seq_length=network.seq_length):
            feed_dict = {
                network.input_var: x_batch,
                network.target_var: y_batch
            }

            _, loss_value, y_pred = sess.run(
                [train_op, network.loss_op, network.pred_op],
                feed_dict=feed_dict
            )

            # Extract memory cells
            each_y_true.extend(y_batch)
            each_y_pred.extend(y_pred)

            total_loss += loss_value
            n_batches += 1

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        y.append(each_y_pred)
        y_true.append(each_y_true)

    # Save memory cells and predictions
    save_dict = {
        "y_true": y_true,
        "y_pred": y
    }
    save_path = os.path.join(
        output_dir,
        "output_subject{}.npz".format(subject_idx)
    )
    np.savez(save_path, **save_dict)
    print("Saved outputs to {}".format(save_path))

    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)

    return total_y_true, total_y_pred, total_loss, duration


def predict(
    data_dir, 
    model_dir, 
    output_dir, 
    n_subjects, 
    dataset,
    n_subjects_per_fold
):
    # Ground truth and predictions
    y_true = []
    y_pred = []

    # The model will be built into the default Graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build the network
        valid_net = Model(
            batch_size=1, 
            input_dims=EPOCH_SEC_LEN*100 if dataset == 'sleep-EDF' else EPOCH_SEC_LEN*256, 
            n_classes=NUM_CLASSES, 
            seq_length=30,
            is_train=False, 
            reuse_params=False
        )

        # Initialize parameters
        valid_net.init_ops()

        for subject_idx in range(n_subjects):
            fold_idx = subject_idx // n_subjects_per_fold

            checkpoint_path = os.path.join(
                model_dir, 
                "fold{}".format(fold_idx), 
                "staging"
            )
            # Restore the trained model
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

            data_loader = SeqDataLoader(
                data_dir=data_dir,
                dataset=dataset,
                fold_idx=fold_idx
            )

            # Load testing data
            x, y = data_loader.load_subject_data()

            # Loop each epoch
            print("[{}] Predicting ...\n".format(datetime.now()))

            # Evaluate the model on the subject data
            y_true_, y_pred_, loss, duration = \
                run_epoch(
                    sess=sess, network=valid_net,
                    inputs=x, targets=y,
                    train_op=tf.no_op(),
                    output_dir=output_dir,
                    subject_idx=subject_idx
                )
            n_examples = len(y_true_)
            cm_ = confusion_matrix(y_true_, y_pred_)
            acc_ = np.mean(y_true_ == y_pred_)
            mf1_ = f1_score(y_true_, y_pred_, average="macro")
            ck_ = cohen_kappa_score(y_true_, y_pred_)

            # Report performance
            print_performance(
                sess, valid_net.name,
                n_examples, duration, loss, 
                cm_, acc_, mf1_, ck_
            )

            y_true.extend(y_true_)
            y_pred.extend(y_pred_)
        
    # Overall performance
    print("[{}] Overall prediction performance\n".format(datetime.now()))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_examples = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    ck = cohen_kappa_score(y_true, y_pred)
    print(
        "n={}, acc={:.3f}, f1={:.3f}, ck={:.3f}".format(
            n_examples, acc, mf1, ck
        )
    )
    print(cm)
    return y_true, y_pred


def main(argv=None):

    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    n_subjects = 20 if FLAGS.dataset == 'sleep-EDF' else 31
    n_subjects_per_fold = 1
    predict(
        data_dir=FLAGS.data_dir,
        model_dir=FLAGS.model_dir,
        output_dir=FLAGS.output_dir,
        n_subjects=n_subjects,
        dataset=FLAGS.dataset,
        n_subjects_per_fold=n_subjects_per_fold
    )


if __name__ == "__main__":
    tf.app.run()
