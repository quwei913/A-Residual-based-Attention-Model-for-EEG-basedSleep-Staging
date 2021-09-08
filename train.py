#! /usr/bin/python
# -*- coding: utf8 -*-
import os

import tensorflow as tf
from trainer import Trainer
from utils import NUM_CLASSES, EPOCH_SEC_LEN, SAMPLING_RATE


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('epochs', 100,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_string('dataset', 'sleep-EDF',
                            """Dataset name.""")


def main(argv=None):
    # Output dir
    output_dir = os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))

    if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
    tf.gfile.MakeDirs(output_dir)
    
    if FLAGS.dataset not in SAMPLING_RATE.keys():
        raise Exception("Dataset not supported")

    trainer = Trainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir,
        fold_idx=FLAGS.fold_idx,
        batch_size=10, 
        n_classes=NUM_CLASSES,
        input_dims=EPOCH_SEC_LEN * SAMPLING_RATE[FLAGS.dataset],
        interval_print_cm=10,
        dataset=FLAGS.dataset,
        seq_length=30
    )
    trainer.train(n_epochs=FLAGS.epochs)


if __name__ == "__main__":
    tf.app.run()
