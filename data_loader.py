import os

import numpy as np

from utils import print_n_samples_each_class
import re


class SeqDataLoader(object):

    def __init__(self, data_dir, dataset, fold_idx):
        self.data_dir = data_dir
        self.dataset = dataset
        self.fold_idx = fold_idx

    @staticmethod
    def _load_npz_file(npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _train_test_split(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for f in allfiles:
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        subject_files = []
        if self.dataset == 'MASS':
            if self.fold_idx < 0 or self.fold_idx > 30:
                 raise Exception("Only support up to 31 folds for MASS dataset")
            subject_files = npzfiles[self.fold_idx*2:self.fold_idx*2+2]
        else:
            if self.fold_idx < 0 or self.fold_idx > 19:
                 raise Exception("Only support up to 20 folds for sleep-EDF dataset")
            for f in allfiles:
                if self.fold_idx < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
                if pattern.match(f):
                    subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()
        return train_files, subject_files

    def load_train_data(self):
        train_files, subject_files = self._train_test_split()
        # Load training and validation sets
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(subject_files)
        print

        print("Training set: n_records={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            print(d.shape)
            n_train_examples += d.shape[0]
        print("Number of examples = {}".format(n_train_examples))
        print_n_samples_each_class(np.hstack(label_train))
        print
        print("Validation set: n_records={}".format(len(data_val)))
        n_valid_examples = 0
        for d in data_val:
            print(d.shape)
            n_valid_examples += d.shape[0]
        print("Number of examples = {}".format(n_valid_examples))
        print_n_samples_each_class(np.hstack(label_val))
        print

        return data_train, label_train, data_val, label_val

    def load_subject_data(self):
        _, subject_files = self._train_test_split()

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        print("Load test data from: {}".format(subject_files))
        data, labels = self._load_npz_list_files(subject_files)

        return data, labels
