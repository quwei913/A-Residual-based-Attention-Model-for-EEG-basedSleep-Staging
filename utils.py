import numpy as np

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

NUM_CLASSES = 5  # exclude UNKNOWN

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

SAMPLING_RATE = {
    "MASS": 256,
    "sleep-EDF": 100
}

SEED = {
    "MASS": 1,
    "sleep-EDF": 7
}

EPOCH_SEC_LEN = 30  # seconds


def print_n_samples_each_class(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))


def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size

    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:],
                          dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    arr = np.arange(epoch_size)
    for i in arr:
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        yield flatten_x, flatten_y

def get_cls_weight(labels):
    unique_labels = np.unique(labels)
    all_labels = [0] * len(class_dict)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        all_labels[c] = n_samples
    sum_n = sum(all_labels)
    all_labels = [sum_n / n if n != 0 else 0 for n in all_labels]
    sum_n = sum(all_labels)
    return [len(unique_labels) * n / sum_n for n in all_labels]
