import argparse
import zipfile
import glob
import ntpath
import os
import codecs
import numpy as np
from mne.io import concatenate_raws, read_raw_edf
from rarfile import RarFile
import dhedfreader
import gzip
import shutil

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="MASS/SS3",
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default="MASS/eeg_f4",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG F4",
                        help="File path to the trained model used to estimate walking speeds, multiple channels should be delimited by comma.")
    args = parser.parse_args()
    
    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    fnames = glob.glob(os.path.join(args.data_dir, "SS*.zip"))

    for f in fnames:
        zip_ref = zipfile.ZipFile(f, 'r')
        zip_ref.extractall(args.output_dir)
        zip_ref.close()
        os.remove(f)
        print("Unzipped {}, remove to free space".format(f))
    ver_2014 = glob.glob(os.path.join(args.output_dir, '*/version2014'))
    for d in ver_2014:
        shutil.rmtree(d)
        print("Deleted 2014 version of annotations {}".format(d))
    ann_fnames = glob.glob(os.path.join(args.output_dir, "*/annotations/MASS*.rar"))
    for f in ann_fnames:
        rf = RarFile(f)
        rf.extractall(path=os.path.dirname(f))

    unrarann_fnames = glob.glob(os.path.join(args.output_dir, "*/annotations/MASS-*-EDF/*Annotations.edf.gz"))
    for f in unrarann_fnames:
        with gzip.open(f, 'rb') as f_in:
            out_f = f.replace('.gz', '')
            with open(out_f, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                os.remove(f)
                print("Delete gz file {} to free space".format(f))

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.output_dir, "*/*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.output_dir, "*/annotations/MASS-*-EDF/*Annotations.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)
    for i in range(len(psg_fnames)):
        select_ch = "{}-CLE".format(args.select_ch)
        unzipped_dir = os.path.dirname(psg_fnames[i])
        filename = os.path.join(unzipped_dir,
                                ntpath.basename(psg_fnames[i]).replace(" PSG.edf", ".npz"))
        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        if select_ch not in raw.info['ch_names']:
            ler_ch = select_ch.replace('-CLE', '-LER')
            print("Cannot find {}, try {}...".format(select_ch, ler_ch))
            select_ch = ler_ch
            if select_ch not in raw.info['ch_names']:
                print("Cannot find {}, skip...".format(select_ch))
                continue
                
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        max_len = len(raw_ch_df)

        # Get raw header
        f = codecs.open(psg_fnames[i], 'r', encoding="utf-8")
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()

        # Read annotation and its header
        f = codecs.open(ann_fnames[i], 'r', encoding="utf-8")
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        rectime, _, ann = zip(*reader_ann.records())
        f.close()

        # Generate label indices
        labels = []  # indicies of the data that have labels
        label_idx = []
        for n, a in enumerate(ann):
            # skip not labeled
            if not a: continue
            onset_sec = rectime[n]
            for an in a:
                _, duration_sec, ann_char = an
                ann_str = "".join([ac.decode('utf-8') for ac in ann_char])
                # skip invalid label
                if ann_str not in ann2label: continue
                label = ann2label[ann_str]
                if label != UNKNOWN:
                    if duration_sec % EPOCH_SEC_SIZE != 0:
                        raise Exception("Duration should be 30 seconds")

                    idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                    # cannot be split into 30s epoch
                    if idx[-1] > max_len - 1: break
                    label_idx.append(idx)
                    labels.append(label)

                    print("Include onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str
                    ))
                    break
        label_idx = np.hstack(label_idx)
        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        selected_idx = np.intersect1d(np.arange(len(raw_ch_df)), label_idx)
        print("after remove unwanted: {}".format(selected_idx.shape))

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[selected_idx]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Cannot be splitted into 30-s epochs")

        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = np.asarray(labels).astype(np.int32)

        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))
        
        # downsampling
        if int(sampling_rate / 256.0) < 1:
            print("Invalid sampling rate {}, skip...".format(sampling_rate))
            continue
        elif int(sampling_rate / 256.0) > 1:
            downsample_ratio = int(sampling_rate / 256.0)
            sampling_rate = 256.0
            print("Downsampling with ratio: {}".format(downsample_ratio))
            x = x[::, ::downsample_ratio, ::]
            print("Data after downsampling: {}, {}".format(x.shape, y.shape))

        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": args.select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }
        np.savez(filename, **save_dict)
        print("\n=======================================\n")


if __name__ == "__main__":
    main()
