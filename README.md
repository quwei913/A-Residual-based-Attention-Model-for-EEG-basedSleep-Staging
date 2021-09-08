# A Residual based Attention Model for EEG based Sleep Staging
This repo is the implementation of the paper:
`A Residual based Attention Model for EEG basedSleep Staging`. Link [here](https://ieeexplore.ieee.org/document/9022981).

## Description
A residual based attention model for sleep staging using EEG.
The model contains two parts: feature extraction and temporal context learning.
The feature extraction part consists of two CNNs inspired by Hilbert Transform.
The temporal context learning is taking the advantage of transformer.
The model can be trained end to end with cost sensitive learning incorporated.


## Mode Architecture
The architecture of the network:

![architecture](./overview.jpg)

## How to run
Use `data/download_physionet.sh` to download sleep-EDF.

Use `prepare_physionet.py` and `prepare_mass.py` to prepare sleep-EDF and MASS respectively.

Use `batch_train.sh` to run training on each fold.

For example, run a training for 31 folds on channel C3 using GPU 2:

```./batch_train.sh mass/C3 output_C3 0 30 MASS 2```

## Citation

If you find this useful, please cite our works as following:

```
@article{9022981,
author = {Wei, Q and Wang, Z and Hong, H and Chi, Z and Feng, D D and Grunstein, R and Gordon, C},
doi = {10.1109/JBHI.2020.2978004},
journal = {IEEE Journal of Biomedical and Health Informatics},
mendeley-groups = {Phenotyping},
pages = {2833--2843},
title = {{A residual based attention model for EEG based sleep staging}},
volume = {24},
year = {2020}
}
```