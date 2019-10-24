# kd-switch


This repository contains a Python implementation of the kd-switch online predictor and the KDS-seq sequential two-sample test proposed in our paper:

    Low-Complexity Nonparametric Bayesian Online Prediction with Universal Guarantees
    Alix Lhéritier & Frédéric Cazals
    NeurIPS 2019

[ArXiv version](https://arxiv.org/abs/1901.07662)

It also contains a Python implementation of the k-nearest neighbors based online predictor and the KNN-seq sequential two-sample test of our previous paper:

    A Sequential Non-Parametric Multivariate Two-Sample Test
    Alix Lhéritier & Frédéric Cazals
    IEEE Transactions on Information Theory, 64(5):3361–3370, 2018.


## Dependencies

* python >= 2.7
* numpy 
* h5py
* lxml
* scikit-learn
* scipy
* pandas
* matplotlib
* psutil
* pomegranate



## Reproducing experimental results

Each experiment is defined in its own bash script file in the scripts folder. 
The trial index determines the seed used for randomness. For each trial, two files are generated: an xml with the summary of the results and a text file containing elpased time, predicted probability for the observed label and cumulated log loss for each data point. 

In order to reproduce the results of the paper, you should run each of the following scripts in an empty folder: 

```
experiment1.sh 
experiment2.sh
```

The DATAPATH variable needs to point to the folder containing HIGGS and GANMNIST datasets.
The PYTHONPATH variable needs to include the code and scripts folders.


## Higgs boson dataset

In our experiments, we used a random subset of the HIGGS dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS). 
This subset can be downloaded in HDF5 format from [here](https://www.dropbox.com/s/x7qdf9bmsvfezl9/HIGGSsubset.zip?dl=0).
HDF5 format is convenient for sequential tests since it allows constant time sampling.

## GAN generated vs real MNIST dataset

This dataset was generated using the pretrained model available at https://github.com/csinva/gan-pretrained-pytorch.
The dataset can be downloaded in HDF5 format from [here](https://www.dropbox.com/s/qsg0ujbph1d0ul3/GANMNIST.zip?dl=0)



## License
[MIT license](https://github.com/alherit/kd-switch/blob/master/LICENSE).

If you have questions or comments about anything regarding this work, please see the paper for contact information.


