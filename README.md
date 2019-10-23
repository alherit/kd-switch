# kd-switch


This repository contains a Python implementation of the kd-switch online predictor and the KDS-seq sequential two-sample test proposed in our paper:

    Low-Complexity Nonparametric Bayesian Online Prediction with Universal Guarantees
    Alix Lhéritier & Frédéric Cazals
    NeurIPS 2019

[ArXiv version](https://arxiv.org/abs/1901.07662)

## Dependencies

* pyhon 2.7
* numpy >= 1.13.0
* h5py
* lxml
* scikit-learn
* scipy
* pandas
* matplotlib
* psutil

In order to reproduce the experimental results the following library is required to generate the samples:

* pip install git+https://github.com/wittawatj/interpretable-test



## Reproducing experimental results

Each experiment is defined in its own shell file in the scripts folder. 
The scripts parallelize the execution of the different trials. 
The number of parallel executions needs to be adapted accordingly to the number of cores and memory available. The trial index determines the seed used for randomness.
For each trial, an xml result file is generated. 
The script computePower.py  allows to compute power estimates from these xml files.

In order to reproduce the results of each experiment, you should run, in a subfolder of the repository root, the following

```
experiment1.sh 
experiment2.sh
experiment3.sh
```

For the first experiment, the CASE variable must be set with one of following values: {sg,gmd,gvd,blobs}. 

Then, in the same folder where the xml files were produced, you should execute for each experiment, respectively

```
python ../../scripts/computePower.py -n 1000,2000,3000,4000,5000 -g
python ../../scripts/computePower.py -n 804
python ../../scripts/computePower.py -n 27424
```

Note that -g makes the script interpret n as in [Jitkrittum et al. 2016](https://papers.nips.cc/paper/6148-interpretable-distribution-features-with-maximum-testing-power).

## Higgs boson dataset

In our experiments, we used a random subset of the HIGGS dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS). 
This subset can be downloaded in HDF5 format from [here](https://www.dropbox.com/s/x7qdf9bmsvfezl9/HIGGSsubset.zip?dl=0).
HDF5 format is convenient for sequential tests since it allows constant time sampling.
These .h5 files need to be located in a folder called data at the root of the repository.  

## License
[MIT license](https://github.com/alherit/kd-switch/blob/master/LICENSE).

If you have questions or comments about anything regarding this work, please see the paper for contact information.


