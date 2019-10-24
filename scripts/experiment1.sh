#!/bin/bash
DATAPATH=  ## add path to data
PYTHONPATH=$PYTHONPATH: ## add path to scripts and code folders

## BREASTCANCER

mkdir BREASTCANCER
cd BREASTCANCER

mkdir kds50
cd kds50
python run_online_experiment.py -M kds -N 50 -D sklearn_breast_cancer --alpha-size=2   --trials=30 --trials-from=0    
cd ..

mkdir kdw50
cd kdw50
python run_online_experiment.py -M kds -w -N 50 -D sklearn_breast_cancer --alpha-size=2   --trials=30 --trials-from=0      
cd ..

mkdir kds1
cd kds1
python run_online_experiment.py -M kds -N 1 -D sklearn_breast_cancer --alpha-size=2   --trials=30 --trials-from=0     
cd ..

mkdir kdw1
cd kdw1
python run_online_experiment.py -M kds -w -N 1 -D sklearn_breast_cancer --alpha-size=2   --trials=30 --trials-from=0      
cd ..

mkdir knn
cd knn
python run_online_experiment.py -M knn -D sklearn_breast_cancer --alpha-size=2   --trials=30 --trials-from=0      
cd ..

mkdir gp
cd gp
python run_online_experiment.py -M gp -D sklearn_breast_cancer --alpha-size=2   --trials=30 --trials-from=0      
cd ..


python plotCurve.py -d . -r 10 -n 30 

cd ..

## GANMNIST

mkdir GANMNIST
cd GANMNIST

mkdir kds50
cd kds50
python run_online_experiment.py -M kds -N 50 -f $DATAPATH/mnist_real.h5 -g $DATAPATH/mnist_fake.h5 -e  --trials=10 --trials-from=0   --max-time 1800 -t .5  
cd ..

mkdir kdw50
cd kdw50
python run_online_experiment.py -M kds -w -N 50 -f $DATAPATH/mnist_real.h5 -g $DATAPATH/mnist_fake.h5 -e  --trials=10 --trials-from=0    --max-time 1800  -t .5  
cd ..

mkdir kds1
cd kds1
python run_online_experiment.py -M kds -N 1 -f $DATAPATH/mnist_real.h5 -g $DATAPATH/mnist_fake.h5 -e  --trials=10 --trials-from=0   --max-time 1800  -t .5  
cd ..

mkdir kdw1
cd kdw1
python run_online_experiment.py -M kds -w -N 1 -f $DATAPATH/mnist_real.h5 -g $DATAPATH/mnist_fake.h5 -e  --trials=10 --trials-from=0    --max-time 1800 -t .5   
cd ..

mkdir knn
cd knn
python run_online_experiment.py -M knn -f $DATAPATH/mnist_real.h5 -g $DATAPATH/mnist_fake.h5 -e  --trials=10 --trials-from=0    --max-time 1800 -t .5   
cd ..

mkdir gp
cd gp
python run_online_experiment.py -M gp -f $DATAPATH/mnist_real.h5 -g $DATAPATH/mnist_fake.h5 -e  --trials=10 --trials-from=0    --max-time 1800 -t .5   
cd ..

python plotCurve.py -d . -r 100 -n 10 -g
python plotCurve.py -d . -r 10 -n 10 -c

cd ..

## GMM

mkdir GMM
cd GMM

mkdir kds50
cd kds50
python run_online_experiment.py -M kds -N 50 -D gmm --synth-dim 2 -e  --trials=10 --trials-from=0   --max-time 1800 -t .5  
cd ..

mkdir kdw50
cd kdw50
python run_online_experiment.py -M kds -w -N 50  -D gmm --synth-dim 2 -e  --trials=10 --trials-from=0    --max-time 1800  -t .5   
cd ..

mkdir kds1
cd kds1
python run_online_experiment.py -M kds -N 1 -D gmm --synth-dim 2 -e  --trials=10 --trials-from=0   --max-time 1800 -t .5  
cd ..

mkdir kdw1
cd kdw1
python run_online_experiment.py -M kds -w -N 1  -D gmm --synth-dim 2 -e  --trials=10 --trials-from=0    --max-time 1800  -t .5   
cd ..

mkdir knn
cd knn
python run_online_experiment.py -M knn  -D gmm --synth-dim 2 -e  --trials=10 --trials-from=0    --max-time 1800  -t .5   
cd ..

mkdir gp
cd gp
python run_online_experiment.py -M gp  -D gmm --synth-dim 2 -e  --trials=10 --trials-from=0    --max-time 1800  -t .5   
cd ..


mkdir true
cd true
python run_online_experiment.py -M true  -D gmm --synth-dim 2 -e  --trials=1 --trials-from=0    --max-samples  1091430  -t .5   
cd ..

python plotCurve.py -d . -r 100 -n 10  -g -T  1091430 --min-y .85 --max-y 1.025  
python plotCurve.py -d . -r 10 -n 10   -c 

cd ..

## HIGGS

mkdir HIGGS
cd HIGGS


mkdir kds50
cd kds50
python run_online_experiment.py -M kds -N 50 -f $DATAPATH/HIGGS.csv_Xsubset.h5 -g $DATAPATH/HIGGS.csv_Ysubset.h5 -e  --trials=10 --trials-from=0   --max-time 1800 -t .5  
cd ..

mkdir kdw50
cd kdw50
python run_online_experiment.py -M kds -w -N 50 -f $DATAPATH/HIGGS.csv_Xsubset.h5 -g $DATAPATH/HIGGS.csv_Ysubset.h5 -e  --trials=10 --trials-from=0    --max-time 1800  -t .5  
cd ..

mkdir kds1
cd kds1
python run_online_experiment.py -M kds -N 1 -f $DATAPATH/HIGGS.csv_Xsubset.h5 -g $DATAPATH/HIGGS.csv_Ysubset.h5 -e  --trials=10 --trials-from=0   --max-time 1800  -t .5  
cd ..

mkdir kdw1
cd kdw1
python run_online_experiment.py -M kds -w -N 1 -f $DATAPATH/HIGGS.csv_Xsubset.h5 -g $DATAPATH/HIGGS.csv_Ysubset.h5 -e  --trials=10 --trials-from=0    --max-time 1800 -t .5   
cd ..

mkdir knn
cd knn
python run_online_experiment.py -M knn -f $DATAPATH/HIGGS.csv_Xsubset.h5 -g $DATAPATH/HIGGS.csv_Ysubset.h5 -e  --trials=10 --trials-from=0    --max-time 1800 -t .5   
cd ..


mkdir gp
cd gp
python run_online_experiment.py -M gp -f $DATAPATH/HIGGS.csv_Xsubset.h5 -g $DATAPATH/HIGGS.csv_Ysubset.h5 -e  --trials=10 --trials-from=0    --max-time 1800 -t .5   
cd ..


python plotCurve.py -d . -r 100 -n 10 -g --min-y .995 --max-y 1.0075


