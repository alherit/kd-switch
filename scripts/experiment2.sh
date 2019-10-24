#!/bin/bash

PYTHONPATH=$PYTHONPATH: ## add path to scripts and code folders

## with rotations
KDS_OPT="kds -N 50 -R 500"
## without rotations (Appendix figure)
#KDS_OPT="kds -N 50"

KNN_OPT="knn"



for NSAMPLES in 4000 8000 12000 16000 20000
do
	mkdir n$NSAMPLES
	cd n$NSAMPLES
	
	for DATA in sg blobs gmd gvd
	do
		DATA_DIR=${DATA^^}
		mkdir $DATA_DIR #uppercase
		cd $DATA_DIR
		for METHOD in kds knn
		do
			mkdir $METHOD
			cd $METHOD
			
			if [ $METHOD == 'kds' ]
			then
				MET_OPT=$KDS_OPT
			else
				MET_OPT=$KNN_OPT
			fi

			python run_online_experiment.py -M $MET_OPT -D $DATA  -a 0.01 -b --trials=500 --trials-from=0   --max-samples $NSAMPLES  -t .5 

			cd ..
			
	
		done
		
		cd ..
	
	done
	
	cd ..

done


python computePowerMultipleN.py -g