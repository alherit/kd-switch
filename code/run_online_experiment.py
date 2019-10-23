#!/usr/bin/python
from __future__ import print_function

import random
import sys
import math
import os

import h5py
import psutil
import numpy as np


from KDSwitch import SeqForest
from SeqKNN import SeqKnnBmaSw
from SeqGP import SeqGPBmaSw
from SeqGNB import SeqGNB
from SeqTrue import NClassGaussianMixture
from DataGen import mySSSameGauss,mySSBlobs,mySSGaussMeanDiff,mySSGaussVarDiff

from Utils import Commons as cm
from Utils import LogWeightProb as logPr

from lxml import etree

import pickle

from sklearn import datasets 

import time




class TST:

    
    def __init__(self, max_samples, max_time, alpha, dataset, seq_dataset_fname, fname0, fname1, key0, key1, theta0, maxTrials ,
                  maxMem, trial_n=None, saveData=False, probs_fname=None, alpha_size=None, data_gen_seed=None, stop_when_reject=False, 
                  synth_dim=None, mean_gmd=None, var_gvd=None , estimate_median_pdist=False):

        self.tstData = None

        self.stop_when_reject = stop_when_reject 

        if data_gen_seed == None:
            data_gen_seed = trial_n
            
        self.start_time = None

        self.alpha_size = alpha_size

        self.probs_file = open("./probs_seq_"+str(trial_n)+".dat","w")

        #max memory usage allowed
        self.maxMem = maxMem

        self.gadgetSampler = random.Random()
        self.gadgetSampler.seed(trial_n)

        self.dataSampler = random.Random()
        self.dataSampler.seed(data_gen_seed)


        #set global seed
        random.seed(trial_n)
        np.random.seed(trial_n)

        #self.min_samples = min_samples
        self.max_samples = max_samples
        self.max_time = max_time

        self.alpha = alpha

        self.maxTrials = maxTrials

        self.theta0 = theta0
        self.cumTheta0 = np.cumsum(self.theta0)



        #create xml output file
        self.XMLroot = etree.Element('experiment')
        params = etree.SubElement(self.XMLroot,'parameters')
        etree.SubElement(params, 'maxMem').text = str(self.maxMem)
        etree.SubElement(params, 'max_time').text = str(self.max_time)
        etree.SubElement(params, 'max_samples').text = str(self.max_samples)
        etree.SubElement(params, 'alpha').text = str(self.alpha)
        etree.SubElement(params, 'maxTrials').text = str(self.maxTrials)
        etree.SubElement(params, 'theta0').text = str(self.theta0)
        etree.SubElement(params, 'gadget_sampler_seed').text = str(trial_n)
        etree.SubElement(params, 'global_seed').text = str(trial_n)


        xml_data = etree.SubElement(self.XMLroot,'data')

        self.synthGen = None
        self.seqIndex = None
        
        self.unlimitedData = False
        
        if dataset is None and seq_dataset_fname is not None:
            data = pickle.load( open( seq_dataset_fname , "rb" ) )
            self.seqFeatures = data["features"]
            self.seqLabels = data["labels"]
            self.seqIndex = 0
            
            etree.SubElement(xml_data, 'seq_dataset_fname').text = seq_dataset_fname

            self.dim =  self.seqFeatures.shape[1]
            etree.SubElement(xml_data, 'dim').text = str(self.dim)

            
            
        elif dataset is None and fname0 is not None and fname1 is not None :
            self.fname0 = fname0
            self.fname1 = fname1


            releaseFiles = True #load datasets into memory and release files
            if releaseFiles:
                self.hf0 = h5py.File(self.fname0,'r')
                self.hf1 = h5py.File(self.fname1,'r')
    
                #use the first dataset of each file  ## [()] to get everything and put it in memory as numpy array (faster)
                self.datasets = {0: self.hf0[key0][()] ,
                                 1: self.hf1[key1][()] }
                self.hf0.close()
                self.hf1.close()
            else:
                self.hf0 = h5py.File(self.fname0,'r')
                self.hf1 = h5py.File(self.fname1,'r')
    
                #use the first dataset of each file  
                self.datasets = {0: self.hf0[key0] ,
                                 1: self.hf1[key1] }
                

            etree.SubElement(xml_data, 'fname0').text = str(self.fname0)
            etree.SubElement(xml_data, 'fname1').text = str(self.fname1)
            etree.SubElement(xml_data, 'seed').text = str(data_gen_seed)

            self.dim =  self.datasets[0].shape[1]
            etree.SubElement(xml_data, 'dim').text = str(self.dim)



        elif dataset.startswith("sklearn_"):
            from sklearn.utils import shuffle
            X,y = getattr(datasets, 'load'+dataset[7:])(return_X_y=True)
            X,y = shuffle(X,y, random_state=data_gen_seed)

            self.seqFeatures = X
            self.seqLabels = y
            self.seqIndex = 0
            self.dim =  self.seqFeatures.shape[1]



            etree.SubElement(xml_data, 'sklearn').text = dataset[8:]

        elif dataset=="gmm":

            self.dim = synth_dim
            
             
            
            self.synthGen =  NClassGaussianMixture(dim=synth_dim, seed=data_gen_seed)


            self.unlimitedData = True
            self.X = None 
            self.Y = None 
            self.xi = None 
            self.yi = None 
            

            etree.SubElement(xml_data, 'dim').text = str(self.dim)

            

        elif dataset=="blobs":
            num_blobs = 4
            distance = 5
            stretch = 2
            angle = math.pi / 4.0

            self.synthGen = mySSBlobs(blob_distance=distance, num_blobs=num_blobs, stretch=stretch, angle=angle)
            
            self.sample_as_gretton(data_gen_seed)

            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume

            if saveData:
                if not os.path.exists("X"):
                    os.makedirs("X")
                if not os.path.exists("Y"):
                    os.makedirs("Y")
                  
                n=self.max_samples/2

                np.savetxt("X/blobs_X"+str(data_gen_seed)+".dat",self.X,header=str(2)+" "+str(n), comments='')
                np.savetxt("Y/blobs_Y"+str(data_gen_seed)+".dat",self.X,header=str(2)+" "+str(n), comments='')


            etree.SubElement(xml_data, 'seed').text = str(data_gen_seed)
            etree.SubElement(xml_data, 'num_blobs').text = str(num_blobs)
            etree.SubElement(xml_data, 'distance').text = str(distance)
            etree.SubElement(xml_data, 'stretch').text = str(stretch)
            etree.SubElement(xml_data, 'angle').text = str(angle)
            self.dim = 2
            etree.SubElement(xml_data, 'dim').text = str(self.dim)

        elif dataset=="gmd":
            if synth_dim is  None:
                self.dim = 100
            else:
                self.dim = synth_dim
                
            meanshift = mean_gmd
            
            self.synthGen = mySSGaussMeanDiff(d=self.dim, my=meanshift)
            
            self.sample_as_gretton(data_gen_seed)

            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume
            

            etree.SubElement(xml_data, 'meanshift').text = str(meanshift)
            etree.SubElement(xml_data, 'dim').text = str(self.dim)


        elif dataset=="gvd":
            if synth_dim is  None:
                self.dim = 50
            else:
                self.dim = synth_dim
            
            var_d1 = var_gvd
            
            self.synthGen = mySSGaussVarDiff(d=self.dim, var_d1=var_d1)
            

            self.sample_as_gretton(data_gen_seed)
            
            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume
            

            etree.SubElement(xml_data, 'vardiff').text = str(var_d1)
            etree.SubElement(xml_data, 'dim').text = str(self.dim)

        elif dataset=="sg":
            if synth_dim is  None:
                self.dim = 50
            else:
                self.dim = synth_dim
            
            self.synthGen = mySSSameGauss(d=self.dim)
            
            self.sample_as_gretton(data_gen_seed)
            
            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume


            etree.SubElement(xml_data, 'same_gaussian').text = "default_value"
            etree.SubElement(xml_data, 'dim').text = str(self.dim)



        else:
            exit("Wrong dataset")



        #sets of row indexes already sampled
        #this is used to ensure sampling without replacement
        self.sampled = {}
        for i in range(self.alpha_size):
            self.sampled[i] = set()

        self.processed = list()
 

        self.model = None # will be set later
        
        #here will be stored the p-value resulting from the two-sample test
        self.pvalue = logPr.LogWeightProb(1.)

    def sample_as_gretton(self,data_gen_seed):
        n=round(self.max_samples/2)
        self.tstData = self.synthGen.sample(n=n,seed=data_gen_seed)
        (self.X,self.Y) = self.tstData.xy()
        #then generate an extra 20% since the datasets are consumed randomly
        (X,Y) = self.synthGen.sample(n=round(n*.2)).xy()
        self.X = np.concatenate([self.X,X])
        self.Y = np.concatenate([self.Y,Y])
        
        

        
    def set_start_time(self):
        self.start_time = time.process_time()

    def __del__(self):
        self.probs_file.close()
        
    def setModel(self,model):
        self.model = model
        model.putXML(self.XMLroot)
        


    def writeXML(self,xmldir,index):
        et = etree.ElementTree(self.XMLroot)
        et.write("%s/res_seq_%s.xml" % (xmldir,index), pretty_print=True)        

    def sampleCat(self):
        u = self.gadgetSampler.uniform(0,1)
        return np.searchsorted(self.cumTheta0, u) 



    def getSample(self,pop=None):
        if pop is None:
            pop =  self.sampleCat()
            

        if self.synthGen is not None:
            if self.unlimitedData:
                point = self.synthGen.get_sample(pop)
            
            else:
                if pop==0:
                    if self.xi < len(self.X):
                        point = self.X[self.xi]
                        self.xi+=1
                    else:
                        return None
                else:
                    if self.yi < len(self.Y):
                        point = self.Y[self.yi]
                        self.yi+=1
                    else:
                        return None

        else:
            #let's try maxTrials times
            for i in range(self.maxTrials):
                row = self.dataSampler.randrange(0,self.datasets[pop].shape[0]-1)
                if not row in self.sampled[pop]:
                    break
                else:
                    row = -1

            if row==-1:
                print('Tried %i times. All rejected. ' % self.maxTrials)
                return None

            point = self.datasets[pop][row,:]

            self.sampled[pop].add(row)


        return cm.LabeledPoint(point,pop)

    def predictTheta0(self,label):
        if self.theta0 is not None:
            return logPr.LogWeightProb(self.theta0[label])
        else:
            return logPr.LogWeightProb(0.)



    def tst(self):

#        #cumulate in log form
        cumCProb = logPr.LogWeightProb(1.)
        cumTheta0Prob = logPr.LogWeightProb(1.)
        self.pvalue = logPr.LogWeightProb(1.)

        # convert to log form
        alpha = logPr.LogWeightProb(self.alpha)


        i = 1

        reject = False
 

        while (self.max_time is None or time.process_time()-self.start_time <= self.max_time) and (self.max_samples is None or i <= self.max_samples):
            
            if psutil.virtual_memory().percent > self.maxMem:
                print('Not enough memory to proceed. Used percentage %s. ' % psutil.phymem_usage().percent)
                if not reject:
                    print("No difference detected so far.")
                break

            if self.max_samples is not None:
                if i > self.max_samples:
                    print('Max samples %s reached. ' % self.max_samples)
                    if not reject:
                        print("No difference detected so far.")
                    break
            
            if self.seqIndex is not None:
                if self.seqIndex<self.seqFeatures.shape[0]:
                    lp = cm.LabeledPoint(self.seqFeatures[self.seqIndex],self.seqLabels[self.seqIndex])
                    self.seqIndex+=1
                else:
                    lp = None
            else:
                lp = self.getSample()

            if lp is None:
                print('No more samples.')
                if not reject:
                    print("No difference detected so far.")
                break

            #print("label: ",lp.label, " point: ", lp.point)

            condProb = self.model.predict(lp.point,lp.label,update=True)
            theta0Prob = self.predictTheta0(lp.label)

            cumCProb *= condProb
            cumTheta0Prob *= theta0Prob


            self.pvalue =  cumTheta0Prob/cumCProb #min(1.0,math.pow(2,log_theta0Prob-log_CTWProb))
            n = len(self.processed)+1
            if n%10 ==0 :
                print ('n=',n,'p-value', self.pvalue, 'alpha', self.alpha)

            nll = -cumCProb.getLogWeightProb()/n
            self.probs_file.write(str(time.process_time()-self.start_time)+" "+str(condProb)+" "+str(nll)+ "\n")
            #str(lp.label)+" "+

            self.processed.append(lp)

            i += 1
            
            if not reject and self.pvalue <= alpha :
                reject = True
                n_reject = n
                p_value_reject = self.pvalue
                
                if self.stop_when_reject and reject:
                    break
        
            
        if not reject:
            n_ = n
            p_value_ = self.pvalue
        else:
            n_ = n_reject
            p_value_ = p_value_reject
        
        print ('n=',n_,'p-value', p_value_, 'alpha', self.alpha)
    
        print ('n=',n,'norm_log_loss', nll)
    
        result = etree.SubElement(self.XMLroot,'result')
        etree.SubElement(result,'stopping_time').text = str(n_)
        etree.SubElement(result,'reject').text = str(reject)

        etree.SubElement(result,'final_norm_log2_loss').text = str(nll)
        etree.SubElement(result,'final_n').text = str(n)





if __name__ == "__main__":
    from optparse import OptionParser

    def foo_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, [float(p) for p in value.split(',')])
            

    parser = OptionParser()
    parser.add_option('-D','--dataset', dest='dataset', type='string',
                      help='dataset: blobs, meanshift [default: use -f -g]', default=None)

    parser.add_option('-s','--seq-dataset-fname', dest='seq_dataset_fname', type='string',
                      help='pickle file with {feat,labels} [default: %default]', default=None)




    parser.add_option('-f', '--hdf-fileX', dest='hdfFnameX', help='hdf file X', default=None)
    parser.add_option('-g', '--hdf-fileY', dest='hdfFnameY', help='hdf file Y', default=None)
    parser.add_option('-F', '--pathX', dest='pathX', help='path to table in hdf file X [default: %default]', default='data')
    parser.add_option('-G', '--pathY', dest='pathY', help='path to table in hdf file Y [default: %default]', default='data')


    parser.add_option('-S', '--max-samples', dest='max_samples', type='int',
                      help='max number of samples to process [default: %default]', default=None)#

    parser.add_option('-T', '--max-time', dest='max_time', type='float',
                      help='max process time [default: %default]', default=None)#


    parser.add_option('-X', '--max-memory', dest='max_memory_usage', type='float',
                      help='max memory usage in percentage [default: %default]', default=900)

    parser.add_option('-a', '--alpha', dest='alpha', type='float',
                  help='alpha threshold for rejecting [default: %default]', default=0.05)

    parser.add_option('-t', '--theta0', dest='theta0', type='string',
                      help='theta0 for the random sampler all probs except last one [default: %default]', default=None, action='callback', callback=foo_callback)
    

    parser.add_option('-m', '--maxTrials', dest='maxTrials', type='int',
                      help='maxTrials for rejection sampling [default: %default]', default=100)

    parser.add_option('-N', '--num-trees', dest='nTrees', type='int',
                      help='number of trees [default: %default]', default=1)


    parser.add_option('-M','--model', dest='model', type='string',
                      help='seq model to be used, options are: kds(default), knn', default="kds")

    parser.add_option('-R', '--max-rot_dim', dest='max_rot_dim', type='int',
                      help='max dimension for random rotations [default: %default]', default=0)#


    parser.add_option('-I','--trials', dest='trials', type='int',
                      help='trials for power estimation [default: %default]', default=1)

    parser.add_option('-r','--trials-from', dest='trials_from', type='int',
                      help='number trials from [default: %default]', default=0)

    parser.add_option('-w','--use-weighting', action="store_true", dest='ctw', 
                      help='use weighting instead of switching [default: %default]', default=False)
    

    parser.add_option('-e','--fixed-data-seed', action="store_true", dest='fixed_data_seed', 
                      help='fixed data seed [default: %default]', default=False)

    parser.add_option('-b','--stop-when-reject', action="store_true", dest='stop_when_reject', 
                      help='stop when reject [default: %default]', default=False)

    parser.add_option( '--synth-dim', dest='synth_dim', type='int',
                      help='dimension for synthetic data (gmd,gvd,sg) [default: %default]', default=None)

    parser.add_option( '--mean-gmd', dest='mean_gmd', type='float',
                      help='mean for gmd [default: %default]', default=1.)

    parser.add_option( '--var-gvd', dest='var_gvd', type='float',
                      help='var for gvd [default: %default]', default=2.)

    parser.add_option( '--alpha-size', dest='alpha_size', type='int',
                      help='alphabet size when theta0 not specified [default: %default]', default=None)

    parser.add_option('--global-rot', action="store_false", dest='local_rot', 
                      help='only one random rotation for all trees[default: %default]', default=True)


    (options, args) = parser.parse_args()


    saveData = False    
    
    
    #add last probability
    if options.theta0 is not None:
        options.theta0.append(1.-sum(options.theta0)) 
        print("theta0:",options.theta0)
        alpha_size = len(options.theta0) 
    else:
        alpha_size = options.alpha_size
    
    for i in range(options.trials):
        trial_n = i+options.trials_from
        
        if options.fixed_data_seed:
            data_gen_seed = 0
        else:
            data_gen_seed = trial_n
            
        print("trial ",trial_n)


        tst = TST(options.max_samples,options.max_time,options.alpha, options.dataset, options.seq_dataset_fname, options.hdfFnameX, options.hdfFnameY,
                          options.pathX, options.pathY, options.theta0, options.maxTrials, 
                          options.max_memory_usage, trial_n=trial_n , saveData=saveData, alpha_size=alpha_size, 
                          data_gen_seed = data_gen_seed, stop_when_reject=options.stop_when_reject, synth_dim=options.synth_dim,
                          mean_gmd=options.mean_gmd, var_gvd=options.var_gvd, estimate_median_pdist=(options.model=="gp"))



        if options.model == "kds":
            tst.setModel ( SeqForest(J=options.nTrees, dim=tst.dim, alpha_label=alpha_size, 
                               theta0=options.theta0, keepItems = False, max_rot_dim = options.max_rot_dim, ctw=options.ctw, global_rot= not options.local_rot))
        elif options.model == "knn":
            tst.setModel ( SeqKnnBmaSw(alpha_label=alpha_size, theta0=options.theta0) )
        elif options.model == "gp":
            tst.setModel ( SeqGPBmaSw(alpha_label=alpha_size, theta0=options.theta0,  switching=False) )
        elif options.model == "gnb":
            tst.setModel ( SeqGNB(alpha_label=alpha_size, theta0=options.theta0) )
                
        elif options.model == "true": 
            tst.setModel ( tst.synthGen) 
            
        else:
            sys.exit("wrong model")            
        

        
        

        if not saveData:
            tst.tst()
            tst.writeXML(".",trial_n)

    
