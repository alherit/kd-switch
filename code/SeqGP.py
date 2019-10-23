# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:59:22 2019

@author: alheritier
"""


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


import random

import numpy as np
from Utils import LogWeightProb as lp
import sys

from lxml import etree




class SeqGP(object):
    def __init__(self, alpha_label, theta0=None, optimizer=None, scale=1.):

        self.scale = scale
        self.optimizer = optimizer
        
        if theta0 is None:
            theta0=[1./alpha_label]*alpha_label
            print("default dist:",theta0)

        if alpha_label != 2:
            sys.exit("alpha_label != 2: not implemented")

        self.theta0 = theta0

        self.trainingPoints = []
        self.trainingLabels = []

        self.gpc = None




    def putXML(self,xmlroot):
        params = etree.SubElement(xmlroot,'model',{"type":"gp"})
        etree.SubElement(params, 'optimizer').text = str(self.optimizer)
        etree.SubElement(params, 'scale').text = str(self.scale )
        etree.SubElement(params, 'theta0').text = str(self.theta0)



    def predict(self, point, label, update):
        '''
        Give prob of label given point, using KNNBMASW
        '''
        
        
            
        prob_gp = self.predictGP(point,label)
            
        if update:
            self.trainingPoints.append(point)
            self.trainingLabels.append(label)


            ## compute GP posterior for next time
            if self.gpc is not None or len(np.unique(self.trainingLabels))==len(self.theta0) :  ## fit needs all the classes 
                kernel = 1. * RBF(1.)
                 
                self.gpc = GaussianProcessClassifier(kernel=kernel, optimizer=self.optimizer, n_jobs=1).fit(self.trainingPoints, self.trainingLabels)


        return prob_gp
    

    def predictGP(self,point,label):
        if self.gpc is not None:
            probas =  self.gpc.predict_proba(point.reshape(1, -1))[0]
            prob = lp.LogWeightProb( probas[label] )
            
        else:
            prob = self.predictTheta0(label)
            
        return prob
    
    def predictTheta0(self,label):
        return lp.LogWeightProb(self.theta0[label])





class SeqGPBmaSw(object):
    def __init__(self, alpha_label, theta0=None, scales = np.logspace(-20,28,num=(28+20)/4+1, base=2), 
                  switching=True):

        self.switching = switching
        
        if theta0 is None:
            theta0=[1./alpha_label]*alpha_label
            print("default dist:",theta0)


        self.alpha_label = alpha_label

        self.theta0 = theta0


        self.scales = scales
        self.N = len(scales)
        #cumprob for each expert
        self.cumprob =  np.repeat( lp.LogWeightProb(1.), self.N)
    
        self.weights = np.repeat( lp.LogWeightProb(1./self.N), self.N)

        self.trainingPoints = []
        self.trainingLabels = []

        self.gpc = None

        if switching:
            # initialize switching
            # no switch before time 1
            self.w_theta0 = lp.LogWeightProb(1 - self.mu(1))
            # switch before time 1
            self.w_thetaz = lp.LogWeightProb(self.mu(1))



    def putXML(self,xmlroot):
        params = etree.SubElement(xmlroot,'model',{"type":"gp"})
        etree.SubElement(params, 'scales').text = str(self.scales)
        etree.SubElement(params, 'switching').text = str(self.switching)



    #switching prior
    #index i from 1
    def mu(self,i):
        return 1./(i+1)



    def predict(self, point, label, update):
        '''
        Give prob of label given point, using KNNBMASW
        '''
        
        
        prob_mixture = lp.LogWeightProb(0.)
        n = len(self.trainingPoints)
        #MIX USING POSTERIOR
        for i in range(self.N):
            
            prob_gp = self.predictGP(point,label,i)
            
            if update:
                self.cumprob[i] *= prob_gp
            prob_mixture += prob_gp * self.weights[i]

        if self.switching:
            theta0Prob = self.predictTheta0(label)
            switchProb = self.w_theta0*theta0Prob + self.w_thetaz*prob_mixture


        if update:
            self.trainingPoints.append(point)
            self.trainingLabels.append(label)

            acc = lp.LogWeightProb(0.)
            for i in range(self.N):
                self.weights[i] = self.cumprob[i] * lp.LogWeightProb(1./self.N)
                acc += self.weights[i]

            for i in range(self.N):
                self.weights[i] /= acc


            if self.switching:
                # update switching posteriors
                #we saw n points, we just predicted the n+1-th and, now we compute the weights for the n+2-th
                self.w_theta0 = self.w_theta0*theta0Prob* lp.LogWeightProb(1 - self.mu(n+2))
                self.w_thetaz = self.w_theta0*theta0Prob* lp.LogWeightProb(self.mu(n+2)) + self.w_thetaz * prob_mixture
                total = self.w_theta0 + self.w_thetaz
                self.w_theta0 /= total
                self.w_thetaz /= total

            ## compute GP posterior for next time
            if self.gpc is not None or len(np.unique(self.trainingLabels))==self.alpha_label :  ## fit needs all the classes 
                self.gpc = []
                for s in self.scales:
                    kernel = 1.0 * RBF(s)
                    self.gpc.append( GaussianProcessClassifier(kernel=kernel, optimizer=None, n_jobs=1).fit(self.trainingPoints, self.trainingLabels))



        if self.switching:
            return switchProb
        else:
            return prob_mixture
    

    def predictGP(self,point,label,scale_indx):
        if self.gpc is not None:
            probas =  self.gpc[scale_indx].predict_proba(point.reshape(1, -1))[0]
            prob = lp.LogWeightProb( probas[label] )
            
        else:
            prob = self.predictTheta0(label)


            
        return prob
    
    def predictTheta0(self,label):
        return lp.LogWeightProb(self.theta0[label])


