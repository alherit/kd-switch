# -*- coding: utf-8 -*-

import math
from sklearn.metrics import pairwise
import numpy as np
from Utils import LogWeightProb as lp


from lxml import etree


class SeqKnnBmaSw(object):
    def __init__(self, alpha_label, theta0=None, powers = [.3,.5,.7,.9], 
                 lmbda = .9999, switching=True):

        self.switching = switching
        
        if theta0 is None:
            theta0=[1./alpha_label]*alpha_label
            print("default dist:",theta0)

        self.alpha_label = alpha_label

        self.theta0 = theta0

        self.lmbda = lp.LogWeightProb(lmbda)

        self.powers = powers
        self.N = len(powers)
        #cumprob for each expert
        self.cumprob =  np.repeat( lp.LogWeightProb(1.), self.N)
    
        self.weights = np.repeat( lp.LogWeightProb(1./self.N), self.N)

        self.trainingPoints = []
        self.trainingLabels = []

        if switching:
            # initialize switching
            # no switch before time 1
            self.w_theta0 = lp.LogWeightProb(1 - self.mu(1))
            # switch before time 1
            self.w_thetaz = lp.LogWeightProb(self.mu(1))



    def putXML(self,xmlroot):
        params = etree.SubElement(xmlroot,'model',{"type":"knn"})
        etree.SubElement(params, 'powers').text = str(self.powers)
        etree.SubElement(params, 'lambda').text = str(self.lmbda)
        etree.SubElement(params, 'switching').text = str(self.switching)



    #switching prior
    #index i from 1
    def mu(self,i):
        return 1./(i+1)


    def kofn_pow(self,n,power):
        return int(math.ceil(math.pow(n,power)))

    def predict(self, point, label, update):
        '''
        Give prob of label given point, using KNNBMASW
        '''
        
        
        prob_mixture = lp.LogWeightProb(0.)
        n = len(self.trainingPoints)
        #MIX USING POSTERIOR
        for i in range(self.N):
            if n==0:
                k=0
            else:
                k = self.kofn_pow(n,self.powers[i])
            #print "n= ",n," k= ",k
            
            prob_knn = self.predictKNNlambda(point,label,k)
            
            if update:
                self.cumprob[i] *= prob_knn
            prob_mixture += prob_knn * self.weights[i]

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


        if self.switching:
            return switchProb
        else:
            return prob_mixture
    

    def predictKNNlambda(self,point,label,k):
        if k>0:
        
            dists = pairwise.pairwise_distances(X=[point],Y=self.trainingPoints)[0]
    
            #1 neighbor = taking the smallest distance = 0th element
            k-=1
            kthElem = np.partition(dists, kth=k )[k]
           
            cond = dists<=kthElem
            
            dist = np.histogram(np.array(self.trainingLabels)[cond],bins=self.alpha_label,range=[0,self.alpha_label])[0] 
            dist = dist/np.sum(dist,dtype=float)
            #print(dist)
            prob = dist[label]
            

            prob = lp.LogWeightProb(prob)


        else:
            prob = self.predictTheta0(label)
                
        ## lambda mix as in the paper: mix with uniform prob (could be theta0 as well)
        prob = self.lmbda*prob + (lp.LogWeightProb(1.)-self.lmbda)*lp.LogWeightProb(1./self.alpha_label)

            
        return prob
        
    def predictTheta0(self,label):
        return lp.LogWeightProb(self.theta0[label])


