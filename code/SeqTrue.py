# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:50:51 2019

@author: alheritier
"""

import numpy as np
from scipy.special import softmax
from lxml import etree
from Utils import LogWeightProb as lp

from pomegranate import MultivariateGaussianDistribution, UniformDistribution, DirichletDistribution, GeneralMixtureModel
from scipy.stats import invwishart



class SeqTrue():
    
    def __init__(self, dataset, d , theta0, mean_d1=0., var_d1=1.):
        
        ## weird logic, sorry
        if dataset=="gmd":
            self.mean_d1 = mean_d1
        else:
            self.mean_d1 = 0.
            
        if dataset=="gvd":
            self.var_d1 = var_d1
        else:
            self.var_d1 = 1.
        
        self.d = d        
        
        self.sigma = [np.ones(d), np.hstack((self.var_d1, np.ones(d-1) ))]
        self.sigmainv = [ np.diag(1./ s) for s in self.sigma ]
        self.logDetSigma = [np.log(np.prod(s)) for s in self.sigma]
        
        self.mean = [np.zeros(d), np.hstack((self.mean_d1, np.zeros(d-1) )) ]      
 
        self.lnTheta0 = np.log(theta0)
       
        

        
    def predict(self, point, label, update):
     
             
        logprobs = [ -.5 * (np.dot(np.dot( point-self.mean[c] , self.sigmainv[c] ) , point-self.mean[c] ) +\
                   self.logDetSigma[c]) + self.lnTheta0[c] for c  in [0,1] ]
        
        #print(logprobs)
        probs = softmax(logprobs)
        #print(probs)
        prob = lp.LogWeightProb( probs[label])
        return prob
        
        


    def putXML(self,xmlroot):
        params = etree.SubElement(xmlroot,'model',{"type":"true"})
        etree.SubElement(params, 'mean_d1').text = str(self.mean_d1)
        etree.SubElement(params, 'var_d1').text = str(self.var_d1)



class NClassGaussianMixture():
    
    def __init__(self, dim , seed=None): #
        
        K = 9
        theta0=[.5,.5]
        beta=np.ones(K)
        Psi = .1*np.diag(np.ones(dim))
        #mu0= np.zeros(dim) 
        #lambd=.1,
        nu=dim+2.
        
        
        rstate = np.random.get_state()
        np.random.seed(seed)

        
        unif_dist = UniformDistribution(0.,1.)
        
        self.theta0 = theta0        
        beta_dist = DirichletDistribution(beta)

        self.dim = Psi.shape[0]

        self.dists = [] 

        #same weights for both
        weights = beta_dist.sample()
            
        mus = []
        for i,_ in enumerate(theta0):
            
            
            #weights = beta_dist.sample()
            #print(weights)
            mix = []
            for j,_ in enumerate(weights):
                
                
                if j%3==0:
                    Sigma = invwishart.rvs(df=nu, scale=Psi)
                    
                elif j%3==1:
                    Sigma = invwishart.rvs(df=nu, scale=.01*Psi)
                else:
                    Sigma = invwishart.rvs(df=nu, scale=.0001*Psi)

                if i==0:
                    mu = unif_dist.sample(self.dim) 
                    #mu =MultivariateGaussianDistribution(mu0,Sigma/lambd).sample()
                    mus.append(mu)
                else:
                    mu = mus[j]
                
                mix.append( MultivariateGaussianDistribution(mu, Sigma) )
                
            model = GeneralMixtureModel(mix, weights=weights)
            self.dists.append(model)
            
            
        for d in self.dists:
            print(d)
        
        self.rstate = np.random.get_state()
        np.random.set_state(rstate)

                
        
    def get_sample(self,c,n=1):
        
        rstate = np.random.get_state()
        np.random.set_state(self.rstate)

        ret= self.dists[c].sample(n)
        
        self.rstate = np.random.get_state()
        np.random.set_state(rstate)

    
        return ret
        

        
    def predict(self, point, label, update=None):
        jointprobs = [lp.LogWeightProb(d.probability(np.array(point))[0]) * lp.LogWeightProb(prior) for prior,d  in zip(self.theta0,self.dists)]
        
        sum_jp = lp.LogWeightProb(0.)
        
        for p in jointprobs:
            sum_jp+= p
        
        return jointprobs[label]/sum_jp

    def putXML(self,xmlroot):
        params = etree.SubElement(xmlroot,'model',{"type":"true_gmm"})



