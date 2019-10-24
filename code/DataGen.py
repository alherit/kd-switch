#parts of this code are taken from https://github.com/wittawatj/interpretable-test

#The MIT License (MIT)
#
#Copyright (c) 2015 Wittawat Jitkrittum
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
from past.utils import old_div
import matplotlib.pyplot as plt
import math

def tr_te_indices(n, tr_proportion, seed=9282 ):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=28):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind


class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array 
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        #if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n'%(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' %(np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def mean_std(self):
        """Compute the average standard deviation """

        #Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        mstd = old_div((stdx + stdy),2.0)
        return mstd
        #xy = self.stack_xy()
        #return np.mean(np.std(xy, 0)**2.0)**0.5
    
    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same 
        for both X, Y. 
        
        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = subsample_ind( self.X.shape[0], n, seed )
        ind_y = subsample_ind( self.Y.shape[0], n, seed )
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

    ### end TSTData class  

class SampleSource(with_metaclass(ABCMeta, object)):
    """A data source where it is possible to resample. Subclasses may prefix 
    class names with SS"""

    @abstractmethod
    def sample(self, n, seed):
        """Return a TSTData. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """Return the dimension of the problem"""
        raise NotImplementedError()

    def visualize(self, n=400):
        """Visualize the data, assuming 2d. If not possible graphically,
        subclasses should print to the console instead."""
        data = self.sample(n, seed=1)
        x, y = data.xy()
        d = x.shape[1]

        if d==2:
            plt.plot(x[:, 0], x[:, 1], '.r', label='X')
            plt.plot(y[:, 0], y[:, 1], '.b', label='Y')
            plt.legend(loc='best')
        else:
            # not 2d. Print stats to the console.
            print(data)



class mySSGaussVarDiff(SampleSource):
    """Toy dataset two in Chwialkovski et al., 2015. 
    P = N(0, I), Q = N(0, diag((2, 1, 1, ...))). Only the variances of the first 
    dimension differ."""

    def __init__(self, d, var_d1=2.0):
        """
        d: dimension of the data 
        var_d1: variance of the first dimension. 2 by default.
        """
        self.d = d
        self.var_d1 = var_d1
        self.rstate = None

    def dim(self):
        return self.d

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)
            
        d = self.d
        var_d1 = self.var_d1
        std_y = np.diag(np.hstack((np.sqrt(var_d1), np.ones(d-1) )))
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d).dot(std_y)

        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state
            
        return TSTData(X, Y, label='gvd')


class mySSBlobs(SampleSource):
    """Mixture of 2d Gaussians arranged in a 2d grid. This dataset is used 
    in Chwialkovski et al., 2015 as well as Gretton et al., 2012. 
    Part of the code taken from Dino Sejdinovic and Kacper Chwialkovski's code."""

    def __init__(self, blob_distance=5, num_blobs=4, stretch=2, angle=old_div(math.pi,4.0)):
        self.blob_distance = blob_distance
        self.num_blobs = num_blobs
        self.stretch = stretch
        self.angle = angle

    def dim(self):
        return 2

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)


        x = gen_blobs(stretch=1, angle=0, blob_distance=self.blob_distance,
                num_blobs=self.num_blobs, num_samples=n)

        y = gen_blobs(stretch=self.stretch, angle=self.angle,
                blob_distance=self.blob_distance, num_blobs=self.num_blobs,
                num_samples=n)

        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state
        
        return TSTData(x, y, label='blobs')

def gen_blobs(stretch, angle, blob_distance, num_blobs, num_samples):
    """Generate 2d blobs dataset """

    # rotation matrix
    r = np.array( [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]] )
    eigenvalues = np.diag(np.array([np.sqrt(stretch), 1]))
    mod_matix = np.dot(r, eigenvalues)
    mean = old_div(float(blob_distance * (num_blobs-1)), 2)
    mu = np.random.randint(0, num_blobs,(num_samples, 2))*blob_distance - mean
    return np.random.randn(num_samples,2).dot(mod_matix) + mu



class SSTwoSpirals(SampleSource):
    """Mixture of two spirals in 2D."""

    def __init__(self, noise=.5):
        self.noise = noise

    def dim(self):
        return 2

    def twospirals(self,n_points):
        """
         Returns the two spirals dataset.
        """
        n = np.sqrt(np.random.rand(n_points,1)) * 6000 * (2*np.pi)/360
        d1x = -np.cos(n)*n + np.random.rand(n_points,1) * self.noise
        d1y = np.sin(n)*n + np.random.rand(n_points,1) * self.noise
        return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
                np.hstack((np.zeros(n_points),np.ones(n_points))))


    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)


        coords,labels = self.twospirals(n_points=2*n)

        x = coords[labels==0]
        y = coords[labels==1]

        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state

        return TSTData(x, y, label='twospirals_s%d'%seed)

class mySSSameGauss(SampleSource):
    """Two same standard Gaussians for P, Q. The null hypothesis 
    H0: P=Q is true."""
    def __init__(self, d):
        """
        d: dimension of the data 
        """
        self.d = d

    def dim(self):
        return self.d

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)


        d = self.d
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d) 

        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state

        return TSTData(X, Y, label='sg_d%d'%self.d)

class mySSGaussMeanDiff(SampleSource):
    """Toy dataset one in Chwialkovski et al., 2015. 
    P = N(0, I), Q = N( (my,0,0, 000), I). Only the first dimension of the means 
    differ."""
    def __init__(self, d, my=1.0):
        """
        d: dimension of the data 
        """
        self.d = d
        self.my = my

    def dim(self):
        return self.d

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)


        d = self.d
        mean_y = np.hstack((self.my, np.zeros(d-1) ))
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d) + mean_y
        
        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state

        return TSTData(X, Y, label='gmd_d%d'%self.d)
