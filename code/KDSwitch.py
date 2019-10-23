
from Utils import Commons as cm
from Utils import LogWeightProb as lp


import numpy as np

from lxml import etree

from scipy.stats import special_ortho_group

## Bayesian forest
## each tree uses a random rotation
class SeqForest(object):
    def __init__(self, J, dim, alpha_label, theta0=None, keepItems=False, max_rot_dim = None, local_alpha= True, ctw=False, global_rot=True):

        self.max_rot_dim = max_rot_dim
                
        self.local_alpha = local_alpha
        self.J = J
        self.trees = []
        self.ctw =ctw

        for i in range(J):
            self.trees.append( SeqTree(dim=dim, alpha_label=alpha_label, theta0=theta0,
                                       keepItems=keepItems, max_rot_dim = max_rot_dim, local_alpha=local_alpha, ctw=ctw)
            )
    
            # initialize with uniform weights
            self.weights = np.repeat( lp.LogWeightProb(1./J), J)
            


    def predict(self, point, label, update=True):
        '''
        Give prob of label given point, using CT*
        '''
        
        assert update # mandatory for kdSwitch
                
        prob_trees_mixture = lp.LogWeightProb(0.)
        #MIX USING POSTERIOR
        for i in range(self.J):
            prob_tree = self.trees[i].predictUpdateKDSwitch(point,label) 
            prob_trees_mixture += prob_tree * self.weights[i]

        acc = lp.LogWeightProb(0.)
        for i in range(self.J):
            self.weights[i] = self.trees[i].root.CTprob * lp.LogWeightProb(1./self.J)
            acc += self.weights[i]

        for i in range(self.J):
            self.weights[i] /= acc

        return prob_trees_mixture

    def putXML(self,xmlroot):
        params = etree.SubElement(xmlroot,'model',{"type":"kdSwitch"})
        etree.SubElement(params, 'J').text = str(self.J)
        etree.SubElement(params, 'max_rot_dim').text = str(self.max_rot_dim)
        etree.SubElement(params, 'local_alpha').text = str(self.local_alpha)
        etree.SubElement(params, 'ctw').text = str(self.ctw)


class SeqTree(object):

    def __init__(self, dim, alpha_label, theta0 = None, local_alpha=True,
                 keepItems = False, max_rot_dim = None, ctw=False):

            self.ctw = ctw
        
            self.max_rot_dim = max_rot_dim
            self.local_alpha = local_alpha

            if max_rot_dim > 0:
                #random rotation matrix
                print("generating random rotation matrix...")
                if dim > max_rot_dim:
                    print("using "+ str(max_rot_dim) + "random axes")
                    rot_axes = np.random.choice(range(dim),max_rot_dim,replace=False)
                    self.rot_mask = [True if i in rot_axes else False for i in range(dim)]  
                    rot_dim = max_rot_dim
                else:
                    self.rot_mask = None
                    rot_dim = dim
    
                self.R =  special_ortho_group.rvs(rot_dim)
            else:
                self.R = None
                print("No rotation.")


            self.theta0 = theta0

            self.dim = dim
            self.alpha_label =  alpha_label
            
            #set this value to avoid learning global proportion
            if self.theta0 is not None:
                self.P0Dist = [lp.LogWeightProb(p) for p in theta0]
            else:
                self.P0Dist = None

            self.n = 0

            #keep items in each node for post-hoc analysis: high memory consumption
            self.keepItems = keepItems

            self.root = SeqNode(depth=0,tree=self)
            



    def alpha(self,n):
        if self.ctw: #never switches
            return lp.LogWeightProb(0.) 
        else:
            return lp.LogWeightProb(1.)/lp.LogWeightProb(n)

    def predictUpdateKDSwitch(self, point, label):
        
        #rotate
        if self.R is not None:
            if self.rot_mask is not None:
                rotated = np.dot(self.R, point[self.rot_mask])
                point[self.rot_mask] = rotated
            else:
                point = np.dot(self.R, point)
                
            
        self.n += 1
        return self.root.predictUpdateKDSwitch(point,label,updateStructure=True)


class SeqNode(object):
    
    def __init__(self, depth=0, tree=None ):

            self.depth = depth
            self.tree = tree

            self.projDir = np.random.randint(0,self.tree.dim) 

            self.items = [] #samples observed in this node. if not self.tree.keep_items, they are moved to children node when created
            
            self.Children = None #references to children of this node

            self.pivot = None #splitting point

            self.counts = [0 for x in range(self.tree.alpha_label)]

            # CTProb is the prob on whole seq giving by doing cts on this node (=mixing with children)
            self.CTprob = lp.LogWeightProb(1.)

            ## for CTS
            self.wa = lp.LogWeightProb(.5)
            self.wb = lp.LogWeightProb(.5)

 
        



    ## main difference with original version: tree can be indefinitely refined: thus at current leaves, we must assume implicit children predicting as leaf (kt)    
    ## in the original version nodes can be adde, but leaf at depth D will remain leaf (this is why k and s don't matter at this node)
    def predictUpdateKDSwitch(self,point,label, updateStructure=True):

        splitOccurredHere = False
        
        #STEP 1: UPDATE STRUCTURE if leaf (full-fledged kd tree algorithm)
        if updateStructure and self.Children is None:

            splitOccurredHere = True
            
            self.pivot = self.computeProj(point)
            
            self.Children = [
                    self.__class__(depth=self.depth+1, tree=self.tree),
                    self.__class__(depth=self.depth+1, tree=self.tree)
                    ]
            
            for p in self.items: ### make all the update steps, notice that we are not yet adding current point
                i = self._selectBranch(p.point)
                self.Children[i].items.append(cm.LabeledPoint(point=p.point,label=p.label))
                self.Children[i].counts[p.label] += 1


            i = self._selectBranch(point)
            self.Children[i].items.append(cm.LabeledPoint(point=point,label=label)) #add current point to children's collection but not to label count, it will be done after prediction


            #initialize children using already existing symbols
            for i in [0,1]:
                self.Children[i].CTprob = lp.LogWeightProb(log_wp=-cm.KT(self.Children[i].counts,alpha = self.tree.alpha_label))                  
                self.Children[i].wa *= self.Children[i].CTprob
                self.Children[i].wb *= self.Children[i].CTprob


            if not self.tree.keepItems:
                self.items = []  #all items have been sent down to children, so now should be empty
            
            
        #STEP 2 and 3: PREDICT AND UPDATE
        #save CTS_prob before update (lt_n = <n , cts prob up to previous symbol)        
        prob_CTS_lt_n = self.CTprob
        
        if self.depth ==0 and self.tree.P0Dist is not None: #known distribution
            prob_KT_next = self.tree.P0Dist[label]
        else:
            prob_KT_next = lp.LogWeightProb(cm.seqKT(self.counts,label,alpha = self.tree.alpha_label)) #labels in self.items are not used, just counts are used 


        #now we can UPDATE the label count       
        self.counts[label] += 1


        if self.Children is None: 
            #PREDICT
            self.CTprob*=prob_KT_next #just KT

            #UPDATE
            self.wa*= prob_KT_next
            self.wb*= prob_KT_next
        else:
            #PREDICT (and recursive UPDATE)
            i = self._selectBranch(point)
            pr = self.Children[i].predictUpdateKDSwitch(point,label, updateStructure=not splitOccurredHere) 
            
            self.CTprob = self.wa * prob_KT_next + self.wb * pr

            #UPDATE
            if self.tree.local_alpha:
                alpha_n_plus_1 = self.tree.alpha(sum(self.counts)+1)
            else:
                alpha_n_plus_1 = self.tree.alpha(self.tree.n+1)

            self.wa = alpha_n_plus_1 * self.CTprob + (lp.LogWeightProb(1.)- lp.LogWeightProb(2.)*alpha_n_plus_1)* self.wa * prob_KT_next;
            self.wb = alpha_n_plus_1 * self.CTprob + (lp.LogWeightProb(1.)- lp.LogWeightProb(2.)*alpha_n_plus_1)* self.wb * pr;

       
        prob_CTS_up_to_n = self.CTprob
        
        return prob_CTS_up_to_n / prob_CTS_lt_n 
        
        
    def computeProj(self,point):
        if type(point)==dict: #sparse rep
            return point.get(self.projDir, 0.)
        else:
            return point[self.projDir]
    

    def _selectBranch(self, point):

        D = self.computeProj(point)

        if D <= self.pivot:
            return 0
        else:
            return 1
        
        
       


