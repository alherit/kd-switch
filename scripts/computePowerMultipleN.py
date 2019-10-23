
from __future__ import print_function

import glob
from lxml import etree as ET

import os
import pickle

def computePower(base_dir,grettonScale=False):

    print(base_dir)
    
    ####TODO: make a dictionary to put results
    dic = dict()
    



    for ndir in os.listdir(base_dir):
        n = int(ndir[1:])
        if grettonScale:
            print("n=",int(n/4))
        else:
            print("n=",n)


        

        for dataset in os.listdir(os.path.join(base_dir,ndir)):

            if not dataset in dic:
                dic[dataset] = dict()
            
            for method in os.listdir(os.path.join(base_dir,ndir,dataset)):

                if not method in dic[dataset]:
                    dic[dataset][method] = dict()


                rejected = 0

                base_path = os.path.join(base_dir,ndir, dataset, method)
                print(base_path)

                path = os.path.join(base_path, '*.xml')
                files = glob.glob(path)
                alphas = []
                
                for f in files:    
                    tree = ET.parse(f)
                    root = tree.getroot()
                    stopping_time = int(root.findtext("result/stopping_time"))
                    alpha = float(root.findtext("parameters/alpha"))
                    alphas.append (alpha)
 
                    if method=="kds50" and not (tree.find("model").attrib["type"]=='kdSwitch' and root.findtext("model/J")=="50"  and root.findtext("model/ctw")=="False" and root.findtext("model/max_rot_dim")=="500"):
                        exit("wrong kds50 in"+str(f))
                    if method=="kds50_norot" and not (tree.find("model").attrib["type"]=='kdSwitch' and root.findtext("model/J")=="50" and root.findtext("model/ctw")=="False"  and root.findtext("model/max_rot_dim")=="0"):
                        exit("wrong kds50_norot"+str(f))
                    if method=="knn" and tree.find("model").attrib["type"]!='knn':
                        exit("wrong method knn")
                    if method=="gp" and tree.find("model").attrib["type"]!='gp':
                        exit("wrong method gp")
                        
                    if dataset=="GMD"  and float(root.findtext("data/meanshift"))!=1.:
                        exit("wrong dataset GMD")
                    if dataset=="GVD"  and float(root.findtext("data/vardiff"))!=2.:
                        exit("wrong dataset GVD")
                    if dataset=="BLOBS"  and int(root.findtext("data/num_blobs"))!=4:
                        exit("wrong dataset BLOBS")
                    if dataset=="SG"  and root.findtext("data/same_gaussian")!="default_value":
                        exit("wrong dataset BLOBS")

                    if alpha!= 0.01:
                        exit("wrong alpha")
                    
                    reject = root.findtext("result/reject")=='True'
                    if reject:
                        rejected += int(n>=stopping_time) #double check
                    elif stopping_time < n:
                        print("WARNING file " + f + ": didn't reject and didn't reach required time=> assuming no reject. Stopping time was: " + str(stopping_time), "n is", n)
            
            
#                if len(set(alphas)) <= 1:
#                    print("alpha: ",alphas[0])
#                else:
#                    exit("inconsistent alphas:" + str(alphas))

                ntrials = len(files)            
                if ntrials!=500:
                    print("wrong number of trials")
                
                
                rejected /= ntrials
                print(base_path,"rejection rate:", rejected)
                
                if grettonScale:
                    n_save = int(n/4)
                else:
                    n_save = n

                dic[dataset][method][n_save] = rejected
    
    with open('results.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=2)
    
    
    for data in dic.keys():
        print("-------------")
        print(data)
        for method in dic[data].keys():
            print(method)
            for key in sorted (dic[data][method].keys()):
                print(dic[data][method][key])
   
    


def foo_callback(option, opt, value, parser):
  setattr(parser.values, option.dest,  [int(v) for v in value.split(',')] )

if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-d','--dir', dest='base_dir', type='string', 
                      help='directory containing results [default: %default]', default=".")

    
    parser.add_option('-g','--grettonScale', help='multiply n by 4', dest = 'grettonScale', default = False, action = 'store_true')

    (options, args) = parser.parse_args()



    computePower(options.base_dir,options.grettonScale)
