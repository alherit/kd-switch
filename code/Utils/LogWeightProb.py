
import numpy as np

class LogWeightProb:
    def __init__(self, wp = None, log_wp = None):
        if wp is None:
            if log_wp is None:
                self.zero = True
                self.log_wp = 0
            else:
                self.zero = False
                self.log_wp = log_wp

        elif wp < 0:
            exit("Negative Prob")

        else:
            self.zero = (wp==0)
            if not self.zero:
                self.log_wp = np.log2(wp)
            else:
                self.log_wp = 0


    def getWeightProb(self):
        if self.zero:
            return 0
        else:
            return np.exp2(self.log_wp)


    def getLogWeightProb(self):
        if self.zero:
            return -float("Inf")
        else:
            return self.log_wp


    # given log(x) and log(y), compute log(x+y). uses the following identity:
    # log(x + y) = log(x) + log(1 + y/x) = log(x) + log(1+exp(log(y)-log(x)))
    def __add__(self, other):
        ret = LogWeightProb() #initialized 0
        if not (self.zero and other.zero):
            ret.zero = False
            if self.zero:
                ret.log_wp = other.log_wp
            elif other.zero:
                ret.log_wp = self.log_wp
            else:
                log_x = self.log_wp
                log_y = other.log_wp
                ## ensure log_y >= log_x, can save some expensive log/exp calls
                if log_x > log_y:
                    t = log_x
                    log_x = log_y
                    log_y = t

                ret.log_wp = log_y - log_x
                # only replace log(1+exp(log(y)-log(x))) with log(y)-log(x)
                # if the the difference is small enough to be meaningful
                if ret.log_wp < 100:
                    ret.log_wp= np.log2(1.0 + np.exp2(ret.log_wp))
                ret.log_wp += log_x

        return ret

    def __mul__(self, other):
        ret = LogWeightProb() # initialized 0
        if not (other.zero or self.zero):
            ret.zero = False
            ret.log_wp = self.log_wp + other.log_wp

        return ret

    def div(self,other):
        if other.zero:
            exit("operator/: division by 0")

        ret = LogWeightProb()
        if not self.zero:
            ret.zero = False
            ret.log_wp = self.log_wp - other.log_wp

        return ret


    def __div__(self, other):
        return self.div(other)
        
    #for python 3
    def __truediv__(self, other):
        return self.div(other)


    def __str__(self):
        return str(self.getWeightProb())

    def __eq__(self, other):
        return self.zero==other.zero and self.log_wp==other.log_wp
        

    def __ge__(self, other):
        if self.zero and other.zero:
            return True
        elif self.zero:
            return False
        elif other.zero:
            return True
        else:
            return self.log_wp >= other.log_wp

    def __lt__(self, other):
        return not self>=other

    def __gt__(self, other):
        if self.zero and other.zero:
            return False
        elif self.zero:
            return False
        elif other.zero:
            return True
        else:
            return self.log_wp > other.log_wp

    def __le__(self, other):
        return not self>other

    # given log(x) and log(y), compute log(x-y). uses the following identity:
    # log(x - y) = log(x) + log(1-exp(log(y)-log(x)))
    # simple version
    def __sub__(self, other):
        ret = LogWeightProb() # initialized 0

        if not(other.zero and self.zero):
            ret.zero = False

        if other.zero:
            ret.log_wp = self.log_wp

        elif self.zero or other.log_wp > self.log_wp:
            print (self.zero)
            print (other.zero)
            print (other.log_wp , self.log_wp)
            exit("operator-: Can't store negative numbers in log representation")

        elif other.log_wp == self.log_wp:
            ret.zero = True

        else:
            ret.log_wp= self.log_wp + np.log2(1. - np.exp2(other.log_wp - self.log_wp))

        return ret

