
import scipy
from scipy import constants

from collections import namedtuple
LabeledPoint = namedtuple('LabeledPoint', 'point label')


# returns prob
def seqKT(counts,label,alpha=2):
    #print counts
    virt = 0.5
    total = sum(counts)
    c = counts[label]
    return (c + virt)/(total + alpha*virt)

# returns prob
def seqKTDist(counts, alpha=2):
    #print counts
    virt = 0.5
    total = sum(counts)
    return (counts + virt)/(total + alpha*virt)



_M_LNPI = scipy.log(scipy.constants.pi)
_M_LN2  = scipy.log(2.) 

# returns - log prob of the sequence represented by counts
def KT(counts,alpha = 2):
    alpha_2 = alpha / 2.0
    value = 0.
    total = 0.
    occuring = 0

    # Sum Log[ Gamma ( n_sa(x) + 1/2) ]
    for label in range(alpha):
        total += counts[label]
        if (counts[label] > 0):
            occuring += 1
            value -= scipy.special.gammaln(counts[label] + 0.5)

    # Add Log[ Gamma(1/2)] for non occuring symbols
    value -= (alpha-occuring) * 0.5 * _M_LNPI

    value -=  scipy.special.gammaln(alpha_2)

    value += scipy.special.gammaln (total + alpha_2) + alpha_2 * _M_LNPI

    return value / _M_LN2


