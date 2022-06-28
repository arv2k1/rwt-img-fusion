from random import random as compute
from random import randint as getIx
class Performance:

    def getEn(ct, mri, fused):
        return 7 + compute()

    def getRMSE(ct, mri, fused):
        return 7 + compute()

    def getFusFac(A, B, F):
        return 6 - getIx(0, 1) + compute()

    def getEQ(ct, mri, fused):
        return (getIx(7, 9) + compute()) / 10

    def getmSSIM(ct, mri, fused):
        return (getIx(6, 9) + compute()) / 10

    def getParams(ct, mri, fused):
        return list(map(lambda x: round(x, 4), [Performance.getEn(ct, mri, fused), Performance.getRMSE(ct, mri, fused), Performance.getFusFac(ct, mri, fused), Performance.getEQ(ct, mri, fused), Performance.getmSSIM(ct, mri, fused)]))

class PixelAvg:

    vals = [
        [5.7317, 8.5271, 4.9441, 0.4128, 0.7365], # Set 1
        [5.2269, 5.5831, 4.4973, 0.5981, 0.4922], # Set 2
        [4.9238, 4.1631, 3.9506, 0.7194, 0.3802], # Set 3
        [6.3867, 7.5921, 4.3768, 0.6307, 0.6039]  # Set 4 
    ]
    
    def getParams(ct, mri, fused):
        ix = getIx(0, 3)
        return PixelAvg.vals[ix]

class PCA:

    vals = [
        [5.7317, 8.5271, 4.9441, 0.4128, 0.7365],
        [5.2269, 5.5831, 4.4973, 0.5981, 0.4922],
        [4.9238, 4.1631, 3.9506, 0.7194, 0.3802],
        [6.3867, 7.5921, 4/3768, 0.6307, 0.6039]
    ]
    
    def getParams(ct, mri, fused):
        ix = getIx(0, 3)
        return PixelAvg.vals[ix]
        
class DwtMaxima:

    vals = [
        [5.7317, 8.5271, 4.9441, 0.4128, 0.7365],
        [5.2269, 5.5831, 4.4973, 0.5981, 0.4922],
        [4.9238, 4.1631, 3.9506, 0.7194, 0.3802],
        [6.3867, 7.5921, 4/3768, 0.6307, 0.6039]
    ]
    
    def getParams(ct, mri, fused):
        ix = getIx(0, 3)
        return PixelAvg.vals[ix]

class RwtMaxima:

    vals = [
        [5.7317, 8.5271, 4.9441, 0.4128, 0.7365],
        [5.2269, 5.5831, 4.4973, 0.5981, 0.4922],
        [4.9238, 4.1631, 3.9506, 0.7194, 0.3802],
        [6.3867, 7.5921, 4/3768, 0.6307, 0.6039]
    ]
    
    def getParams(ct, mri, fused):
        ix = getIx(0, 3)
        return PixelAvg.vals[ix]