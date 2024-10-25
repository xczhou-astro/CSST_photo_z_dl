import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import bisect
import os

bin_size = 0.1
CI = np.arange(0, 1.0 + bin_size, bin_size)

def root_func(x):
    return np.sum(curve(x) - np.arange(0.0, 1.0 + bin_size, bin_size))

def calibration():
    ini = root_func(x=1)
    if np.abs(ini) < 1e-3:
        alpha = 1.0
        return alpha
    
    aa = np.arange(0.1, 3.0, 0.2)
    roots = []
    for a in aa:
        roots.append(root_func(a))
    roots = np.array(roots)
    
    index_inverse = np.where(roots > 0)[0][0]
    low = aa[index_inverse - 1]
    high = aa[index_inverse]
    
    alpha = bisect(root_func, low, high, xtol=1e-4)
    return alpha

def beta_calibration(z_true, z_pred, z_errs):
    
    def curve(alpha):
        counts = []
        for low, high in zip(CI[0:-1], CI[1:]):
            ll, lh = norm.interval(low, z_pred, z_errs * alpha)
            hl, hh = norm.interval(high, z_pred, z_errs * alpha)
            
            idx_1 = (z_true > lh) & (z_true < hh)
            idx_2 = (z_true > hl) & (z_true < ll)
            
            idx = idx_1 | idx_2
            counts.append(np.sum(idx))
            
        cp = []
        ini = 0
        cp.append(ini)
        for i in range(len(counts)):
            ini = (ini + counts[i])
            cp.append(ini)
            
        cp = [p/z_true.shape[0] for p in cp]
        return cp
    
    def root_func(x):
        return np.sum(curve(x) - np.arange(0.0, 1.0 + bin_size, bin_size))
    
    ini = root_func(x=1)
    if np.abs(ini) < 1e-3:
        alpha = 1.0
        return alpha
    
    aa = np.arange(0.1, 3.0, 0.2)
    roots = []
    for a in aa:
        roots.append(root_func(a))
    roots = np.array(roots)
    
    index_inverse = np.where(roots > 0)[0][0]
    low = aa[index_inverse - 1]
    high = aa[index_inverse]
    
    alpha = bisect(root_func, low, high, xtol=1e-4)
    
    return alpha

def curve(alpha, z_true, z_pred, z_errs, CI):
    counts = []
    for low, high in zip(CI[0:-1], CI[1:]):
        ll, lh = norm.interval(low, z_pred, z_errs * alpha)
        hl, hh = norm.interval(high, z_pred, z_errs * alpha)
        
        idx_1 = (z_true > lh) & (z_true < hh)
        idx_2 = (z_true > hl) & (z_true < ll)
        
        idx = idx_1 | idx_2
        counts.append(np.sum(idx))
        
    cp = []
    ini = 0
    cp.append(ini)
    for i in range(len(counts)):
        ini = (ini + counts[i])
        cp.append(ini)
        
    cp = [p/z_true.shape[0] for p in cp]
    return cp

