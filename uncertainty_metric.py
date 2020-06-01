from scipy.special import psi, gammaln
import numpy as np


def differential_entropy (logits):
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-10, 10e10)    
    alpha_0 = np.sum(alpha_c, axis=1)
    gammaln_alpha_c = gammaln(alpha_c)
    gammaln_alpha_0 = gammaln(alpha_0)
    
    psi_alpha_c = psi(alpha_c)
    psi_alpha_0 = psi(alpha_0)
    psi_alpha_0 = np.expand_dims(psi_alpha_0, axis = 1)
    
    temp_mat = np.sum((alpha_c-1)*(psi_alpha_c-psi_alpha_0), axis = 1)
    
    metric = np.sum(gammaln_alpha_c, axis=1) - gammaln_alpha_0 - temp_mat
    return metric

def mutual_info(logits):
    logits = logits.astype('float64')
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-10, 10e10)    
    alpha_0 = np.sum(alpha_c, axis=1, keepdims = True)
 
    psi_alpha_c = psi(alpha_c+1)
    psi_alpha_0 = psi(alpha_0+1)
    alpha_div = alpha_c / alpha_0
    
    temp_mat = np.sum(- alpha_div*(np.log(alpha_c) - psi_alpha_c), axis=1)
    metric = temp_mat + np.squeeze(np.log(alpha_0) - psi_alpha_0)
    return metric

def _get_prob(logits):
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-40, 10e40)
    alpha_0 = np.sum(alpha_c, axis=1)
    alpha_0 = np.expand_dims(alpha_0, axis=1)
    
    return (alpha_c/ alpha_0)
    

def entropy (logits):
    prob = _get_prob(logits)
    exp_prob = np.log(prob)
    
    ent = -np.sum(prob*exp_prob, axis=1)
    return ent

def maxP (logits):
    prob = _get_prob(logits)
    metric = np.max(prob, axis=1)
    return metric    
