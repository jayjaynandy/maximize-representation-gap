import numpy as np

def kl_div(set2, set1,  mean_val, std_val):
    
    score_1 = np.copy(set1)
    score_2 = np.copy(set2)    
    
    mu_1 = np.mean(score_1, axis=0)
    cov_1 = np.cov(score_1.T)
    
    mu_2 = np.mean(score_2, axis=0)
    cov_2 = np.cov(score_2.T)
    
    det_cov_1 = np.linalg.det(cov_1)
    det_cov_2 = np.linalg.det(cov_2)
    
    inv_cov_2 = np.linalg.inv(cov_2)
    
    term1 = np.trace(np.matmul(inv_cov_2, cov_1))
    term2 = np.matmul(np.matmul( (mu_2- mu_1), inv_cov_2), (mu_2- mu_1).T)
    term3 = -2
    term4 = np.log(det_cov_2/ det_cov_1)
    
    total_div = 0.5*(term1 + term2 + term3 + term4)
    print(total_div)
    return total_div
    
    
    
    
    
