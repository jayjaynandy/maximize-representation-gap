import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(9988)

def get_data():

    d = 300
    
    cov = [[4, 0], [0, 4]]
    mean1 = [-4, 0]
    x1 = np.random.multivariate_normal(mean1, cov, d)
    y1 = np.zeros([d,3])
    y1[:,0] = 1
    plt.plot(x1[:, 0], x1[:, 1],'bx')
    
    mean2 = [4, 0]
    x2 = np.random.multivariate_normal(mean2, cov, d)
    y2 = np.zeros([d,3])
    y2[:,1] = 1
    plt.plot(x2[:, 0], x2[:, 1], 'yx')
    
    mean3 = [0, 5]
    x3 = np.random.multivariate_normal(mean3, cov, d)
    y3 = np.zeros([d,3])
    y3[:,2] = 1
    plt.plot(x3[:, 0], x3[:, 1], 'rx')
    
    h = 200
    dataX = np.concatenate((x1[:h,:], x2[:h,:], x3[:h,:]), axis = 0)
    dataY = np.concatenate((y1[:h,:], y2[:h,:], y3[:h,:]), axis = 0)
    
    
    dist_val = 6.5
    
    def euclid_dist(i,j, mu):
        
        dist = np.sqrt((i-mu[0])*(i-mu[0]) + (j-mu[1])*(j-mu[1]))
        return dist
    
    count = 0
    while count< 600:
        i = np.random.uniform(-15, 15)
        j = np.random.uniform(-13, 17)
        
        d1 = euclid_dist(i,j,mean1)
        d2 = euclid_dist(i,j,mean2)
        d3 = euclid_dist(i,j,mean3)    
        
        if d1>dist_val and d2>dist_val and d3>dist_val:
            dataX = np.concatenate( (dataX, [[i,j]]), axis=0 )
            dataY = np.concatenate( (dataY, [[0.33, 0.33, 0.33]]), axis=0 )
            count += 1
    
    plt.plot(dataX[h*3:, 0], dataX[h*3:, 1], 'k,')
    
    
#    np.savez('synt_1', x_train = dataX, y_train= dataY, x_test = testX, y_test= testY)    
    #plt.title("Mixture of 3 Gaussian distributions")
    plt.plot(-20,20)
    plt.plot(20,-20)
    plt.plot(-20,-20)
    plt.plot(20,20)
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    return dataX, dataY

get_data()
