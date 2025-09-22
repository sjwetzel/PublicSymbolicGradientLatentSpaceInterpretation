# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:27:59 2019

@author: Sebastian
"""

from scipy.integrate import odeint
import numpy as np

### movement in a potential 
### H = 1/2 m (v1**2+v2**2) - k 1 / sqrt(x1**2+x2**2) 
### q'=dH/dp, p'=-dH/dq
### x'=v
### x''= v'= 1/m (-grad V )= k/m (x1,x2) /sqrt(x1**2+x2**2)**3/2

def motion_de(y,t,m,k):
    x1,x2,v1,v2=y
    dydt=[v1,v2, -k/m* x1/np.sqrt(x1**2+x2**2)**3, -k/m* x2 /np.sqrt(x1**2+x2**2)**3]
    return dydt


def invariant(x):
    return x[0]*x[3]-x[1]*x[2]

def data_point():
    m = 1
    k = 1
    t = np.linspace(0, 3, 10001)

    y_init = np.random.rand(4)
    sol = odeint(motion_de, y_init, t, args=(m, k))
    i = np.random.randint(len(t))
    j = np.random.randint(len(t))

    return np.array([sol[i],sol[j]])

    
def positive_data(size):
    
    X=[]
    Y=[]
    
    for i in range(size):
        X.append(data_point())
        Y.append(0)
        
    return np.array(X),np.array(Y)
        

def negative_data(data):
    
    a=data[:,0:1]
    b=np.roll(data[:,1:2],1,axis=0)
    
    return np.concatenate((a,b),axis=1),np.ones(len(data))
    

# Can optimize with numpy     
def triplet_data(size):

    X = []

    for i in range(size):
        ap_pair = data_point()
        n = np.random.rand(4)

        triplet = np.append(ap_pair, n)

        X.append(triplet)
    
    return np.array(X)
    
    
        
    
    

