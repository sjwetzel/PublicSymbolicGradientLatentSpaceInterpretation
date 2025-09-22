# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:27:59 2019

@author: Sebastian
"""

from scipy.integrate import odeint
import numpy as np

### movement in a potential 
### H = 1/2 m (v**2) + 1/2 * k * x^2 + exp(x+1)
### q'=dH/dp, p'=-dH/dq
### x'=v
### x''= v'= 1/m (-grad V ) = -1/m * (k * x + exp(x+1))

def motion_de(y,t,m,k):
    x, v = y
    # dydt = [x_dot, v_dot]
    dydt=[v, -1/m * (k * x)]
    return dydt

def invariant(x):
    return 1/2*x[1]**2+1/2*x[0]**2

def data_point():
    m = 1
    k = 1
    t = np.linspace(0, 3, 10001)

    y_init = np.random.rand(2)
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

        # Sample negative data point from another trajectory
        n = data_point()[0]

        triplet = np.append(ap_pair, n)
        X.append(triplet)
    
    return np.array(X)
    
    
        
    
    