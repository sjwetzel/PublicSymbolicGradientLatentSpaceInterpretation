# -*- coding: utf-8 -*-
"""
@author: Sebastian
"""

import numpy as np


### use create_dataset(size) to create a list of datapoints. X consists of a combination of 2 space time coordinates (x,t) seen from two different reference frames. Y contrains the space time intervals.
#### Lorentz transformations

def invariant(x):
    return x[0]**2-x[1]**2-x[2]**2-x[3]**3
    

def lorentz_boost_4d(beta):
    '''
    
    '''
    gamma=1/np.sqrt(1-beta**2)
    
    Lambda=np.identity(4)
    Lambda[0,0]=gamma
    Lambda[1,1]=gamma
    Lambda[0,1]=-beta*gamma
    Lambda[1,0]=-beta*gamma
    
    return Lambda

def rotation_embedding_4d(phi_1,phi_2,phi_3):
    A_x=np.identity(4)
    A_y=np.identity(4)
    A_z=np.identity(4)
    
    A_x[2,2]=np.cos(phi_1)
    A_x[2,3]=np.sin(phi_1)
    A_x[3,2]=-np.sin(phi_1)
    A_x[3,3]=np.cos(phi_1)
    
    A_y[1,1]=np.cos(phi_2)
    A_y[1,3]=-np.sin(phi_2)
    A_y[3,1]=np.sin(phi_2)
    A_y[3,3]=np.cos(phi_2)      
    
    A_z[1,1]=np.cos(phi_3)
    A_z[1,2]=np.sin(phi_3)
    A_z[2,1]=-np.sin(phi_3)
    A_z[2,2]=np.cos(phi_3)
    
    return np.linalg.multi_dot([A_x,A_y,A_z])

def lorentz_transformation_4d(beta,phi_1,phi_2,phi_3,psi_1,psi_2,psi_3):
    return np.linalg.multi_dot([rotation_embedding_4d(phi_1,phi_2,phi_3),lorentz_boost_4d(beta),rotation_embedding_4d(psi_1,psi_2,psi_3)])

def random_lorentz_transformation_4d(return_parameters=False):
    beta=np.random.uniform(0,1)
    phi_1=np.random.uniform(0,2*np.pi)
    phi_2=np.random.uniform(0,2*np.pi)
    phi_3=np.random.uniform(0,2*np.pi)
    psi_1=np.random.uniform(0,2*np.pi)
    psi_2=np.random.uniform(0,2*np.pi)
    psi_3=np.random.uniform(0,2*np.pi)
    
    if return_parameters==False:
        return lorentz_transformation_4d(beta,phi_1,phi_2,phi_3,psi_1,psi_2,psi_3)
    else:
        return lorentz_transformation_4d(beta,phi_1,phi_2,phi_3,psi_1,psi_2,psi_3),np.array([beta,phi_1,phi_2,phi_3,psi_1,psi_2,psi_3])

def data_point():
    
    x=np.random.rand(4)
    L=random_lorentz_transformation_4d()
    x_new=np.linalg.multi_dot([L,x])

    return np.array([x,x_new])
        
    
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

