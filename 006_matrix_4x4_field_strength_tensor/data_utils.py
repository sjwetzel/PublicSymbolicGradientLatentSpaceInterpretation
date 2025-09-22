# -*- coding: utf-8 -*-
"""
@author: Sebastian
"""

import numpy as np


### use create_dataset(size) to create a list of datapoints. X consists of a combination of 2 space time coordinates (x,t) seen from two different reference frames. Y contrains the space time intervals.


#### Lorentz transformations


def lorentz_boost_4d(beta):
    """takes a list x_mu=[t,x,y,z] and transforms it into a comoving frame with velocity v measured in multiples of the speed of light c"""
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
    

def space_time_interval(x):
    s=x**2
    s[0]=-s[0]
    return np.sum(s)

##### electromagnetic fields


def field_str_tensor(A,A2=0):
    """calculates field strength tensor input a list of length 6 containing E,B or input a pair of E,B"""
    if len(A)==6:
        E=A[:3]
        B=A[3:]
    else:
        E=A
        B=A2
    tensor=np.zeros([4,4])
    tensor[1:,0]=-np.array(E)
    tensor[0,1:]=np.array(E)
    tensor[1,2]=-B[2]
    tensor[1,3]=B[1]
    tensor[2,3]=-B[0]
    tensor[2,1]=-tensor[1,2]
    tensor[3,1]=-tensor[1,3]
    tensor[3,2]=-tensor[2,3]
    return np.array(tensor)


def fields_from_tensor(tensor,single_out=True):
    """calculates E,B from field strength tensor"""
    E=np.zeros(3)
    B=np.zeros(3)
    E=tensor[0,1:]
    B[0]=-tensor[2,3]
    B[1]=tensor[1,3]
    B[2]=-tensor[1,2]
    if single_out==False:
        return E,B
    else:
        return np.concatenate([E,B])


def invariant(A):
    """returns E dot B"""
    E=A[:3]
    B=A[3:]
    return np.dot(E,B)

def invariant2(A):
    """returns E dot E - B dot B"""
    E=A[:3]
    B=A[3:]
    return np.dot(E,E)-np.dot(B,B)


##### create data set


def data_point():
    
    A=np.random.rand(6)*2
    
    F=field_str_tensor(A)
    
    L=random_lorentz_transformation_4d()
    
    F_new=np.linalg.multi_dot([L,F,np.transpose(L)])
    
    A_new = fields_from_tensor(F_new)
    return np.array([A,A_new])


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
        n = np.random.rand(6)

        triplet = np.append(ap_pair, n)

        X.append(triplet)
    
    return np.array(X)