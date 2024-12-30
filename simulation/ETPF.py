import numpy as np
from scipy.optimize import linprog
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
np.random.seed(114222)

def op(x,noise=True):
    return x**2+np.log(x**2+1)+noise*C*np.random.normal(size=x.shape)

def obs(x):
    return x + D*np.random.normal(size=x.shape)

def post(x0,x1):
    return np.exp(-(x0**2+(x0-y[0])**2+(x1-op(x0,False))**2/(C**2)+(x1-y[1])**2)/2)

def marginal(x1):
    low_x0 = -10
    up_x0 = 10
    seg = 20000
    delta = np.linspace(low_x0,up_x0,seg)
    length = (up_x0-low_x0)/seg
    return np.array([np.sum(post(delta,i)*length) for i in x1])

def density(x,mean,variance):
    return np.exp(-(x-mean)**2/2/variance)

def distance(x,y):
    return (x-y)**2

dim = 1  # dimension of x

A = 1
B = 1
C = 0.1
D = 1
T = 1

y = np.array([0.0,0.0])

M = 100 # the number of ETPF samples

#initialize particels
pred_x = []
recon_x = np.random.normal(size=M)
delta = 0 # inflation coefficient
post_x = np.zeros(shape=[2,M])
weights = np.array([1/M for i in range(M)])

# as time evolves
for t in range(T+1):
    # sample p(x_{t+1}|x_t)
    if t != 0:
        recon_x = op(recon_x,noise=False)
        
    weights = density(y[t],recon_x,D**2)
    weights = weights/np.sum(weights)
    coeff = np.zeros([M,M])
    for i in range(M):
        for j in range(i,M):
            coeff[i][j] = distance(recon_x[i],recon_x[j])
            coeff[j][i] = coeff[i][j]
    coeff = coeff.reshape(M**2)
    Aeq = np.zeros([2*M-1,M*M])
    Aub = np.zeros([M*M,M*M])
    beq = list(weights)
    beq.extend([1/M for i in range(M-1)])
    beq = np.array(beq)
    bub = np.zeros(M*M)
    for i in range(M):
        tmp = [0 for i in range(i*M)]
        tmp.extend([1 for i in range(M)])
        tmp.extend([0 for i in range((M-i-1)*M)])
        Aeq[i] = np.array(tmp)
    for i in range(M-1):
        tmp = [0 for i in range(M*M)]
        for j in range(M):
            tmp[j*M+i] = 1
        Aeq[M+i] = np.array(tmp)
    for i in range(M*M):
        tmp = np.zeros(M*M)
        tmp[i] = -1
        Aub[i] = tmp
    res = linprog(coeff,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,method='highs')
    P = M*res.x.reshape([M,M])
    recon_x = P.T@recon_x
    
    post_x[t] = recon_x
    # record the best single estimate
    pred_x.append(np.mean(recon_x,axis=0))

np.save('ETPF.npy',post_x[1])
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(post_x[1],bins=40,density=True)
plt.title('ETPF',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
