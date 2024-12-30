import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)

def op(x,noise=True):
    return x**2+np.log(x**2+1)+noise*C*np.random.normal(size=x.shape)

def obs(x):
    return x + D*np.random.normal(size=x.shape)

def density(x,mean,variance):
    return np.exp(-(x-mean)**2/2/variance)+eps

def post(x0,x1):
    return np.exp(-(x0**2+(x0-y[0])**2+(x1-op(x0,False))**2/(C**2)+(x1-y[1])**2)/2)

def post(x0,x1):
    tmp = op(torch.tensor(x0),noise=False).detach().cpu().numpy()
    y0 = y[0]
    y1 = y[1]
    return np.exp(-(x0**2+(x0-y0)**2+(x1-tmp)**2/(C**2)+(x1-y1)**2)/2)

def marginal(x1):
    low_x0 = -10
    up_x0 = 10
    seg = 20000
    delta = np.linspace(low_x0,up_x0,seg)
    length = (up_x0-low_x0)/seg
    return np.array([np.sum(post(delta,i)*length) for i in x1])

dim = 1  # dimension of x

A = 1
B = 1
C = 0.1
D = 1
T = 1
eps = 1e-6
tau = 0.0

y = np.zeros(2)

M = 10000 # the number of NLEAF2 samples

#initialize particels
pred_x = []
seqMC = np.random.normal(size=M)
delta = 0 # inflation coefficient
post_x = np.zeros(shape=[2,M])

# as time evolves
cover = 0
for t in range(T+1):
    if t != 0:
        seqMC = op(seqMC)
    weights = density(y[t],seqMC,D**2)
    weights /= np.sum(weights)
    obs_y = obs(seqMC)
    x_f = seqMC.copy()
    mu_y_o = weights @ seqMC
    P_y_o = (weights*(seqMC - mu_y_o).T) @ (seqMC - mu_y_o)
    Pyo12 = P_y_o**0.5
    for k in range(M):
        tmp = density(obs_y[k],x_f,D**2)
        tmp /= np.sum(tmp)
        mu_y_a = tmp @ x_f
        P_y_a = (tmp * (x_f - mu_y_a).T) @ (x_f - mu_y_a)
        Pya_12 = P_y_a ** (-0.5)
        seqMC[k] = mu_y_o + (x_f[k] - mu_y_a) * Pya_12 * Pyo12

    # record the best single estimate
    pred_x.append(np.mean(seqMC))
    post_x[t] = seqMC

np.save('NLEAF2.npy',post_x[1])
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(post_x[1],bins=40,density=True)
plt.title('NLEAF2',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
