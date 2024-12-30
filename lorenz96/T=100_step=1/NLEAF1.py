import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)

delta_t = 0.01
t_max = 1
eps = 1e-6

def op(x,noise=True):
    z = x.clone()
    y = torch.zeros(z.shape)
    if len(x.shape) == 1:
        for i in range(dim):
            y[i] = (z[(i+1)%dim]-z[(i-2)%dim])*z[(i-1)%dim]-z[i%dim]+8
    else:
        for i in range(dim):
            y[:,i] = (z[:,(i+1)%dim]-z[:,(i-2)%dim])*z[:,(i-1)%dim]-z[:,i%dim]+8  
    return z + delta_t * y + noise*C*torch.normal(0,1,size=z.shape)

def obs(x):
    return x@B + D*torch.normal(0,1,size=(x@B).shape)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 40  # dimension of x
T = 100
A = torch.eye(dim)
B = torch.eye(dim)[:,::2]
C = 0
D = 1
K = 1
L = 3

# generate data
x = torch.zeros(size=[T+1,dim])
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 500 # the number of NLEAF1 samples

#initialize particels
pred_x = []
recon_x = torch.normal(0,1,size=[M,dim])
RMSE = 0
delta = 0.01

for t in range(T+1):
    if t != 0:
        recon_x = op(recon_x)
    obs_y = obs(recon_x)
    x_f = recon_x.clone()
    for m in range(M):
        change = torch.zeros(size=[dim,dim])
        for d in range(dim):
            low = (d-L+1)//2
            up = (d+L)//2+1
            if up >= 21:
                low -= 20
                up -= 20
            loc = torch.arange(low,up,1)
            weights = density(y[t][loc],(x_f@B)[:,loc],D**2)
            weights /= torch.sum(weights)
            mu_y_o = weights @ x_f
            tmp = density(obs_y[m][loc],(x_f@B)[:,loc],D**2)
            tmp /= torch.sum(tmp)
            change[d] = (mu_y_o - tmp @ x_f)/(2*K+1)
        for d in range(dim):
            p = 0
            low = d-K
            up = d+K+1
            if up >= 41:
                up -= 40
                low -= 40
            loc = torch.arange(low,up,1)
            for s in loc:
                p += change[s][d]
            recon_x[m][d] += p
    # compute RMSE
    RMSE += torch.sqrt(torch.sum((torch.mean(recon_x,0)-x[t])**2))/T/np.sqrt(40)

    # record the best single estimate
    pred_x.append(torch.mean(recon_x,0))

    recon_x = torch.mean(recon_x,0)+(1+delta)*(recon_x-torch.mean(recon_x,0))

print('RMSE', RMSE)
np.save('Data/NLEAF1_T=100.npy',pred_x)
