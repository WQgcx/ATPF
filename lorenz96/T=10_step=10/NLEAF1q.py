import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)

delta_t = 0.02
t_max = 10
eps = 1e-6

def op(x,noise=True):
    z = x.clone()
    y = torch.zeros(z.shape)
    if len(x.shape) == 1:
        for t in range(t_max):
            for i in range(dim):
                y[i] = (z[(i+1)%dim]-z[(i-2)%dim])*z[(i-1)%dim]-z[i%dim]+8
            z = z + delta_t * y
    else:
        for t in range(t_max):
            for i in range(dim):
                y[:,i] = (z[:,(i+1)%dim]-z[:,(i-2)%dim])*z[:,(i-1)%dim]-z[:,i%dim]+8
            z = z + delta_t * y
    return z

def obs(x):
    return x@B + D*torch.normal(0,1,size=(x@B).shape)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 40  # dimension of x
T = 10
A = torch.eye(dim)
B = torch.eye(dim)[:,::2]
C = 0
D = 1
K = 4
L = 4

# generate data
x = torch.zeros(size=[T+1,dim])
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 500 # the number of EnKF samples

#initialize particels
pred_x = []
recon_x = torch.normal(0,1,size=[M,dim])
RMSE = 0
delta = 0.02

for t in range(T+1):
    recon_x = op(recon_x)
    obs_y = obs(recon_x)
    x_f = recon_x.clone()
    change = torch.zeros([M,dim,dim])
    for d in range(dim):
        low = (d-L+1)//2
        up = (d+L)//2+1
        if up >= 21:
            low -= 20
            up -= 20
        loc = torch.arange(low,up,1)
        low_x = d - L
        up_x = d + L + 1
        if up_x >= 41:
            low_x -= 40
            up_x -= 40
        loc_x = torch.arange(low_x,up_x,1)
        tmp = len(loc)
        design_matrix = torch.ones(size=[M,tmp*2+1])
        design_matrix[:,:tmp] = obs_y[:,loc]**2
        design_matrix[:,tmp:(2*tmp)] = obs_y[:,loc]
        coeff = torch.linalg.inv(design_matrix.T@design_matrix+eps*torch.eye(2*tmp+1))@design_matrix.T@x_f[:,loc_x]
        design_y_o = torch.cat([y[t][loc]**2,y[t][loc],torch.tensor([1])])
        mu_y_o = design_y_o@coeff
        mu_y_a = design_matrix@coeff
        change[:,d,loc_x] = (mu_y_o-mu_y_a)/(2*K+1)
    for d in range(dim):
        p = 0
        low = d - K
        up = d + K + 1
        if up >= 41:
            up -= 40
            low -= 40
        loc = torch.arange(low,up,1)
        for s in loc:
            p += change[:,s,d]
        recon_x[:,d] += p
    # compute RMSE
    RMSE += torch.sqrt(torch.sum((torch.mean(recon_x,0)-x[t])**2))/T/np.sqrt(40)

    # record the best single estimate
    pred_x.append(torch.mean(recon_x,0))

    recon_x = torch.mean(recon_x,0)+(1+delta)*(recon_x-torch.mean(recon_x,0))

print('RMSE', RMSE)
np.save('Data/NLEAF1q_T=10.npy',pred_x)
