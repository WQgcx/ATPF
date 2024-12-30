import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)
device = 'cpu'

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
T = 1000
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)[:,::2]
C = 0
D = 1

# MPF parameter
MPF_n = 3
alpha = torch.tensor([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0])

# generate data
x = torch.zeros(size=[T+1,dim])
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 30000 # the number of MPF samples

#initialize particels
pred_x = []
seqMC = torch.normal(0,1,size=[M,dim]).to(device)
RMSE = 0
delta = 0.01 # inflation coefficient
weights = torch.tensor([1/M for i in range(M)])

# as time evolves
for t in range(T+1):
    seqMC = op(seqMC)
    weights *= density(y[t],seqMC@B,D**2)
    weights /= torch.sum(weights)
    
    # compute loss
    RMSE += torch.sqrt(torch.sum((torch.mean(seqMC,0)-x[t])**2))/T/np.sqrt(40)
    
    # resample, merge and inflat PF particles
    seqMC_M = torch.zeros_like(seqMC)
    for i in range(MPF_n):
        t_seqMC = seqMC[torch.multinomial(weights,M,replacement=True)]
        seqMC_M += alpha[i] * t_seqMC
    seqMC = seqMC_M
    weights = torch.tensor([1/M for i in range(M)])
    seqMC += delta * (seqMC-torch.mean(seqMC,axis=0))

    # record the best single estimate
    pred_x.append(torch.mean(seqMC,0).detach().cpu())

print('RMSE', RMSE)
np.save('Data/MPF_T=1000.npy',pred_x)
