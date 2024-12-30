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
eps = 1e-6

def op(x,noise=True):
    z = x.clone()
    y = torch.zeros(z.shape).to(device)
    if len(x.shape) == 1:
        for i in range(dim):
            y[i] = (z[(i+1)%dim]-z[(i-2)%dim])*z[(i-1)%dim]-z[i%dim]+8
    else:
        for i in range(dim):
            y[:,i] = (z[:,(i+1)%dim]-z[:,(i-2)%dim])*z[:,(i-1)%dim]-z[:,i%dim]+8  
    return z + delta_t * y + noise*C*torch.normal(0,1,size=z.shape).to(device)

def obs(x):
    return x@B + D*torch.normal(0,1,size=(x@B).shape).to(device)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 40  # dimension of x
T = 100
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)[:,::2]
C = 0
D = 1
tau = 0.165

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M2 = 30000 # the number of PF samples

# initialize particles
seqMC = torch.normal(0,1,size=[M2,dim]).to(device)
weights = torch.tensor([1/M2 for i in range(M2)])
loss = 0
pred_x = []

# as time evolves
for t in range(T+1):
    # sample p(x_{t+1}|x_t)
    if t != 0:
        seqMC = op(seqMC,noise=False)

    # update PF weights
    weights *= density(y[t],seqMC@B,D**2)
    weights /= torch.sum(weights)
    pred_x.append(weights@seqMC)

    # compute loss
    loss += torch.sqrt(torch.sum((pred_x[-1]-x[t])**2))/np.sqrt(40)/T
    
    # resample and inflat PF particles
    seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
    weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
    cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
    eigvalues, eigvec = np.linalg.eigh(cov)
    eigvalues += eps
    seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@np.diag(eigvalues**(0.5))@eigvec.T)

print('loss:',float(loss))
np.save('Data/PF_T=100.npy',pred_x)
