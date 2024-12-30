import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
np.random.seed(1232)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(42323)
device = 'cpu'

t_max = 2
eps = 1e-6

def op(x,noise=True):
    sigma = 10
    rho = 28
    beta = 8/3
    delta_t = 0.02
    z = x.clone()
    if len(x.shape) == 1:
        for i in range(t_max):
            z = torch.tensor([z[0]+delta_t*sigma*(z[1]-z[0]),
                    z[1]+delta_t*(z[0]*(rho-z[2])-z[1]),
                    z[2]+delta_t*(z[0]*z[1]-beta*z[2])]).to(device)
        return z + noise*C*torch.normal(0,1,size=z.shape).to(device)
    else:
        for i in range(t_max):
            tmp = torch.zeros(size=z.shape).to(device)
            tmp[:,0] = z[:,0]+delta_t*sigma*(z[:,1]-z[:,0])
            tmp[:,1] = z[:,1]+delta_t*(z[:,0]*(rho-z[:,2])-z[:,1])
            tmp[:,2] = z[:,2]+delta_t*(z[:,0]*z[:,1]-beta*z[:,2])
            z = tmp.clone()
        return z + noise*C*torch.normal(0,1,size=z.shape).to(device)

def obs(x):
    return x@B + D*torch.normal(0,1,size=x.shape).to(device)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 3  # dimension of x
T = 2000
A = torch.eye(dim)
B = torch.eye(dim)
C = 0
D = 1

# MPF parameter
MPF_n = 3
alpha = torch.tensor([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0])

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 1000 # the number of MPF samples

#initialize particels
pred_x = []
seqMC = torch.normal(0,1,size=[M,dim]).to(device)
RMSE = 0
delta = 0.01 # inflation coefficient
weights = torch.tensor([1/M for i in range(M)])

# as time evolves
for t in range(T+1):
    seqMC = op(seqMC)
    weights *= density(y[t],seqMC,D**2)
    weights /= torch.sum(weights)
    
    # compute loss
    RMSE += torch.sqrt(torch.sum((torch.mean(seqMC,0)-x[t])**2))/T/np.sqrt(3)
    
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

x = x.detach().cpu()
pred_x = np.array(pred_x)
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],x[:,2])
ax.plot3D(pred_x[:len(x),0],pred_x[:len(x),1],pred_x[:len(x),2])
plt.title('MPF, T = '+str(T)+', step = '+str(t_max))
plt.show()
