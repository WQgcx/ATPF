import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)
device = 'cpu'

torch.set_default_dtype(torch.float64)

delta_t = 0.02
eps = 1e-6
t_max = 10

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
    return z + noise * torch.normal(0,1,size=z.shape).to(device)

def obs(x):
    return x@B + D*torch.normal(0,1,size=(x@B).shape).to(device)

def density(x,mean,variance):
    val = -torch.sum((x-mean)**2,1)/2/variance
    val_max = torch.max(val)
    return torch.exp(val - val_max)

def gaspari_cohn(r):
    """Gaspari-Cohn function."""
    if type(r) is float:
        ra = torch.tensor([r])
    else:
        ra = r
    ra = torch.abs(ra)
    gp = torch.zeros(size=ra.shape)
    i = torch.where(ra<=1.)[0]
    gp[i] = -0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
    i = torch.where((ra>1.)*(ra<=2.))[0]
    gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
    if type(r) is float:
        gp = gp[0]
    return gp

dim = 40  # dimension of x
T = 10
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)[:,::2]
C = 0
D = 1
tau = 0.344

# tapering
crad = 5  # cutoff radius
x_grid = torch.arange(dim)
rho = torch.zeros([dim,dim])
for j in range(dim):
    for i in range(dim):
        rad = torch.abs(x_grid[i]-x_grid[j])
        rad = torch.min(rad,dim-rad)     # For a periodic domain, again
        rad = torch.tensor([rad/crad])
        rho[i][j] = gaspari_cohn(rad)

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M2 = 1000000 # the number of PF samples

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
    pred_x[-1] = pred_x[-1].detach().cpu().numpy()
    
    # resample and inflat PF particles
    seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
    weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
    cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
    eigvalues, eigvec = np.linalg.eigh(cov)
    eigvalues += eps
    seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@np.diag(eigvalues**(0.5))@eigvec.T)

print('loss:',float(loss))
np.save('Data/PF_T=10.npy',pred_x)
