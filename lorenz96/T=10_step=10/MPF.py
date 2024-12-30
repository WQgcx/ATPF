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
    return z

def obs(x):
    return x@B + D*torch.normal(0,1,size=(x@B).shape)

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
T = 20
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)[:,::2]
C = 0
D = 1

# MPF parameter
MPF_n = 3
alpha = torch.tensor([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0])

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
x = torch.zeros(size=[T+1,dim])
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 1000000 # the number of MPF samples

#initialize particels
pred_x = []
seqMC = torch.normal(0,1,size=[M,dim]).to(device)
RMSE = 0
tau = 0.3  # inflation coefficient
weights = torch.tensor([1/M for i in range(M)])

# as time evolves
for t in range(T+1):
    if t != 0:
        seqMC = op(seqMC)
    weights = density(y[t],seqMC@B,D**2)
    weights /= torch.sum(weights)
    
    # record the best single estimate
    pred_x.append((weights@seqMC).detach().cpu().numpy())
    
    # compute loss
    RMSE += torch.sqrt(torch.sum((weights@seqMC-x[t])**2))/T/np.sqrt(40)
    
    # resample, merge and inflat PF particles
    seqMC_M = torch.zeros_like(seqMC)
    for i in range(MPF_n):
        t_seqMC = seqMC[torch.multinomial(weights,M,replacement=True)]
        seqMC_M += alpha[i] * t_seqMC
    seqMC = seqMC_M
    cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M
    eigvalues, eigvec = torch.linalg.eigh(cov)
    eigvalues += eps
    seqMC += 2 * tau * torch.normal(0,1,size=[M,dim]).to(device)@(eigvec@np.diag(eigvalues**(0.5))@eigvec.T)

print('RMSE', RMSE)
np.save('Data/MPF_T=10.npy',pred_x)
