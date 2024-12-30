import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)
np.random.seed(12133)

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
A = torch.eye(dim)
B = torch.eye(dim)[:,::2]
C = 0
D = 1

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

M = 500 # the number of EnKF samples

#initialize particels
pred_x = []
recon_x = torch.normal(0,1,size=[M,dim])
RMSE = 0
delta = 0.0 # inflation coefficient

# as time evolves
for t in range(T+1):
    # sample p(x_{t+1}|x_t)
    if t != 0:
        recon_x = op(recon_x,noise=False)

    # update EnKF particles
    Ct = rho * ((recon_x-torch.mean(recon_x,0)).T@(recon_x-torch.mean(recon_x,0))/(len(recon_x)-1))
    Kt = Ct@B@torch.linalg.inv(B.T@Ct@B+D**2*torch.eye(dim//2))
    recon_x += (y[t]+torch.normal(0,D,size=[M,dim//2])-recon_x@B)@Kt.T
    recon_x = torch.mean(recon_x,0)+(1+delta)*(recon_x-torch.mean(recon_x,0))

    # compute RMSE
    RMSE += torch.sqrt(torch.sum((torch.mean(recon_x,0)-x[t])**2))/T/np.sqrt(40)

    # record the best single estimate
    pred_x.append(torch.mean(recon_x,0))

print('RMSE', RMSE)
np.save('Data/EnKF_T=10.npy',pred_x)
