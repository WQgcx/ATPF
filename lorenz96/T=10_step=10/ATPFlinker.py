import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
import copy
import os
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)
os.environ['PYTHONHASHSEED'] = '1123'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

torch.set_default_dtype(torch.float64)

device = 'cuda'

delta_t = 0.02
eps = 1e-6
t_max = 10

def op(x,noise=True):
    z = x.clone()
    y = torch.zeros(z.shape).to(device)
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
    ra = torch.abs(ra).to(device)
    gp = torch.zeros(size=ra.shape).to(device)
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
tau = 0.344 # inflation coeff. of MPF
delta_reconx = 0.45 # inflaction coeff. of ATPF
lamda = 0.0

# tapering
crad = 5  # cutoff radius
x_grid = torch.arange(dim).to(device)
rho = torch.zeros([dim,dim]).to(device)
for j in range(dim):
    for i in range(dim):
        rad = torch.abs(x_grid[i]-x_grid[j]).to(device)
        rad = torch.min(rad,dim-rad).to(device)     # For a periodic domain, again
        rad = torch.tensor([rad/crad]).to(device)
        rho[i][j] = gaspari_cohn(rad)

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

# MPF parameter
MPF_n = 3
alpha = torch.tensor([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0]).to(device)

M = 500 # the number of EnKFker samples
M2 = 1000000 # the number of PF samples
max_iter = 500

class generator(nn.Module):
    def __init__(self,indim=dim+dim//2,middim=512,outdim=dim):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.CELU(),
                                  nn.Linear(middim,middim),
                                  nn.Tanh(),
                                  nn.Linear(middim,outdim))
    def forward(self,x):
        return self.nets(x)

transformation = [generator().to(device) for t in range(T+1)]
# optimizer = torch.optim.Adam(itertools.chain(*[transformation[t].parameters() for t in range(T+1)]),lr=0.001)

# initialize EnKFker and PF particles
seqMC = torch.normal(0,1,size=[M2,dim]).to(device)
weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
recon_x = torch.normal(0,1,size=[M,dim]).to(device)

# as time evolves
for t in range(T+1): # train transformation[t]
    if t != 0:
        transformation[t] = copy.deepcopy(transformation[t-1])

    optimizer = torch.optim.Adam(transformation[t].parameters(),lr=0.006)
    
    # update PF weights
    weights = density(y[t],seqMC@B,D**2)
    weights /= torch.sum(weights)
    
    yt = torch.stack(tuple(y[t] for _ in range(M)))
    last_x = recon_x.detach()
    pre_loss = -10000
    for iteration in range(1,max_iter+1):
        optimizer.zero_grad()

        # update EnKFker particles to the fitted posterior
        tmp = transformation[t](torch.cat((recon_x.detach(),yt),1))
        
        # compute loss
        loss = torch.sum((weights@seqMC-torch.mean(tmp,axis=0))**2) + lamda * torch.mean((tmp-recon_x.detach())**2)

        # backpropagation
        loss.backward()
        optimizer.step()

        if iteration == max_iter or loss.item() < 1e-12:
            print('Training transformation',t,'finished after',iteration,'iterations. Loss =',loss.item())
            torch.save(transformation[t],'net_'+str(t)+'.pt')
            break

        pre_loss = loss.item()
    
    # transform EnKFker particles to the posterior of the current state
    yt = torch.stack(tuple(y[t] for _ in range(M)))
    recon_x = transformation[t](torch.cat((recon_x,yt),1))

    # inflat EnKFker particles
    recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))

    # resample, merge and inflat PF particles
    seqMC_M = torch.zeros_like(seqMC)
    for i in range(MPF_n):
        t_seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
        seqMC_M += alpha[i] * t_seqMC
    seqMC = seqMC_M
    cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
    eigvalues, eigvec = torch.linalg.eigh(cov)
    eigvalues += eps
    seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@torch.diag(eigvalues**(0.5))@eigvec.T)

    # sample p(x_{t+1}|x_t)
    if t != T:
        seqMC = op(seqMC,noise=False)
        recon_x = op(recon_x,noise=False)

# test
M = 100
recon_x = torch.normal(0,1,size=[M,dim]).to(device)
pred_x = []
RMSE = 0
for t in range(T+1):
    yt = torch.stack(tuple(y[t] for i in range(M)))
    recon_x = transformation[t](torch.cat((recon_x,yt),1))
    RMSE += torch.sqrt(torch.sum((torch.mean(recon_x,axis=0)-x[t])**2))/np.sqrt(40)/T
    pred_x.append(torch.mean(recon_x,axis=0).detach().cpu().numpy())
    recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))
    if t != T:
        recon_x = op(recon_x,noise=False)
print('RMSE',RMSE)
np.save('ATPFlinker_T=10.npy',pred_x)