import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
import copy
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)
device = 'cuda'

delta_t = 0.01
t_max = 1

def op(x,noise=True):
    z = x.clone()
    y = torch.zeros(z.shape).to(device)
    if len(x.shape) == 1:
        for i in range(dim):
            y[i] = (z[(i+1)%dim]-z[(i-2)%dim])*z[(i-1)%dim]-z[i%dim]+8
    else:
        for i in range(dim):
            y[:,i] = (z[:,(i+1)%dim]-z[:,(i-2)%dim])*z[:,(i-1)%dim]-z[:,i%dim]+8  
    return z + delta_t * y

def obs(x):
    return x@B + D*torch.normal(0,1,size=(x@B).shape).to(device)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance).to(device) + eps

dim = 40  # dimension of x
T = 100
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)[:,::2]
C = 0
D = 1
tau = 0.022 # inflation coeff. of MPF
delta_reconx = 0.05 # inflaction coeff. of ATPF
eps = 1e-6
lamda = 250.0

# MPF parameter
MPF_n = 3
alpha = torch.tensor([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0]).to(device)

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim]).to(device)
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 100 # the number of ATPF samples
M2 = 30000 # the number of MPF samples

class generator(nn.Module):
    def __init__(self,indim=dim+dim//2,middim=256,outdim=dim):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.Tanh(),
                                  nn.Linear(middim,middim*4),
                                  nn.Tanh(),
                                  nn.Linear(middim*4,middim*2),
                                  nn.Tanh(),
                                  nn.Linear(middim*2,outdim))
    def forward(self,x):
        return self.nets(x)

transformation = [generator().to(device) for t in range(T+1)]
optimizer = torch.optim.Adam(itertools.chain(*[transformation[t].parameters() for t in range(T+1)]),lr=0.0006)

pre_loss = -10000
# for i in range(1000):
#     print('iteration:',i)

#     #initialize particels
#     recon_x = torch.normal(0,1,size=[M,dim]).to(device)
#     seqMC = torch.normal(0,1,size=[M2,dim]).to(device)

#     loss = 0
#     optimizer.zero_grad()
    
#     # as time evolves
#     for t in range(T+1):
#         if t != 0:
#             recon_x = op(recon_x,noise=False)
#             seqMC = op(seqMC,noise=False)
        
#         # update MPF particles
#         weights = density(y[t],seqMC@B,D**2)
#         weights /= torch.sum(weights)
        
#         # update ATPS particles
#         yt = torch.stack(tuple(y[t] for i in range(M)))
#         last_x = recon_x.clone()
#         recon_x = transformation[t](torch.cat((recon_x,yt),1))

#         # compute loss
#         loss += (torch.sum((weights@seqMC-torch.mean(recon_x,axis=0))**2)+lamda*torch.sum((recon_x-last_x)**2)/M)/T/dim
        
#         # resample, merge and inflat PF particles
#         seqMC_M = torch.zeros_like(seqMC).to(device)
#         for i in range(MPF_n):
#             t_seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
#             seqMC_M += alpha[i] * t_seqMC
#         seqMC = seqMC_M
#         seqMC += tau * (seqMC-torch.mean(seqMC,axis=0))
        
#         # inflat ATPS particles
#         recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))
    
#     # backpropagation
#     loss.backward()
#     optimizer.step()
#     if np.abs(loss.item() - pre_loss) < 1e-7:
#         break
#     print('loss:',loss.item())
#     pre_loss = loss.item()

# # save net:
# for t in range(T+1):
#     torch.save(transformation[t],'net_'+str(t)+'.pt')

for t in range(T+1):
    transformation[t] = torch.load('net_'+str(t)+'.pt')

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
np.save('ATPF-JOlinker_T=100.npy',pred_x)