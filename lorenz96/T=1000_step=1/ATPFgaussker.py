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

def ker(x,y):
    return torch.exp(-torch.sum((x.unsqueeze(1)-y.unsqueeze(0))**2,2)/2/width/width)

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
T = 1000
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)[:,::2]
C = 0
D = 1
tau = 0.01 # inflation coeff. of MPF
delta_reconx = 0.0 # inflaction coeff. of ATPF
eps = 1e-6
lamda = 0.01

# MPF parameter
MPF_n = 3
alpha = torch.tensor([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0]).to(device)

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 50 # the number of EnKFker samples
M2 = 30000 # the number of PF samples
max_iter = 500

class generator(nn.Module):
    def __init__(self,indim=dim+dim//2,middim=256,outdim=dim):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.ReLU(),
                                  nn.Linear(middim,middim*2),
                                  nn.Tanh(),
                                  nn.Linear(middim*2,outdim))
    def forward(self,x):
        return self.nets(x)

transformation = [generator().to(device) for t in range(T+1)]
# optimizer = torch.optim.Adam(itertools.chain(*[transformation[t].parameters() for t in range(T+1)]),lr=0.001)

# initialize EnKFker and PF particles
seqMC = torch.normal(0,1,size=[M2,dim]).to(device)
recon_x = torch.normal(0,1,size=[M,dim]).to(device)

# as time evolves
for t in range(T+1): # train transformation[t]
    if t != 0:
        transformation[t] = copy.deepcopy(transformation[t-1])

    optimizer = torch.optim.Adam(transformation[t].parameters(),lr=0.0002)

    # update PF weights to the posterior
    weights = density(y[t],seqMC@B,D**2)
    weights /= torch.sum(weights)

    yt = torch.stack(tuple(y[t] for _ in range(M)))
    pre_loss = -10000
    for iteration in range(1,max_iter+1):
        optimizer.zero_grad()

        # update EnKFker particles to the fitted posterior
        tmp = transformation[t](torch.cat((recon_x.detach(),yt),1))
        
        # compute width
        distance1 = torch.zeros(M*(M-1)//2).to(device)
        nxt = 0
        for i in range(M):
            for j in range(i+1,M):
                distance1[nxt]=torch.sqrt(torch.sum((tmp[i]-tmp[j])**2)).detach()
                nxt += 1
        distance2 = torch.sqrt(torch.sum((tmp.unsqueeze(1)-seqMC.unsqueeze(0))**2,2)).reshape(M*M2).detach().to(device)
        width = torch.median(torch.cat([distance1,distance2]))

        # compute loss
        loss = torch.sum(ker(tmp,tmp))/M/M-2*torch.sum(weights@ker(seqMC,tmp))/M

        # backpropagation
        loss.backward()
        optimizer.step()

        if iteration == max_iter or np.abs(pre_loss - loss.item()) < 1e-10:
            print('Training transformation',t,'finished after',iteration,'iterations. Loss =',loss.item())
            torch.save(transformation[t],'net_'+str(t)+'.pt')
            break
        pre_loss = loss.item()

    # resample, merge and inflat PF particles
    seqMC_M = torch.zeros_like(seqMC).to(device)
    for i in range(MPF_n):
        t_seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
        seqMC_M += alpha[i] * t_seqMC
    seqMC = seqMC_M
    seqMC += tau * (seqMC-torch.mean(seqMC,axis=0))
    
    # transform EnKFker particles to the posterior of the current state
    yt = torch.stack(tuple(y[t] for _ in range(M)))
    recon_x = transformation[t](torch.cat((recon_x,yt),1))

    # inflat EnKFker particles
    recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))

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
np.save('ATPFgaussker_T=1000.npy',pred_x)