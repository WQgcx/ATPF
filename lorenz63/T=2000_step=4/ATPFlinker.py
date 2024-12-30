import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
import copy
np.random.seed(1234)
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
torch.cuda.manual_seed_all(4232356)
device = 'cuda'

t_max = 4
delta_t = 0.02

def op(x,noise=True):
    sigma = 10
    rho = 28
    beta = 8/3
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
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)
C = 0
D = 1
tau = 0.08
delta_reconx = 0.06
eps = 1e-6
lamda = 0.0

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 50 # the number of ATPFker samples
M2 = 1000 # the number of PF samples

class generator(nn.Module):
    def __init__(self,indim=dim*2,middim=512,outdim=dim):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.SiLU(),
                                  nn.Linear(middim,middim),
                                  nn.CELU(),
                                  nn.Linear(middim,outdim))
    def forward(self,x):
        return self.nets(x)

transformation = [generator().to(device) for t in range(T+1)]

# initialize ATPFker and PF particles
seqMC = torch.normal(0,1,size=[M2,dim]).to(device)
weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
recon_x = torch.normal(0,1,size=[M,dim]).to(device)

# as time evolves
for t in range(T+1): # train transformation[t]
    if t != 0:
        transformation[t] = copy.deepcopy(transformation[t-1])
    max_iter = 500
    optimizer = torch.optim.Adam(transformation[t].parameters(),lr=0.002)

    # update PF weights to the posterior
    weights *= density(y[t],seqMC@B,D**2)
    weights /= torch.sum(weights)

    yt = torch.stack(tuple(y[t] for _ in range(M)))
    for iteration in range(1,max_iter+1):
        optimizer.zero_grad()

        # update ATPFker particles to the fitted posterior
        tmp = transformation[t](torch.cat((recon_x.detach(),yt),1))

        # compute loss
        loss = torch.sum((weights@seqMC-torch.mean(tmp,0))**2)

        # backpropagation
        loss.backward()
        optimizer.step()
        if iteration == max_iter:
            print('Training transformation',t,'finished.')
            torch.save(transformation[t],'net_'+str(t)+'.pt')
            break
    
    # resample PF particles
    seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
    weights = torch.tensor([1/M2 for i in range(M2)]).to(device)

    # inflate PF particles
    cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
    eigvalues, eigvec = torch.linalg.eigh(cov)
    eigvalues += eps
    seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@torch.diag(eigvalues**0.5)@eigvec.T)
    
    # transform ATPFker particles to the posterior of the current state
    yt = torch.stack(tuple(y[t] for _ in range(M)))
    recon_x = transformation[t](torch.cat((recon_x,yt),1))

    # inflat ATPFker particles
    recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))

    # sample p(x_{t+1}|x_t)
    if t != T:
        seqMC = op(seqMC,noise=False)
        recon_x = op(recon_x,noise=False)

# save net
for t in range(T+1):
    torch.save(transformation[t],'net_'+str(t)+'.pt')

# test
recon_x = torch.normal(0,1,size=[M,dim]).to(device)
RMSE = 0
pred_x = []
for t in range(T+1):
    yt = torch.stack(tuple(y[t] for i in range(M)))
    recon_x = transformation[t](torch.cat((recon_x,yt),1))
    pred_x.append(list(map(float,torch.mean(recon_x,axis=0))))
    RMSE += torch.sqrt(torch.sum((torch.mean(recon_x,axis=0)-x[t])**2))/np.sqrt(3)/T
    recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))
    if t != T:
        recon_x = op(recon_x,noise=False)
print('RMSE',RMSE)

pred_x = np.array(pred_x)
ax = plt.axes(projection='3d')
cpu_x = x.detach().cpu().numpy()
ax.plot3D(cpu_x[:,0],cpu_x[:,1],cpu_x[:,2])
ax.plot3D(pred_x[:len(x),0],pred_x[:len(x),1],pred_x[:len(x),2])
plt.title('KATPF(linear), T = '+str(T)+', step = '+str(t_max)+', lam = '+str(lamda))
plt.savefig('ATPFlinker (pri T='+str(T)+' step='+str(t_max)+' lam='+str(lamda)+').png')