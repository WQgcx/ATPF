import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
import copy
np.random.seed(1232)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(42323)
device = 'cuda'

delta_t = 0.02

def ker(x,y):
    return torch.exp(-torch.sum((x.unsqueeze(1)-y.unsqueeze(0))**2,2)/2/width/width)

def op(x,noise=True):
    sigma = 10
    rho = 28
    beta = 8/3
    t_max = 1
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
    return x@B + D*torch.normal(0,1,size=(x@B).shape).to(device)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 3  # dimension of x
T = 100
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)
C = 0
D = 1
tau = 0.045
delta_reconx = 0.1
eps = 1e-6
lamda = 0.0

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 50 # the number of EnKFpro samples
M2 = 1000 # the number of PF samples

class generator(nn.Module):
    def __init__(self,indim=dim*2,middim=128,outdim=dim):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.SiLU(),
                                  nn.Linear(middim,middim),
                                  nn.CELU(),
                                  nn.Linear(middim,outdim))
    def forward(self,x):
        return self.nets(x)

transformation = [generator().to(device) for t in range(T+1)]

# as time evolves
for t in range(T+1): # train transformation[t]
    if t != 0:
        transformation[t] = copy.deepcopy(transformation[t-1])
    # total_loss = 0
    max_iter = 80
    optimizer = torch.optim.RMSprop(transformation[t].parameters(),lr=0.000015)
    last_loss = -100000
    for iteration in range(1,max_iter+1):
        optimizer.zero_grad()

        # initialize particles
        recon_x = torch.normal(0,1,size=[M,dim]).to(device)
        seqMC = torch.normal(0,1,size=[M2,dim]).to(device)
        weights = torch.tensor([1/M2 for i in range(M2)]).to(device)

        # transform EnKFker and PF particles to the prior of the current state
        for i in range(t):
            # update EnKFker particles
            yt = torch.stack(tuple(y[i] for _ in range(M)))
            recon_x = transformation[i](torch.cat((recon_x,yt),1))
            
            # inflat EnKFker particles
            recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))

            # update PF weights
            weights *= density(y[i],seqMC@B,D**2)
            weights /= torch.sum(weights)
            
            # resample PF particles
            seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
            weights = torch.tensor([1/M2 for i in range(M2)]).to(device)

            # inflate PF particles
            cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
            eigvalues, eigvec = torch.linalg.eigh(cov)
            eigvalues += eps
            seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@torch.diag(eigvalues**0.5)@eigvec.T)

            # sample p(x_{t+1}|x_t)
            recon_x = op(recon_x,noise=False)
            seqMC = op(seqMC,noise=False)
        
        # update EnKFker particles to the fitted posterior
        yt = torch.stack(tuple(y[t] for _ in range(M)))
        last_x = recon_x.clone()
        recon_x = transformation[t](torch.cat((recon_x,yt),1))

        # update PF weights to the posterior
        weights *= density(y[t],seqMC@B,D**2)
        weights /= torch.sum(weights)

        # compute width
        distance1 = torch.zeros(M*(M-1)//2).to(device)
        nxt = 0
        for i in range(M):
            for j in range(i+1,M):
                distance1[nxt]=torch.sqrt(torch.sum((recon_x[i]-recon_x[j])**2)).detach()
                nxt += 1
        distance2 = torch.sqrt(torch.sum((recon_x.unsqueeze(1)-seqMC.unsqueeze(0))**2,2)).reshape(M*M2).detach().to(device)
        width = torch.median(torch.cat([distance1,distance2]))
        
        # compute loss
        loss = torch.sum(ker(recon_x,recon_x))/M/M-2*torch.sum(weights@ker(seqMC,recon_x))/M+lamda*torch.sum((recon_x-last_x)**2)/M

        # backpropagation
        loss.backward()
        optimizer.step()

        if iteration == max_iter or np.abs(float(loss)-last_loss) < 1e-6:
            print('Training transformation',t,'finished after',iteration,'iterations. Current loss is',float(loss),'.')
            torch.save(transformation[t],'net_'+str(t)+'.pt')
            break
        
        last_loss = float(loss)

# test
M = 500
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
print('RMSE:',RMSE)

pred_x = np.array(pred_x)
cpu_x = x.detach().cpu().numpy()
ax = plt.axes(projection='3d')
ax.plot(cpu_x[:,0],cpu_x[:,1],cpu_x[:,2])
ax.plot(pred_x[:len(x),0],pred_x[:len(x),1],pred_x[:len(x),2])
plt.title('KATPF(Gauss), T = '+str(T)+', step = 1, lam = '+str(lamda))
plt.savefig('ATPFgaussker (pri T='+str(T)+' step=1 lam='+str(lamda)+').png')
