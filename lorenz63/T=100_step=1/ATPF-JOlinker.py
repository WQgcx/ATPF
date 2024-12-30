import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
np.random.seed(1232)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(42323)
device = 'cuda'

def op(x,noise=True):
    sigma = 10
    rho = 28
    beta = 8/3
    delta_t = 0.02
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
    return x@B + D*torch.normal(0,1,size=x.shape).to(device)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 3  # dimension of x
T = 100
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)
C = 0
D = 1
tau = 0.045
delta_reconx = 0.06
eps = 1e-6
lamda = 100.0

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 50 # the number of EnKFpro samples
M2 = 1000 # the number of PF samples

class generator(nn.Module):
    def __init__(self,indim=dim*2,middim=256,outdim=dim):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.ReLU(),
                                  nn.Linear(middim,middim*2),
                                  nn.Tanh(),
                                  nn.Linear(middim*2,outdim))
    def forward(self,x):
        return self.nets(x)

transformation = [generator().to(device) for t in range(T+1)]
optimizer = torch.optim.Adam(itertools.chain(*[transformation[t].parameters() for t in range(T+1)]),lr=0.001)

for i in range(500):
    print('iteration:',i)

    # initialize particles
    recon_x = torch.normal(0,1,size=[M,dim]).to(device)
    seqMC = torch.normal(0,1,size=[M2,dim]).to(device)
    weights = torch.tensor([1/M2 for i in range(M2)]).to(device)

    loss = 0
    optimizer.zero_grad()
    
    # as time evolves
    for t in range(T+1):
        # sample p(x_{t+1}|x_t)
        if t != 0:
            recon_x = op(recon_x,noise=False)
            seqMC = op(seqMC,noise=False)
        
        # update PF weights
        weights *= density(y[t],seqMC,D**2)
        weights /= torch.sum(weights)
        
        # update EnKFpro particles
        yt = torch.stack(tuple(y[t] for i in range(M)))
        last_x = recon_x.clone()
        recon_x = transformation[t](torch.cat((recon_x,yt),1))

        # compute loss
        loss += (torch.sum((weights@seqMC-torch.mean(recon_x,axis=0))**2)+lamda*torch.sum((recon_x-last_x)**2)/M)/T/dim
        
        # resample and inflat PF particles
        seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
        weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
        cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
        eigvalues, eigvec = torch.linalg.eigh(cov)
        eigvalues += eps
        seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@torch.diag(eigvalues**0.5)@eigvec.T)

        # inflat EnKFpro particles
        recon_x = torch.mean(recon_x,axis=0) + (1+delta_reconx)*(recon_x - torch.mean(recon_x,axis=0))
        
    # backpropagation
    loss.backward()
    optimizer.step()
    if float(loss) < 0.0001:
        break
    print('loss:',float(loss))

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
plt.title('KATPF-JO(linear), T = '+str(T)+', step=1, lam = '+str(lamda))
plt.savefig('ATPF-JOlinker (pri T='+str(T)+' step=1 lam='+str(lamda)+').png')
