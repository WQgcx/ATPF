import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(1232)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(42323)
device = 'cpu'

sigma = 10
rho = 28
beta = 8/3
delta_t = 0.02
t_max = 1
eps = 1e-6

def op(x,noise=True):
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

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M2 = 1000 # the number of PF samples

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
    weights *= density(y[t],seqMC,D**2)
    weights /= torch.sum(weights)
    pred_x.append(weights@seqMC)
    
    # compute loss
    loss += torch.sqrt(torch.sum((weights@seqMC-x[t])**2))/np.sqrt(3)/T
    # resample and inflat PF particles
    seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
    weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
    cov = (seqMC-torch.mean(seqMC,axis=0)).T@(seqMC-torch.mean(seqMC,axis=0))/M2
    eigvalues, eigvec = np.linalg.eigh(cov)
    eigvalues += eps
    seqMC += 2 * tau * torch.normal(0,1,size=[M2,dim]).to(device)@(eigvec@np.diag(eigvalues**0.5)@eigvec.T)

print('loss:',float(loss))

pred_x = np.array(pred_x)
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],x[:,2])
ax.plot3D(pred_x[:len(x),0],pred_x[:len(x),1],pred_x[:len(x),2])
plt.title('PF, T = '+str(T)+', step = '+str(t_max))
plt.show()
