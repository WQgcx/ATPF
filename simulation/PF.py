import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)

def op(x,noise=True):
    return x**2+torch.log(x**2+1)+noise*C*torch.normal(0,1,size=x.shape)

def obs(x):
    return x + D*torch.normal(0,1,size=x.shape)

def density(x,mean,variance):
    return torch.exp(-(x-mean)**2/2/variance)+eps

def post(x0,x1):
    tmp = op(torch.tensor(x0),noise=False).detach().cpu().numpy()
    y0 = y[0].detach().cpu().numpy()
    y1 = y[1].detach().cpu().numpy()
    return np.exp(-(x0**2+(x0-y0)**2+(x1-tmp)**2/(C**2)+(x1-y1)**2)/2)

def marginal(x1):
    low_x0 = -10
    up_x0 = 10
    seg = 20000
    delta = np.linspace(low_x0,up_x0,seg)
    length = (up_x0-low_x0)/seg
    return np.array([np.sum(post(delta,i)*length) for i in x1])

dim = 1  # dimension of x

A = 1
B = 1
C = 0.1
D = 1
T = 1
eps = 1e-6
tau = 0.0

y = torch.zeros(2)

M2 = 10000 # the number of PF samples

#initialize particels
seqMC = torch.normal(0,1,size=[M2])
weights = torch.tensor([1/M2 for i in range(M2)])
pred_x = []

# as time evolves
for t in range(T+1):
    # sample p(x_{t+1}|x_t)
    if t != 0:
        seqMC = op(seqMC)
    # update PF weights
    weights *= density(y[t],seqMC,D**2)
    weights /= torch.sum(weights)
    pred_x.append(weights@seqMC)
    
    # resample and inflat PF particles
    seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
    weights = torch.tensor([1/M2 for i in range(M2)])
    cov = torch.sum((seqMC-torch.mean(seqMC))**2)/M2
    seqMC += 2 * tau * (cov**0.5) * torch.normal(0,1,size=[M2])

np.save('PF.npy',seqMC.detach().cpu().numpy())
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(seqMC,bins=40,density=True)
plt.xlim(-2,4)
plt.title('PF',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
