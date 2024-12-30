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

y = torch.tensor([0,0])
M = 10000

#initialize particels
seqMC = torch.normal(0,1,size=[M])
delta = 0.01
post_x = np.zeros(shape=[2,M])

# as time evolves
for t in range(T+1):
    if t != 0:
        seqMC = op(seqMC)
    weights = density(y[t],seqMC,D**2)
    weights /= torch.sum(weights)
    obs_y = obs(seqMC)
    x_f = seqMC.clone()
    mu_y_o = weights @ seqMC
    for k in range(M):
        tmp = density(obs_y[k],x_f,D**2)
        tmp /= torch.sum(tmp)
        seqMC[k] += (mu_y_o - tmp @ x_f)
    post_x[t] = seqMC

np.save('NLEAF1.npy',post_x[1])
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(post_x[1],bins=40,density=True)
plt.title('NLEAF1',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
