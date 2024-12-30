import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
np.random.seed(1234)
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
torch.cuda.manual_seed_all(4232356)
device = 'cpu'

eps = 1e-6
t_max = 4

def op(x,noise=True):
    sigma = 10
    rho = 28
    beta = 8/3
    delta_t = 0.02
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
    return x@B + D*torch.normal(0,1,size=x.shape)

def density(x,mean,variance):
    return torch.exp(-torch.sum((x-mean)**2,1)/2/variance)+eps

dim = 3  # dimension of x
T = 2000
A = torch.eye(dim)
B = torch.eye(dim)
C = 0
D = 1

# generate data
x = torch.zeros(size=[T+1,dim])
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 1000 # the number of NLEAF2 samples

#initialize particels
pred_x = []
seqMC = torch.normal(0,1,size=[M,dim]).to(device)
RMSE = 0
delta = 0.0 # inflation coefficient

# as time evolves
cover = 0
for t in range(T+1):
    if t != 0:
        seqMC = op(seqMC)
    weights = density(y[t],seqMC,D**2)
    weights /= torch.sum(weights)
    obs_y = obs(seqMC)
    x_f = seqMC.clone()
    mu_y_o = weights @ seqMC
    P_y_o = (weights*(seqMC - mu_y_o).T) @ (seqMC - mu_y_o)
    eig1, eig2 = np.linalg.eigh(P_y_o)
    Pyo12 = eig2 @ np.diag((eig1+eps)**0.5) @ eig2.T
    for k in range(M):
        tmp = density(obs_y[k],x_f,D**2)
        tmp /= torch.sum(tmp)
        mu_y_a = tmp @ x_f
        P_y_a = (tmp * (x_f - mu_y_a).T) @ (x_f - mu_y_a)
        eig3, eig4 = np.linalg.eigh(P_y_a)
        Pya_12 = eig4 @ np.diag((eig3+eps)**(-0.5)) @ eig4.T
        seqMC[k] = mu_y_o + (x_f[k] - mu_y_a) @ Pya_12 @ Pyo12
    # compute loss
    RMSE += torch.sqrt(torch.sum((torch.mean(seqMC,0)-x[t])**2))/T/np.sqrt(3)

    # record the best single estimate
    pred_x.append(torch.mean(seqMC,0).detach().cpu())
    seqMC = torch.mean(seqMC,0) + (1+delta)*(seqMC-torch.mean(seqMC,0))
print('RMSE', RMSE)

x = x.detach().cpu()
pred_x = np.array(pred_x)
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],x[:,2])
ax.plot3D(pred_x[:len(x),0],pred_x[:len(x),1],pred_x[:len(x),2])
plt.title('NLEAF2, T = '+str(T)+', step = '+str(t_max))
plt.show()
