import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import itertools
np.random.seed(1232)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(42323)
device = 'cpu'

t_max = 2
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
    return x@B + D*torch.normal(0,1,size=x.shape).to(device)

dim = 3  # dimension of x
T = 2000
A = torch.eye(dim).to(device)
B = torch.eye(dim).to(device)
C = 0
D = 1

# generate data
x = torch.zeros(size=[T+1,dim]).to(device)
x[0] = torch.normal(0,1,size=[dim])
for t in range(1,T+1):
    x[t] = op(x[t-1],noise=False)
y = obs(x)

M = 1000 # the number of EnKF samples

#initialize particels
pred_x = []
recon_x = torch.normal(0,1,size=[M,dim]).to(device)
RMSE = 0
delta = 0.01 # inflation coefficient

# as time evolves
for t in range(T+1):
    # sample p(x_{t+1}|x_t)
    if t != 0:
        recon_x = op(recon_x,noise=False)

    # update EnKF particles
    Ct = (recon_x-torch.mean(recon_x,0)).T@(recon_x-torch.mean(recon_x,0))/(len(recon_x)-1)
    Kt = Ct@torch.linalg.inv(Ct+D**2*torch.eye(dim).to(device)).to(device)
    recon_x += (y[t]+torch.normal(0,D,size=[M,dim]).to(device)-recon_x)@Kt.T
    recon_x = torch.mean(recon_x,0)+(1+delta)*(recon_x-torch.mean(recon_x,0))

    # compute RMSE
    RMSE += torch.sqrt(torch.sum((torch.mean(recon_x,0)-x[t])**2))/T/np.sqrt(3)

    # record the best single estimate
    pred_x.append(torch.mean(recon_x,0).detach().cpu())

print('RMSE', RMSE)

x = x.detach().cpu()
pred_x = np.array(pred_x)
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],x[:,2])
ax.plot3D(pred_x[:len(x),0],pred_x[:len(x),1],pred_x[:len(x),2])
plt.title('EnKF, T = '+str(T)+', step = '+str(t_max))
plt.show()
