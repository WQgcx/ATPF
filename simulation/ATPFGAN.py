import numpy as np
import torch
from torch import nn
import copy
import matplotlib.pyplot as plt
import itertools
np.random.seed(12133)
torch.manual_seed(1123)
torch.cuda.manual_seed(1123)
torch.cuda.manual_seed_all(142323)
device = 'cuda'

def ker(x,y):
    return torch.exp(-torch.sqrt((x.unsqueeze(1)-y.unsqueeze(0))**2+eps)/width)

def op(x,noise=True):
    return x**2+torch.log(x**2+1)+noise*C*torch.normal(0,1,size=x.shape).to(device)

def obs(x):
    return x + D*torch.normal(0,1,size=x.shape).to(device)

def density(x,mean,variance):
    return torch.exp(-(x-mean)**2/2/variance)+eps

def post(x0,x1):
    tmp = op(torch.tensor(x0).to(device),noise=False).detach().cpu().numpy()
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
delta_reconx = 0.15
lamda = 0.001

# generate data
y = torch.tensor([0.0,0.0]).to(device)

M = 50 # the number of EnKFpro samples
M2 = 10000 # the number of PF samples

class generator(nn.Module):
    def __init__(self,indim=2,middim=64,outdim=1):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.SiLU(),
                                  nn.Linear(middim,middim*4),
                                  nn.CELU(),
                                  nn.Linear(middim*4,middim),
                                  nn.Tanh(),
                                  nn.Linear(middim,outdim))
    def forward(self,x):
        return self.nets(x)

class discriminator(nn.Module):
    def __init__(self,indim=1,middim=32,outdim=1,M=40):
        super(discriminator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.SiLU(),
                                  nn.Linear(middim,outdim),
                                  nn.Sigmoid())
        self.M = M
    def forward(self,x):
        return self.nets(x) * self.M

transformation = [generator().to(device) for t in range(T+1)]
optimizer = torch.optim.SGD(itertools.chain(*[transformation[t].parameters() for t in range(T+1)]),lr=0.001)

classifier = discriminator().to(device)
optimizer2 = torch.optim.RMSprop(classifier.parameters(),lr=0.01)

transformation_iteration = 2
classifier_iteration = 1
max_iter = 101

for i in range(max_iter):
    # train transformation
    for _ in range(transformation_iteration):
        # initialize particles
        recon_x = torch.normal(0,1,size=[M]).to(device)
        seqMC = torch.normal(0,1,size=[M2]).to(device)
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
            recon_x = transformation[t](torch.vstack([recon_x,yt]).T).T.reshape(len(recon_x))

            # compute loss
            loss += (torch.abs(torch.mean(classifier(recon_x.reshape([len(recon_x),1])))-weights@classifier(seqMC.reshape([len(seqMC),1])))+lamda*torch.sum((recon_x-last_x)**2)/M)/T/dim
            
            # resample and inflat PF particles
            seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
            weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
            cov = torch.sum((seqMC-torch.mean(seqMC))**2)/M2
            seqMC += 2 * tau * (cov**0.5) * torch.normal(0,1,size=[M2]).to(device)

            # inflat EnKFpro particles
            recon_x = torch.mean(recon_x) + (1+delta_reconx)*(recon_x - torch.mean(recon_x))
            
        # backpropagation
        loss.backward()
        optimizer.step()

    # train classifier
    for _ in range(classifier_iteration):
        # initialize particles
        recon_x = torch.normal(0,1,size=[M]).to(device)
        seqMC = torch.normal(0,1,size=[M2]).to(device)
        weights = torch.tensor([1/M2 for i in range(M2)]).to(device)

        loss2 = 0
        optimizer2.zero_grad()
        
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
            recon_x = transformation[t](torch.vstack([recon_x,yt]).T).T.reshape(len(recon_x))

            # compute loss
            loss2 -= torch.abs(weights@classifier(seqMC.reshape([len(seqMC),1]))-torch.mean(classifier(recon_x.reshape([len(recon_x),1]))))/T/dim
            
            # resample and inflat PF particles
            seqMC = seqMC[torch.multinomial(weights,M2,replacement=True)]
            weights = torch.tensor([1/M2 for i in range(M2)]).to(device)
            cov = torch.sum((seqMC-torch.mean(seqMC))**2)/M2
            seqMC += 2 * tau * (cov**0.5) * torch.normal(0,1,size=[M2]).to(device)

            # inflat EnKFpro particles
            recon_x = torch.mean(recon_x) + (1+delta_reconx)*(recon_x - torch.mean(recon_x))
            
        # backpropagation
        loss2.backward()
        optimizer2.step()

# test
M = 10000
recon_x = torch.normal(0,1,size=[M]).to(device)
RMSE = 0
pred_x = []
post_x = torch.zeros(size=[2,M]).to(device)
for t in range(T+1):
    yt = torch.stack(tuple(y[t] for i in range(M)))
    recon_x = transformation[t](torch.vstack([recon_x,yt]).T).T.reshape(len(recon_x))
    pred_x.append(float(torch.mean(recon_x)))
    recon_x = torch.mean(recon_x) + (1+delta_reconx)*(recon_x-torch.mean(recon_x))
    post_x[t] = recon_x
    if t != T:
        recon_x = op(recon_x)

np.save('ATPFGAN (lam='+str(lamda)+').npy',post_x[1].detach().cpu().numpy())
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(post_x[1].detach().cpu().numpy(),bins=40,density=True)
plt.xlim(-2,4)
plt.title(r'Vanilla ATPF',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('ATPFGAN (lam='+str(lamda)+').png')
