import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import copy

rainfile_prepared = 0
MCMCfile_prepared = 0
EnKFfile_prepared = 0
PFfile_prepared = 0
ATPFLKfile_prepared = 0
ATPFGKfile_prepared = 0
NLEAF1file_prepared = 0
NLEAF2file_prepared = 0

# parameters
mu = np.array([0.0 for _ in range(100)])
cov = np.zeros([100,100])
scale = 1
width = 20
for i in range(100):
    for j in range(i,100):
        cov[i][j] = scale*np.exp(-np.abs(i-j)/width)
        cov[j][i] = cov[i][j]
sigmat = 0.4
loc = np.array([i+1 for i in range(100)])

# data generator
np.random.seed(1234)
if rainfile_prepared:
    rain = np.load('true_rain.npy')
    oracle_rain = np.load('oracle_rain.npy')
    obs_rain = np.load('obs_rain.npy')
else:
    rain = np.random.multivariate_normal(mean=mu,cov=cov)
    oracle_rain = rain.copy()
    rain[rain<0] = 0
    obs_rain = np.random.multivariate_normal(mean=oracle_rain,cov=sigmat**2*np.identity(100))**3
    obs_rain[obs_rain<0] = 0
    np.save('true_rain.npy',rain)
    np.save('obs_rain.npy',obs_rain)
    np.save('oracle_rain.npy',oracle_rain)

truncation_loc = []
for i in range(100):
    if rain[i] == 0:
        truncation_loc.append(i)

obs_rain1 = copy.deepcopy(obs_rain)
obs_rain = obs_rain1**(1/3)

##plt.subplot(2,1,1)
# MCMC
np.random.seed(4321)
if MCMCfile_prepared:
    MCMC = np.load('MCMC.npy')
else:
    M = 500
    MCMC = np.zeros([M,100])
    cov_1 = np.linalg.inv(cov)
    eps = 0.01
    for i in range(500):
        if i % 10 == 9:
            print(i+1)
        x = np.random.multivariate_normal(mean=mu,cov=cov)
        y = obs_rain.copy()
        for _ in range(10000):
            for k in truncation_loc:
                tmp = 1
                flag = 1
                while tmp > 0 and flag <= 10000:
                    tmp = np.random.normal(loc=x[k],scale=sigmat)
                    flag += 1
                y[k] = min(tmp,0)
            x += eps*(cov_1@(mu-x)+(y-x)/sigmat/sigmat)+np.sqrt(eps)*np.random.normal(size=100)
        MCMC[i] = x
    np.save('MCMC.npy',MCMC)
MCMCpostmean = np.mean(MCMC,axis=0)
MCMCpostmean[MCMCpostmean<0] = 0
MCMClowerbound = np.zeros(100)
MCMCupperbound = np.zeros(100)
for i in range(100):
    tmp = sorted(MCMC[:,i])
    MCMClowerbound[i] = max([0,tmp[len(MCMC)//40]])
    MCMCupperbound[i] = max([0,tmp[len(MCMC)-len(MCMC)//40]])
plt.plot(loc,MCMCpostmean,color='#0000FF',label='MCMC mean')
plt.fill_between(loc,MCMClowerbound,MCMCupperbound,color='#23238E',alpha=0.25)
print('MCMC RMSE:',np.sqrt(np.mean((MCMCpostmean-rain)**2)))

# EnKF
np.random.seed(4321)
if EnKFfile_prepared:
    EnKF = np.load('EnKF.npy')
else:
    M = 500
    x = np.random.multivariate_normal(mean=mu,cov=cov,size=M)
    y = np.zeros(x.shape)
    for i in range(M):
        y[i] = np.random.multivariate_normal(mean=x[i],cov=sigmat**2*np.identity(100))
    y[y<0] = 0
    Cxy = (x-np.mean(x,axis=0)).T@(y-np.mean(y,axis=0))/(M-1)
    Cyy = (y-np.mean(y,axis=0)).T@(y-np.mean(y,axis=0))/(M-1)
    EnKF = x + (Cxy@np.linalg.inv(Cyy)@(obs_rain-y).T).T
    np.save('EnKF.npy',EnKF)
EnKFpostmean = np.mean(EnKF,axis=0)
EnKFpostmean[EnKFpostmean<0] = 0
EnKFlowerbound = np.zeros(100)
EnKFupperbound = np.zeros(100)
for i in range(100):
    tmp = sorted(EnKF[:,i])
    EnKFlowerbound[i] = max([0,tmp[len(EnKF)//40]])
    EnKFupperbound[i] = max([0,tmp[len(EnKF)-len(EnKF)//40]])
plt.plot(loc,EnKFpostmean,color='#FF7F00',label='EnKF mean')
plt.fill_between(loc,EnKFlowerbound,EnKFupperbound,color='#D98719',alpha=0.25)
print('EnKF RMSE:',np.sqrt(np.mean((EnKFpostmean-rain)**2)))

# NLEAF1
np.random.seed(4321)

def normal_density2(x,mean,variance):
    return np.exp(-np.sum((x-mean)**2,axis=1)/2/variance)

if NLEAF1file_prepared:
    NLEAF1 = np.load('NLEAF1.npy')
else:
    M = 500
    NLEAF1 = np.random.multivariate_normal(mean=mu,cov=cov,size=M)
    obs_y = np.random.normal(loc=NLEAF1,scale=sigmat)
    NLEAF1[:,truncation_loc] = 0
    NLEAF1[NLEAF1<0] = 0
    weights = normal_density2(obs_rain,NLEAF1,sigmat**2)
    weights /= np.sum(weights)
    obs_y[:,truncation_loc] = 0
    obs_y[obs_y<0] = 0
    x_f = NLEAF1.copy()
    mu_y_o = weights @ NLEAF1
    for k in range(M):
        tmp = normal_density2(obs_y[k],x_f,sigmat**2)
        tmp /= np.sum(tmp)
        NLEAF1[k] += (mu_y_o - tmp @ x_f)
    np.save('NLEAF1.npy',NLEAF1)
NLEAF1postmean = np.mean(NLEAF1,axis=0)
NLEAF1postmean[NLEAF1postmean<0] = 0
NLEAF1lowerbound = np.zeros(100)
NLEAF1upperbound = np.zeros(100)
for i in range(100):
    tmp = sorted(NLEAF1[:,i])
    NLEAF1lowerbound[i] = max([0,tmp[len(NLEAF1)//40]])
    NLEAF1upperbound[i] = max([0,tmp[len(NLEAF1)-len(NLEAF1)//40]])
plt.plot(loc,NLEAF1postmean,color='#215E21',label='NLEAF1 mean')
plt.fill_between(loc,NLEAF1lowerbound,NLEAF1upperbound,color='#00FF7F',alpha=0.25)
print('NLEAF1 RMSE:',np.sqrt(np.mean((NLEAF1-rain)**2)))

# NLEAF2
np.random.seed(4321)

if NLEAF2file_prepared:
    NLEAF2 = np.load('NLEAF2.npy')
else:
    M = 500
    eps = 1e-6
    NLEAF2 = np.random.multivariate_normal(mean=mu,cov=cov,size=M)
    obs_y = np.random.normal(loc=NLEAF2,scale=sigmat)
    NLEAF2[:,truncation_loc] = 0
    NLEAF2[NLEAF2<0] = 0
    weights = normal_density2(obs_rain,NLEAF2,sigmat**2)
    weights /= np.sum(weights)
    obs_y[:,truncation_loc] = 0
    obs_y[obs_y<0] = 0
    x_f = NLEAF2.copy()
    mu_y_o = weights @ NLEAF2
    P_y_o = (weights*(NLEAF2-mu_y_o).T) @ (NLEAF2-mu_y_o)
    eig1, eig2 = np.linalg.eigh(P_y_o)
    Pyo12 = eig2 @ np.diag((eig1+eps)**0.5) @ eig2.T
    for k in range(M):
        tmp = normal_density2(obs_y[k],x_f,sigmat**2)
        tmp /= np.sum(tmp)
        mu_y_a = tmp @ x_f
        P_y_a = (tmp * (x_f - mu_y_a).T) @ (x_f - mu_y_a)
        eig3, eig4 = np.linalg.eigh(P_y_a)
        Pya_12 = eig4 @ np.diag((eig3+eps)**(-0.5)) @ eig4.T
        NLEAF2[k] = mu_y_o + (x_f[k] - mu_y_a) @ Pya_12 @ Pyo12
    np.save('NLEAF2.npy',NLEAF2)
NLEAF2postmean = np.mean(NLEAF2,axis=0)
NLEAF2postmean[NLEAF2postmean<0] = 0
##NLEAF2lowerbound = np.zeros(100)
##NLEAF2upperbound = np.zeros(100)
##for i in range(100):
##    tmp = sorted(NLEAF2[:,i])
##    NLEAF2lowerbound[i] = max([0,tmp[len(NLEAF2)//40]])
##    NLEAF2upperbound[i] = max([0,tmp[len(NLEAF2)-len(NLEAF2)//40]])
##plt.plot(loc,NLEAF2postmean,color='#5C4033',label='NLEAF2 mean')
##plt.fill_between(loc,NLEAF2lowerbound,NLEAF2upperbound,color='#4F2F4F',alpha=0.25)
print('NLEAF2 RMSE:',np.sqrt(np.mean((NLEAF2-rain)**2)))
##plt.plot(loc,rain,color='#000000',label='true rain')
##plt.plot(loc,obs_rain1,'x',color='#000000',label='obs rain')
##plt.legend(loc='upper right')
##plt.title('rainfall model',fontsize=30)
##plt.subplot(2,1,2)
##plt.plot(loc,MCMCpostmean,color='#0000FF',label='MCMC mean')
##plt.fill_between(loc,MCMClowerbound,MCMCupperbound,color='#23238E',alpha=0.25)

# PF
np.random.seed(4321)

def normal_density(x,mean,cov):
    return np.exp(-(x-mean)@np.linalg.inv(cov)@(x-mean)/2)

if PFfile_prepared:
    PF = np.load('PF.npy')
    weights = np.load('PF_weights.npy')
else:
    M = 100000
    PF = np.random.multivariate_normal(mean=mu,cov=cov,size=M)
    PF[:,truncation_loc] = 0
    PF[PF<0] = 0
    weights = np.zeros(M)
    for i in range(M):
        weights[i] = normal_density(obs_rain,PF[i],sigmat**2*np.identity(100))
    weights /= np.sum(weights)
    np.save('PF.npy',PF)
    np.save('PF_weights.npy',weights)
PFpostmean = weights@PF
plt.plot(loc,PFpostmean,color='#FF0000',label='PF mean')
print('PF RMSE:',np.sqrt(np.mean((PFpostmean-rain)**2)))

# KATPF(linear)
np.random.seed(4321)
torch.manual_seed(4321)

class generator(nn.Module):
    def __init__(self,indim=200,middim=512,outdim=100):
        super(generator, self).__init__()
        self.nets = nn.Sequential(nn.Linear(indim,middim),
                                  nn.SiLU(),
                                  nn.Linear(middim,middim*4),
                                  nn.Tanh(),
                                  nn.Linear(middim*4,middim//16),
                                  nn.LeakyReLU(),
                                  nn.Linear(middim//16,middim*4),
                                  nn.CELU(),
                                  nn.Linear(middim*4,middim),
                                  nn.Tanh(),
                                  nn.Linear(middim,outdim))
    def forward(self,x):
        return self.nets(x)

if ATPFLKfile_prepared:
    ATPFLK = torch.load('ATPFLK.pt')
else:
    M = 500
    lamda = 0.05
    maxvalue = 1
    yt = torch.stack(tuple(torch.tensor(obs_rain,dtype=torch.float32) for _ in range(M)))
    ATPFLK = generator()
    optimizer = torch.optim.Adam(ATPFLK.parameters(),lr=0.00001)
    max_iter = 200
    for iteration in range(1,max_iter+1):
        if iteration % 100 == 0:
            print(iteration,'th iteration of KATPF(linear) now. Loss = ',loss.item())
        recon_x = torch.tensor(np.random.multivariate_normal(mean=mu,cov=cov,size=M),dtype=torch.float32)
        last_x = recon_x.clone()
        recon_x = ATPFLK(torch.cat((recon_x,yt),1))
        recon_x[:,truncation_loc] = 0
        loss = (torch.sum((torch.tensor(PFpostmean)-torch.mean(recon_x,axis=0))**2)+lamda*torch.sum((recon_x-last_x)**2)/M)/100
        loss.backward()
        optimizer.step()
        for p in ATPFLK.parameters():
            p.data.clamp_(-maxvalue,maxvalue)
    torch.save(ATPFLK,'ATPFLK.pt')

M = 500
yt = torch.stack(tuple(torch.tensor(obs_rain,dtype=torch.float32) for _ in range(M)))
recon_x = torch.tensor(np.random.multivariate_normal(mean=mu,cov=cov,size=M),dtype=torch.float32)
recon_x = ATPFLK(torch.cat((recon_x,yt),1)).detach().numpy()
recon_x[:,truncation_loc] = 0
ATPFLKpostmean = np.mean(recon_x,axis=0)
ATPFLKpostmean[ATPFLKpostmean<0] = 0
##ATPFLKlowerbound = np.zeros(100)
##ATPFLKupperbound = np.zeros(100)
##for i in range(100):
##    tmp = sorted(recon_x[:,i])
##    ATPFLKlowerbound[i] = max([0,tmp[M//40]])
##    ATPFLKupperbound[i] = max([0,tmp[M-M//40]])
##plt.plot(loc,ATPFLKpostmean,color='#FF00FF',label='KATPF(linear) mean')
##plt.fill_between(loc,ATPFLKlowerbound,ATPFLKupperbound,color='#FF6EC7',alpha=0.25)
print('KATPF(linear) RMSE:',np.sqrt(np.mean((ATPFLKpostmean-rain)**2)))

# KATPF(gauss)
np.random.seed(4321)
torch.manual_seed(4321)

def ker(x,y):
    return torch.exp(-torch.sum((x.unsqueeze(1)-y.unsqueeze(0))**2,2)/2/width/width)

if ATPFGKfile_prepared:
    ATPFGK = torch.load('ATPFGK.pt')
else:
    M = 100
    M2 = len(MCMC)
    MCMCtorch = torch.tensor(MCMC)
    lamda = 0.0
    maxvalue = 1
    yt = torch.stack(tuple(torch.tensor(obs_rain,dtype=torch.float32) for _ in range(M)))
    ATPFGK = generator()
    optimizer = torch.optim.Adam(ATPFGK.parameters(),lr=0.00001)
    max_iter = 200
    for iteration in range(1,max_iter+1):
        if iteration % 100 == 0:
            print(iteration,'th iteration of KATPF(Gauss) now. Loss = ',loss.item())
        recon_x = torch.tensor(np.random.multivariate_normal(mean=mu,cov=cov,size=M),dtype=torch.float32)
        last_x = recon_x.clone()
        recon_x = ATPFGK(torch.cat((recon_x,yt),1))
        recon_x[:,truncation_loc] = 0

        distance1 = torch.zeros(M*(M-1)//2)
        nxt = 0
        for i in range(M):
            for j in range(i+1,M):
                distance1[nxt]=torch.sqrt(torch.sum((recon_x[i]-recon_x[j])**2)).detach()
                nxt += 1
        distance2 = torch.sqrt(torch.sum((recon_x.unsqueeze(1)-MCMCtorch.unsqueeze(0))**2,2)).reshape(M*M2).detach()
        width = torch.median(torch.cat([distance1,distance2]))
        
        loss = torch.sum(ker(recon_x,recon_x))/M/M-2*torch.sum(ker(MCMCtorch,recon_x))/M/M+lamda*torch.sum((last_x-recon_x)**2)/M
        loss.backward()
        optimizer.step()
        for p in ATPFGK.parameters():
            p.data.clamp_(-maxvalue,maxvalue)
    torch.save(ATPFGK,'ATPFGK.pt')

M = 500
yt = torch.stack(tuple(torch.tensor(obs_rain,dtype=torch.float32) for _ in range(M)))
recon_x = torch.tensor(np.random.multivariate_normal(mean=mu,cov=cov,size=M),dtype=torch.float32)
recon_x = ATPFGK(torch.cat((recon_x,yt),1)).detach().numpy()
recon_x[:,truncation_loc] = 0
ATPFGKpostmean = np.mean(recon_x,axis=0)
ATPFGKpostmean[ATPFGKpostmean<0] = 0
ATPFGKlowerbound = np.zeros(100)
ATPFGKupperbound = np.zeros(100)
for i in range(100):
    tmp = sorted(recon_x[:,i])
    ATPFGKlowerbound[i] = max([0,tmp[M//40]])
    ATPFGKupperbound[i] = max([0,tmp[M-M//40]])
plt.plot(loc,ATPFGKpostmean,color='#9932CD',label='KATPF(Gauss) mean')
plt.fill_between(loc,ATPFGKlowerbound,ATPFGKupperbound,color='#6B238E',alpha=0.25)
print('KATPF(Gauss) RMSE:',np.sqrt(np.mean((ATPFGKpostmean-rain)**2)))

plt.plot(loc,rain,color='#000000',label='true rain')
plt.plot(loc,obs_rain1**(1/3),'x',color='#000000',label='obs rain')
plt.legend(loc='upper right',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('rainfall.png')
plt.show()
