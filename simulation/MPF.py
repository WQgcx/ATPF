import numpy as np
import matplotlib.pyplot as plt
np.random.seed(114222)

def op(x,noise=True):
    return x**2+np.log(x**2+1)+noise*C*np.random.normal(size=x.shape)

def obs(x):
    return x + D*np.random.normal(size=x.shape)

def post(x0,x1):
    return np.exp(-(x0**2+(x0-y[0])**2+(x1-op(x0,False))**2/(C**2)+(x1-y[1])**2)/2)

def marginal(x1):
    low_x0 = -10
    up_x0 = 10
    seg = 20000
    delta = np.linspace(low_x0,up_x0,seg)
    length = (up_x0-low_x0)/seg
    return np.array([np.sum(post(delta,i)*length) for i in x1])

def density(x,mean,variance):
    return np.exp(-(x-mean)**2/2/variance)

dim = 1  # dimension of x

A = 1
B = 1
C = 0.1
D = 1
T = 1

# MPF parameter
MPF_n = 3
alpha = np.array([3.0/4, (np.sqrt(13.0) + 1) / 8.0, (1 - np.sqrt(13.0)) / 8.0])

y = np.array([0.0,0.0])

M = 10000 # the number of MPF samples

#initialize particels
pred_x = []
seqMC = np.random.normal(size=M)
delta = 0 # inflation coefficient
post_x = np.zeros(shape=[2,M])
weights = np.array([1/M for i in range(M)])

# as time evolves
cover = 0
for t in range(T+1):
    if t != 0:
        seqMC = op(seqMC)

    weights *= density(y[t],seqMC,D**2)
    weights /= np.sum(weights)
    
    # resample, merge and inflat PF particles
    seqMC_M = np.zeros(seqMC.shape)
    for i in range(MPF_n):
        t_seqMC = np.random.choice(seqMC,M,replace=True,p=weights)
        seqMC_M += alpha[i] * t_seqMC
    seqMC = seqMC_M
    weights = np.array([1/M for i in range(M)])
    
    seqMC += delta * (seqMC-np.mean(seqMC,axis=0))

    post_x[t] = seqMC
    
    # record the best single estimate
    pred_x.append(np.mean(seqMC,axis=0))

np.save('MPF.npy',post_x[1])
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(post_x[1],bins=40,density=True)
plt.title('MPF',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
