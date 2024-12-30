import numpy as np
from scipy.optimize import linprog
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

dim = 1  # dimension of x

A = 1
B = 1
C = 0.1
D = 1
T = 1

y = np.array([0.0,0.0])
M = 10000 # the number of EnKF samples

#initialize particels
pred_x = []
recon_x = np.random.normal(size=M)
delta = 0 # inflation coefficient
post_x = np.zeros(shape=[2,M])

# as time evolves
for t in range(T+1):
    # sample p(x_{t+1}|x_t)
    if t != 0:
        recon_x = op(recon_x)

    # update EnKF particles
    Ct = np.sum((recon_x-np.mean(recon_x))**2)/(len(recon_x)-1)
    Kt = Ct/(Ct+D**2)
    recon_x += (y[t]+np.random.normal(loc=0,scale=D,size=M)-recon_x)*Kt
    recon_x = np.mean(recon_x)+(1+delta)*(recon_x-np.mean(recon_x))
    post_x[t] = recon_x

    # record the best single estimate
    pred_x.append(np.mean(recon_x))

np.save('EnKF.npy',post_x[1])
plt.figure()
Y = np.linspace(-2,4,600)
Z = marginal(Y)
plt.plot(Y,Z/np.sum(Z*0.01))
plt.hist(post_x[1],bins=40,density=True)
plt.title('EnKF',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
