import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from numpy import *

c = 3.0e8
M = -19.26
H0 = 72.9

data = pd.read_csv("panth.csv")
z = array(data.z)
m = array(data.m)
m_err = array(data.m_err)

def dl_th(zlst,OmegaM,OmegaL):
    ln = len(zlst)
    dllst = zeros(size(zlst))
    for i in range(ln):
        intgr = quad(lambda xx: 1/sqrt(OmegaM*((1+xx)**3)+OmegaL), 0, zlst[i])[0]
        dllst[i] = ((c/H0)/1000)*(1+zlst[i])*intgr
    return dllst

def m_th(zlst,OmegaM,OmegaL):
    return M + 5.0*log10(dl_th(zlst,OmegaM,OmegaL)/1e-5)

def chi2(OmegaM,OmegaL):
    chi2_lst = ((m_th(z,OmegaM,OmegaL)-m)**2)/(m_err**2)
    return sum(chi2_lst)

def L(OmegaM,OmegaL):
    return exp(-chi2(OmegaM,OmegaL)/2)

OmegaM_lst = np.linspace(0.2,0.4,150)
OmegaL_lst = np.linspace(0.5,1.0,150)
OmegaM_grd, OmegaL_grd = np.meshgrid(OmegaM_lst, OmegaL_lst)


prob = zeros(shape(OmegaM_grd))

from tqdm import tqdm

for j in tqdm(range(len(OmegaL_lst))):
    for i in range(len(OmegaM_lst)):
        prob[j][len(OmegaL_lst)-(i+1)] = L(OmegaM_lst[i],OmegaL_lst[j])

cl68 = np.max(prob)/exp(2.3/2.0)
cl95 = np.max(prob)/exp(6.17/2.0)
p_max = np.max(prob)
x1 = linspace(0.2,0.4,50)
y1 = 1 - x1

fig, ax = plt.subplots(figsize=(10,6))
cpf = ax.contourf(OmegaM_grd, OmegaL_grd, prob, levels = [cl95,cl68,p_max],\
                  colors=['r','g'])
plt.colorbar(cpf,label=r"$P(\Omega_{m},\Omega_{\Lambda})$")
cp = ax.contour(OmegaM_grd, OmegaL_grd, prob, levels = [cl95,cl68],\
                colors=['r','g'])
ax.plot(x1,y1,'--',linewidth=2)
ax.text(0.33,0.643,'Flat Universe',rotation=-10,size=14,weight='bold')
ax.text(0.335,0.73,'Pantheon',rotation=-5,size=13, color = 'midnightblue',\
        style='italic')
plt.xlim(0.28,0.35)
plt.ylim(0.6,0.85)
plt.xlabel("$\Omega_{m}$",fontsize=14)
plt.ylabel("$\Omega_{\Lambda}$",fontsize=14)
plt.title("$\Lambda$CDM Constraints for SN-only Sample",size = 15,weight='bold')
fmt = {}
strs = ['95% CL', '68% CL']
for l, s in zip(cp.levels, strs):
    fmt[l] = s

# Label the contour level using strings
plt.clabel(cp, cp.levels[::1], inline=True, fmt=fmt, colors = 'k')
