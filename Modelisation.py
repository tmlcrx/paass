import random as rd
import numpy as np

def kronecker(i,j):
    if i==j:
        return 1
    return 0



# I : Cas où notre matrice stochastique ne dépend pas du temps:

n = 3
x0 = np.array([[rd.random()] for i in range(n)])

def mat_stoch(n): 
    mat = [[rd.random() for k in range(n)] for k in range(n)]

    sum = 0
    for i in range(n) :
        
        sum = 0
        for j in range(n):
            sum += mat[i][j]
        
        for j in range(n):
            mat[i][j] = mat[i][j]/sum
    return np.matrix(mat)

def exp_rap(m,t):

    if t == 1 :
        return m
    if t%2 == 0 :
        return exp_rap(m*m,t//2)
    return m*exp_rap(m*m,(t-1)//2)

x0 = np.array([rd.random() for i in range(n)])
A = mat_stoch(n)

def x(t):
    if t == 0:
        return x0
    return np.dot(exp_rap(A,t),x0)

print(x0)
x(2)

# II : Modèle de Friedkin-Jhonson:

# a):

x0 = np.array([[rd.random()] for i in range(n)])
A = mat_stoch(n)
G = mat_stoch(n)
In = np.eye(n)

np.dot(A,x0)

def x(t):
    if t == 0:
        return x0
    return np.dot(G,x0) + np.dot(In-G,np.dot(A,x(t-1)))

x(1)

# b) Cas où G = diag(A):

x0 = np.array([[rd.random()] for i in range(n)])
In = np.eye(n)
A = mat_stoch(n)
eigen = np.linalg.eig(A)[0]
G = np.matrix([[kronecker(i,j)*eigen[i] for i in range(n)]for j in range(n)])

def x(t):
    if t == 0:
        return x0
    return np.dot(G,x0) + np.dot(In-G,np.dot(A,x(t-1)))

# III Time-variant model:

# IV Opinion dynamics with bounded confidence (BC):

n = 3
x0 = np.array([[rd.random()] for i in range(n)])
eps = [1 for i in range(n)]

def I(i,x):
    res = []
    for j in range(n):
        if abs(x[i][0]-x[j][0]) <= eps[i]:
            res += [j]
    return res

def xBC(t):
    if t == 0:
        return x0
    res = [[] for i in range(n)]
    for i in range(n):
        res[i] = [xiBC(i,t)]
    return np.array(res)

def xiBC(i,t):
    if t == 0:
        return x0
    
    x2 = xBC(t-1)
    ens = I(i,x2)
    m = len(ens)
    if m == 0:
        return 0
    coeff = 1/m
    sum = 0
    for j in ens:
        sum += x2[j][0]
    return sum*(1/m)

xBC(0)
xBC(1)
xBC(15)
## Simulations:

# On utilise le modèle avec BC

#1) Cas symétrique:

n= 625
x0 = np.array([[rd.random()] for i in range(n)])

#1:
epsl = 0.01
epsr = 0.01

x = [x(i,15) for i in range(n)]
eps = [rd.uniform(epsl,epsr) for i in range(n)]

#2:

epsl = 0.15
epsr = 0.15

x = [x(i,15) for i in range(n)]
eps = [rd.uniform(epsl,epsr) for i in range(n)]

#2:

epsl = 0.25
epsr = 0.25

x = [x(i,15) for i in range(n)]
eps = [rd.uniform(epsl,epsr) for i in range(n)]

#3:

n= 3
x0 = np.array([[rd.random()] for i in range(n)])

epsl = 0
epsr = 0

for i in range(40):
    epsl += 0.01
    eps = [epsl for i in range(n)]
    x = xBC(15)
    print(x)


#2) Cas assymétrique:

n = 5
x0 = sorted(np.array([[rd.random()] for i in range(n)]), key=lambda x: x[0])
print(x0)
eps = [(0,0) for k in range(n)]

def I2(i,x):
    res = []
    for j in range(n):
        if ((-1)*eps[i][0] <= abs(x[i][0]-x[j][0]) <= eps[i][1]):
            res += [j]
    return res

def xBC2(t):
    if t == 0:
        return x0
    res = [[] for i in range(n)]
    for i in range(n):
        res[i] = [xiBC2(i,t)]
    return np.array(res)

def xiBC2(i,t):
    if t == 0:
        return x0
    
    x2 = xBC2(t-1)
    ens = I2(i,x2)
    m = len(ens)
    if m == 0:
        return 0
    coeff = 1/m
    sum = 0
    for j in ens:
        sum += x2[j][0]
    return sum*(1/m)

m = 1

def f(x) :
    return m*x+(1-m)/2

def beta_r(x):
    return m*x+(1-m)/2

def beta_l(x): #Def inutile mais bon...
    return 1-beta_r(x)
