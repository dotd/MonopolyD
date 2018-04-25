

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

def compute_general(p,q,f):
    P = np.zeros(shape=(8,8))
    P[0,1] = p
    P[0,2] = 1-p
    P[1,3] = q
    P[1,4] = 1-q
    P[2,5] = q
    P[2,6] = 1-q
    P[3,7] = 1
    P[4,7] = 1
    P[5,7] = 1
    P[6,7] = 1

    R = np.zeros(shape=(8,1))
    R[1] = 1
    R[2] = -1
    R[3] = 1
    R[4] = -1
    R[5] = 1
    R[6] = -1

    J = P @ R + P @ P @ R

    R2 = np.abs(J-R)


def compute_straight(p,q,f):
    J0 = 2*(p+q-1)
    S = p*q*f(2-J0)
    S += (1-p)*q * f(-J0)
    S += p*(1-q) * f(-J0)
    S += (1-p)*(1-q) * f(-2-J0)
    return J0,S

samples = [];
f = lambda x: abs(x)**2
for p in np.linspace(0,1,51):
    for q in np.linspace(0, 1, 51):
        J0,S = compute_straight(p,q,f)
        samples.append([p,q,J0,S])

samples = np.array(samples)
plt.scatter(samples[:,2], samples[:,3])
plt.xlabel("J0")
plt.ylabel("S")
plt.show()

print(samples)