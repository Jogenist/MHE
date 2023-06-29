#mass spring damper system
#m*ddx+b*dx+k*x = 0
#Goal: identification of damping coefficient b
from math import sin
from matplotlib import pyplot as plt
import numpy as np


def mass_spring_damper(x,t):
    m = 7.5 # mass
    k = 50 # spring coefficient
    b = 2 # damping coefficient
    dx1 = x[1]
    dx2 = -k/m*x[0]-b/m*x[1]
    return np.array([dx1,dx2])


def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    print(y0)
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

t = 0
dt = 0.1
SimulationTime = 10
Nsimulation = int(SimulationTime/dt) # Length of simulation
x0 = [0.3,0]

ret = []
for i in range(1,Nsimulation):
    # 1. simulate
    dt_rk = [t,t+dt]
    x0 = rungekutta4(mass_spring_damper,x0,dt_rk)[1]
    print(x0)
    ret.append(x0[0])


plt.plot(ret)
#print(list(list(zip(*ret))[0]))
plt.show()