# MHE implementation using casadi 
# identification of damping coefficient b of mass spring damper system
# 21.03.2023 - Jonas Gentner

from scipy import linalg
from matplotlib import pyplot as plt
from casadi.tools import struct_symSX,struct_SX,entry
from casadi import DM,horzcat,Function,mtimes,vertcat,vec,fabs,nlpsol
import numpy as np
import random as rdn
#Make random numbers predictable
np.random.seed(0)
rdn.seed(3)

#N: horizon length
#dt: step width
#Nsimulation: Simulation Time
#R: initial R
#Q: initial Q
#sigma_x0: initial state vector

def init_MHE(N,dt,Nsimulation,R,Q,sigma_x0,initial_values,params):
    #%% initialize MHE
    m = params[0]
    k = params[1]
    #b = params[2]
    # the states
    states = struct_symSX(["dx1","dx2","b"]) # state vector
    Nstates = states.size # Number of states
    # Set up some aliases
    dx1,dx2,b = states[...]
    
    # the control inputs
    controls = struct_symSX(["u"]) # control vector
    Ncontrols = controls.size # Number of control inputs
    # Set up some aliases
    u = controls[...]
    
    # disturbances
    disturbances = struct_symSX(["w","v","z"]) # Process noise vector
    Ndisturbances = disturbances.size # Number of disturbances
    # Set up some aliases
    w,v,z = disturbances[...]
    
    # measurements
    measurements = struct_symSX(["dx1"]) # Measurement vector
    Nmeas = measurements.size # Number of measurements
    # Set up some aliases
    cu = measurements[...]
    
    # create Structure for the entire horizon
    # Structure that will be degrees of freedom for the optimizer
    shooting = struct_symSX([(entry("X",repeat=N,struct=states),entry("W",repeat=N-1,struct=disturbances))])
    # Structure that will be fixed parameters for the optimizer
    parameters = struct_symSX([(entry("U",repeat=N-1,struct=controls),entry("Y",repeat=N,struct=measurements),entry("S",shape=(Nstates,Nstates)),entry("x0",shape=(Nstates,1)))])
    S = parameters["S"]
    x0 = parameters["x0"]
    # define the ODE right hand side
    rhs = struct_SX(states)
    rhs["dx1"] = dx2
    rhs["dx2"] = -k/m*dx1-b/m*dx2
    rhs["b"] = 0

    f = Function('f', [states,controls,disturbances],[rhs])
    
    # build an integrator for this system: Runge Kutta 4 integrator
    k1 = f(states,controls,disturbances)
    k2 = f(states+dt/2.0*k1,controls,disturbances)
    k3 = f(states+dt/2.0*k2,controls,disturbances)
    k4 = f(states+dt*k3,controls,disturbances)
    
    states_1 = states+dt/6.0*(k1+2*k2+2*k3+k4)
    phi = Function('phi', [states,controls,disturbances],[states_1])
    PHI = phi.jacobian_old(0, 0)
    
    measure = struct_SX(measurements)
    measure["dx1"] = dx1

    
    # define the measurement system
    h = Function('h', [states],[measure])      #Kupfertemperatur wird gemessen (hier nur define)
    H = h.jacobian_old(0, 0)
    
    # create a holder for the estimated states and disturbances
    estimated_X= DM.zeros(Nstates,Nsimulation)
    estimated_W = DM.zeros(Ndisturbances,Nsimulation-1)
    
    # build the objective
    obj = 0
    # first the arrival cost
    obj += mtimes([(shooting["X",0]-parameters["x0"]).T,S,(shooting["X",0]-parameters["x0"])])
    #next the cost for the measurement noise
    for i in range(N):
      vm = h(shooting["X",i])-parameters["Y",i]
      obj += mtimes([vm.T,R,vm])
    #and also the cost for the process noise
    for i in range(N-1):
      obj += mtimes([shooting["W",i].T,Q,shooting["W",i]])
    
    # build the multiple shooting constraints
    g = []
    for i in range(N-1):
      g.append( shooting["X",i+1] - phi(shooting["X",i],parameters["U",i],shooting["W",i]) ) #prädizierte Zustände (phi(...)) müssen gleich X(i+1)  sein 
    
    # formulate the NLP
    nlp = {'x':shooting, 'p':parameters, 'f':obj, 'g':vertcat(*g)}
    
    #build the state constraints
    lbw = []
    ubw = []
    for i in range(N):      #how many?
        #    dx1  dx2  b    u   w  z  
        lbw+=[-1, -1,  0,   0,  0, 0]
        ubw+=[ 1,  1,  5,   5,  5, 5]
    
    for i in range(Nstates): #delete the last Nstates elements from constraint list
        lbw.pop(len(lbw)-1) 
        ubw.pop(len(ubw)-1)
        
    #the initial estimate and related covariance, which will be used for the arrival cost
    P = sigma_x0**2*DM.eye(Nstates)
    x0 = DM(initial_values) + sigma_x0*np.random.randn(Nstates,1)
    
    # create the solver
    opts = {"ipopt.print_level":0, "print_time":False, 'ipopt.max_iter':1000}
    #nlpsol = nlpsol("nlpsol", "ipopt", nlp, opts)
    
    ret =  {}
    for elem, name in [(shooting, 'shooting'),
                       (parameters, 'parameters'),
                       (f, 'f'),
                       (phi, 'phi'),
                       (PHI, 'PHI'),
                       (estimated_X, 'estimated_X'),
                       (estimated_W, 'estimated_W'),
                       (lbw, 'lbw'),
                       (ubw, 'ubw'),
                       (P, 'P'),
                       (x0, 'x0'),
                       (nlpsol("nlpsol", "ipopt", nlp, opts), 'nlpsol'),
                       (Nstates, 'Nstates'),
                       (h, 'h'),
                       (H, 'H'),
                       (Ndisturbances, 'Ndisturbances')]:
        ret[name]=elem
    return ret

def mass_spring_damper(x,t):
    m = 7.5 # mass
    k = 50 # spring coefficient
    b = 2.5 # damping coefficient
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


# Settings of the filter
N = 10 # Horizon length MHE
dt = 0.1; # Time step
SimulationTime = 10 #sekunden

sigma_p = 0.005 # Standard deviation of the measurements
sigma_w = 1     # Standard deviation for the process noise
sigma_x0 = 1  # Standard deviation for the initial values


r = [40000]
R = np.diag(r)          #measurement noise matrix

sensor_noise = False

q = [0.001,0.001,0.001]
Q = np.diag(q)          #process noise matrix

Nsimulation = int(SimulationTime/dt) # Length of simulation
t = np.linspace(0,(Nsimulation-1)*dt,Nsimulation) # Time grid

initial_values = [0.3,0,1]
params = [7.5,50]

#%%init MHE
MHE = init_MHE(N,dt,Nsimulation,R,Q,sigma_x0,initial_values,params)
shooting = MHE['shooting']
parameters = MHE['parameters']
phi = MHE['phi']
PHI = MHE['PHI']
estimated_X = MHE['estimated_X']
estimated_W = MHE['estimated_W']
lbw = MHE['lbw']
ubw = MHE['ubw']
P = MHE['P']
x0 = MHE['x0']
nlpsol = MHE['nlpsol']
Nstates = MHE['Nstates']
h = MHE['h']
H = MHE['H']
Ndisturbances = MHE['Ndisturbances'] 

start_values = [0.3,0]
meas = [0]      #vector for measurement of dx1
meas2 = [0]     #vector for measurement of dx2 -> not used in MHE

dx1_res = [0]
dx2_res = [0]
b_res = [0]

for i in range(1,Nsimulation):
    print('Durchgang: ',i,'/',Nsimulation)
    # 1. simulate
    dt_rk = [0,0+dt]
    start_values = rungekutta4(mass_spring_damper,start_values,dt_rk)[1]
    print(start_values)
    # 2. measurement

    if sensor_noise:
        noise = rdn.uniform(0.01,-0.01)
        val = start_values[0] + noise

        meas.append(val)    #measurement of dx1
        meas2.append(start_values[1])   #measurement of dx2 -> not used in MHE
    else:
        meas.append(start_values[0])
        meas2.append(start_values[1])

    # 3. MHE
    if(i>=(N-1)):
        if(i==(N-1)):
            # for the first instance we run the filter, we need to initialize it.
            current_parameters = parameters(0)
            current_parameters["U",lambda x: horzcat(*x)] =  DM([l*2 for l in meas[0:N-1]])
            current_parameters["Y",lambda theta: horzcat(*theta)] = DM([meas]) #hier ersten gemessenen Horizont (erstes Messfenster)
            current_parameters["S"] = linalg.inv(P) # arrival cost is the inverse of the initial covariance
            current_parameters["x0"] = x0
            initialisation_state = shooting(0)
            Paramlist = [initial_values[1] for i in range(N)]
            Paramlist2 = [initial_values[2] for i in range(N)]
            initialisation_state["X",lambda x: horzcat(*x)] = DM([meas,Paramlist,Paramlist2]) #hier der erste simulierte Horizont unbekannte Zustände einfach als Startwert 
            res = nlpsol(p=current_parameters, x0=initialisation_state, lbg=0, ubg=0)
            
            # Get the solution
            solution = shooting(res["x"])
  
            #calculation of measurement cost
            b = 0
            for k in range(N):
                a = h(solution["X",k])-current_parameters["Y",k]
                b += (mtimes([a.T,R,a])).full()[0][0]
            #measurement_cost.append(b)
            
            #calculation of process cost
            for k in range(N-1):
                a += mtimes([solution["W",k].T,Q,solution["W",k]]).full()[0][0]
           # process_cost.append(a.full()[0][0])
            
            estimated_X[:,0:N] = solution["X",lambda x: horzcat(*x)]
            estimated_W[:,0:N-1] = solution["W",lambda x: horzcat(*x)]
        
        if(i>(N-1) and i<Nsimulation-(N-2)):
            # Now make a loop for the rest of the simulation
        
            # update the arrival cost, using linearisations around the estimate of MHE at the beginning of the horizon (according to the 'Smoothed EKF Update'):
            # first update the state and covariance with the measurement that will be deleted, and next propagate the state and covariance because of the shifting of the horizon
            print("step %d/%d (%s)" % (i-N, Nsimulation-N , nlpsol.stats()["return_status"]))
            H0 = H(solution["X",0])[0]
            K = mtimes([P,H0.T,linalg.inv(mtimes([H0,P,H0.T])+R)])
           # Gain_lst.append(K)
            P = mtimes((DM.eye(Nstates)-mtimes(K,H0)),P)
            h0 = h(solution["X",0])
            x0 = x0 + mtimes(K, current_parameters["Y",0]-h0-mtimes(H0,x0-solution["X",0]))
            x0 = phi(x0, current_parameters["U",0], solution["W",0])
            F = PHI(solution["X",0], current_parameters["U",0], solution["W",0])[0]
            P = mtimes([F,P,F.T]) + linalg.inv(Q)
            # Get the measurements and control inputs
            current_parameters["U",lambda x: horzcat(*x)] =  DM([l*2 for l in meas[0:N-1]])#simulated_U[i-N:i-1]
            current_parameters["Y",lambda x: horzcat(*x)] = DM([meas[i-N:i]])
            current_parameters["S"] = linalg.inv(P)
            current_parameters["x0"] = x0
            # Initialize the system with the shifted solution
            initialisation_state["W",lambda x: horzcat(*x),0:N-2] = estimated_W[:,i-N:i-2] # The shifted solution for the disturbances
            initialisation_state["W",N-2] = DM.zeros(Ndisturbances,1) # The last node for the disturbances is initialized with zeros
            initialisation_state["X",lambda x: horzcat(*x),0:N-1] = estimated_X[:,i-N:i-1] # The shifted solution for the state estimates
            # The last node for the state is initialized with a forward simulation
            phi0 = phi(initialisation_state["X",N-1], current_parameters["U",-1], initialisation_state["W",-1])
            initialisation_state["X",N-1] = phi0
            # And now initialize the solver and solve the problem
            res = nlpsol(p=current_parameters, x0=initialisation_state,lbx=lbw, ubx=ubw, lbg=0, ubg=0)
            solution = shooting(res["x"])
     
            #calculation of measurement cost
            b = 0
            for k in range(N):
                a = h(solution["X",k])-current_parameters["Y",k]
                b += (mtimes([a.T,R,a])).full()[0][0]
    
            
            #calculation of process cost
            for k in range(N-1):
                a += mtimes([solution["W",k].T,Q,solution["W",k]]).full()[0][0]
  
            
            # Now get the state estimate. Note that we are only interested in the last node of the horizon
            estimated_X[:,N-1+i-N] = solution["X",lambda x: horzcat(*x)][:,N-1]
            estimated_W[:,N-2+i-N] = solution["W",lambda x: horzcat(*x)][:,N-2]
            
            #get the solution
            dx1 = solution["X",lambda x: horzcat(*x)][:,N-1][0].full()[0][0]
            dx2 = solution["X",lambda x: horzcat(*x)][:,N-1][1].full()[0][0]
            b = solution["X",lambda x: horzcat(*x)][:,N-1][2].full()[0][0]

            dx1_res.append(dx1)
            dx2_res.append(dx2)
            b_res.append(b)
    else:
        dx1_res.append(0)
        dx2_res.append(0)
        b_res.append(0)

#print(len(meas))
#print(len(meas2))
b_real = [2.5 for i in range(Nsimulation)]
plt.plot(meas)
plt.plot(dx1_res)
plt.show()
plt.plot(b_res)
plt.plot(b_real)
plt.show()
print(b_res)