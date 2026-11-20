# glucose insulin model and rk4 solver
# other files import from here

import numpy as np

# basal (fasting) values
Gb = 90.0    # mg/dL
Ib = 10.0    # uU/mL

# parameters, time is in minutes
a = 0.0015   # insulin dependent uptake
b = 0.005    # insulin independent uptake
d = 0.1      # insulin clearance
c_nom = 0.04 # insulin production rate for a normal person

# these two are chosen so that (Gb, Ib) is the equilibrium
R0 = b*Gb + a*Gb*Ib   # glucose from the liver
s0 = d*Ib             # basal insulin secretion

# meal
D = 60.0       # glucose added by the meal (mg/dL)
t_meal = 20.0  # meal starts here
tau = 5.0      # length of the pulse

# linearised system constants
alpha = b + a*Ib
beta = a*Gb


def meal(t):
    # rectangular pulse with area D, used for plotting
    if t >= t_meal and t < t_meal + tau:
        return D/tau
    return 0.0


def f(t, G, I, c, M, s=s0):
    # nonlinear model. M = meal input, s = basal secretion
    dG = R0 - b*G - a*G*I + M
    sec = s + c*(G - Gb)
    if sec < 0:
        sec = 0   # pancreas cant take insulin out
    dI = sec - d*I
    return dG, dI


def f_lin(t, x, y, c, M):
    # linearised model, x = G - Gb and y = I - Ib
    dx = -alpha*x - beta*y + M
    dy = c*x - d*y
    return dx, dy


def rk4(func, G0, I0, t0, t1, h, c, M, s=s0):
    n = int(round((t1 - t0)/h))
    t = np.zeros(n+1)
    G = np.zeros(n+1)
    I = np.zeros(n+1)
    t[0] = t0
    G[0] = G0
    I[0] = I0

    for i in range(n):
        if func == f:
            k1G, k1I = f(t[i], G[i], I[i], c, M, s)
            k2G, k2I = f(t[i] + h/2, G[i] + h/2*k1G, I[i] + h/2*k1I, c, M, s)
            k3G, k3I = f(t[i] + h/2, G[i] + h/2*k2G, I[i] + h/2*k2I, c, M, s)
            k4G, k4I = f(t[i] + h, G[i] + h*k3G, I[i] + h*k3I, c, M, s)
        else:
            k1G, k1I = f_lin(t[i], G[i], I[i], c, M)
            k2G, k2I = f_lin(t[i] + h/2, G[i] + h/2*k1G, I[i] + h/2*k1I, c, M)
            k3G, k3I = f_lin(t[i] + h/2, G[i] + h/2*k2G, I[i] + h/2*k2I, c, M)
            k4G, k4I = f_lin(t[i] + h, G[i] + h*k3G, I[i] + h*k3I, c, M)

        G[i+1] = G[i] + h/6*(k1G + 2*k2G + 2*k3G + k4G)
        I[i+1] = I[i] + h/6*(k1I + 2*k2I + 2*k3I + k4I)
        t[i+1] = t0 + (i+1)*h

    return t, G, I


def meal_sim(c, h=0.05, T=400.0, Dm=D, s=s0, linear=False):
    # run before / during / after the meal separately
    # if one rk4 step goes over the jump in M the accuracy drops a lot
    if linear:
        func = f_lin
        G0 = 0.0
        I0 = 0.0
    else:
        func = f
        G0 = Gb
        I0 = Ib

    t1, G1, I1 = rk4(func, G0, I0, 0.0, t_meal, h, c, 0.0, s)
    t2, G2, I2 = rk4(func, G1[-1], I1[-1], t_meal, t_meal + tau, h, c, Dm/tau, s)
    t3, G3, I3 = rk4(func, G2[-1], I2[-1], t_meal + tau, T, h, c, 0.0, s)

    t = np.concatenate([t1, t2[1:], t3[1:]])
    G = np.concatenate([G1, G2[1:], G3[1:]])
    I = np.concatenate([I1, I2[1:], I3[1:]])
    return t, G, I


def eig_vals(c, dd=d):
    # roots of lambda^2 - tr*lambda + det = 0
    tr = -(alpha + dd)
    det = alpha*dd + beta*c
    disc = tr**2 - 4*det
    if disc >= 0:
        l1 = (tr + np.sqrt(disc))/2
        l2 = (tr - np.sqrt(disc))/2
    else:
        l1 = complex(tr/2, np.sqrt(-disc)/2)
        l2 = complex(tr/2, -np.sqrt(-disc)/2)
    return tr, det, disc, l1, l2


cs = (alpha - d)**2/(4*beta)   # critical c, disc = 0 here
