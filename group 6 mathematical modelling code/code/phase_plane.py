# phase plane plots (G vs I)

import numpy as np
import matplotlib.pyplot as plt
from model_and_solver import *


def lin_solution(c, x0, y0, tlist):
    # X(t) = V exp(L t) V^-1 X0
    J = np.array([[-alpha, -beta], [c, -d]])
    lam, V = np.linalg.eig(J)
    Vi = np.linalg.inv(V)
    xs = []
    ys = []
    for t in tlist:
        E = np.array([[np.exp(lam[0]*t), 0], [0, np.exp(lam[1]*t)]])
        X = np.dot(V, np.dot(E, np.dot(Vi, np.array([x0, y0]))))
        xs.append(np.real(X[0]))
        ys.append(np.real(X[1]))
    return np.array(xs), np.array(ys)


# 1) linear system, node vs spiral
tlist = np.linspace(0, 300, 1500)
starts = []
for ang in np.linspace(0, 2*np.pi, 13)[:-1]:
    starts.append([50*np.cos(ang), 25*np.sin(ang)])

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
cc = [0.004, 0.04]
titles = ["(a) c = 0.004 < c*: stable NODE (real eigenvalues)", "(b) c = 0.04 > c*: stable SPIRAL (complex eigenvalues)"]
for i in range(2):
    for p0 in starts:
        x, y = lin_solution(cc[i], p0[0], p0[1], tlist)
        ax[i].plot(x, y, 'b', linewidth=1.2)
        ax[i].plot(p0[0], p0[1], 'bo', markersize=4)

    if i == 0:
        # eigenvectors are straight line solutions for the node
        J = np.array([[-alpha, -beta], [cc[i], -d]])
        lam, V = np.linalg.eig(J)
        lam = np.real(lam)
        V = np.real(V)
        s = np.linspace(-80, 80, 10)
        for j in range(2):
            vx = V[0, j]/np.sqrt(V[0, j]**2 + V[1, j]**2)
            vy = V[1, j]/np.sqrt(V[0, j]**2 + V[1, j]**2)
            if abs(lam[j]) < 0.05:
                ax[i].plot(s*vx, s*vy, 'r--', linewidth=2.5, label=r"slow eigenvector, $\lambda$ = %.4f" % lam[j])
            else:
                ax[i].plot(s*vx, s*vy, 'g--', linewidth=2.5, label=r"fast eigenvector, $\lambda$ = %.4f" % lam[j])
        ax[i].legend(loc='lower right')

    ax[i].plot(0, 0, 'k*', markersize=14)
    ax[i].set_xlim(-60, 60)
    ax[i].set_ylim(-30, 30)
    ax[i].axhline(0, color='gray', linewidth=0.7)
    ax[i].axvline(0, color='gray', linewidth=0.7)
    ax[i].set_xlabel("x = G - Gb (mg/dL)")
    ax[i].set_ylabel(r"y = I - Ib ($\mu$U/mL)")
    ax[i].set_title(titles[i])
    ax[i].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_phase_linear_parametric.png", dpi=150)
plt.close()

# 2) nonlinear phase portraits
cc = [0.004, cs, 0.04, 0.15]
titles = ["(a) c = 0.004 (node)", "(b) c = c* = %.4f (critical)" % cs, "(c) c = 0.04 (nominal, spiral)", "(d) c = 0.15 (spiral)"]
Gax = np.linspace(50, 170, 25)
Iax = np.linspace(0, 45, 25)
starts = [[160, 10], [160, 30], [140, 2], [130, 42], [110, 2], [60, 5],
          [60, 25], [60, 42], [100, 42], [80, 2], [160, 18], [70, 40]]

fig, ax = plt.subplots(2, 2, figsize=(13, 11))
for n in range(4):
    c = cc[n]
    r = n//2
    k = n % 2

    # direction field, arrows all same length
    U = np.zeros((len(Iax), len(Gax)))
    W = np.zeros((len(Iax), len(Gax)))
    for i in range(len(Iax)):
        for j in range(len(Gax)):
            dG, dI = f(0, Gax[j], Iax[i], c, 0.0)
            dG = dG/120
            dI = dI/45
            L = np.sqrt(dG**2 + dI**2)
            U[i, j] = dG/L
            W[i, j] = dI/L
    ax[r, k].quiver(Gax, Iax, U, W, color='lightgray', scale=35, width=0.003)

    # nullclines
    Gl = np.linspace(50, 170, 300)
    Gnull = (R0 - b*Gl)/(a*Gl)
    Inull = []
    for g in Gl:
        sec = s0 + c*(g - Gb)
        if sec < 0:
            sec = 0
        Inull.append(sec/d)
    ax[r, k].plot(Gl, Gnull, color='orange', linewidth=2, label="G-nullcline (dG/dt = 0)")
    ax[r, k].plot(Gl, Inull, color='green', linewidth=2, label="I-nullcline (dI/dt = 0)")

    for p0 in starts:
        t, G, I = rk4(f, p0[0], p0[1], 0.0, 400.0, 0.1, c, 0.0)
        ax[r, k].plot(G, I, 'b', linewidth=1.2)
        ax[r, k].plot(p0[0], p0[1], 'bo', markersize=4)

    ax[r, k].plot(Gb, Ib, 'k*', markersize=15, label="equilibrium (90, 10)")
    ax[r, k].set_xlim(50, 170)
    ax[r, k].set_ylim(0, 45)
    ax[r, k].set_xlabel("glucose G (mg/dL)")
    ax[r, k].set_ylabel(r"insulin I ($\mu$U/mL)")
    ax[r, k].set_title(titles[n])
    if n == 0:
        ax[r, k].legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig("../figures/fig_phase_nonlinear.png", dpi=150)
plt.close()

# 3) the meal runs drawn in the phase plane
cc = [0.004, cs, 0.04, 0.15, 0.4]
cols = ['green', 'black', 'blue', 'orange', 'red']
labs = ["c = 0.004", "c = c*", "c = 0.04", "c = 0.15", "c = 0.4"]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for i in range(5):
    t, G, I = meal_sim(cc[i], 0.05, 400.0)
    ax[0].plot(G, I, color=cols[i], linewidth=1.8, label=labs[i])
    ax[1].plot(G - Gb, I - Ib, color=cols[i], linewidth=1.8, label=labs[i])
    kend = int(round((t_meal + tau)/0.05))
    ax[0].plot(G[kend], I[kend], 'o', color=cols[i])

ax[0].plot(90, 10, 'k*', markersize=14)
ax[0].set_xlabel("glucose G (mg/dL)")
ax[0].set_ylabel(r"insulin I ($\mu$U/mL)")
ax[0].set_title("(a) Meal trajectories (dot = end of meal)")
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].plot(0, 0, 'k*', markersize=14)
ax[1].set_xlim(-12, 12)
ax[1].set_ylim(-10, 10)
ax[1].axhline(0, color='gray', linewidth=0.7)
ax[1].axvline(0, color='gray', linewidth=0.7)
ax[1].set_xlabel("x = G - Gb (mg/dL)")
ax[1].set_ylabel(r"y = I - Ib ($\mu$U/mL)")
ax[1].set_title("(b) Zoom near the equilibrium")
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_phase_meal.png", dpi=150)
plt.close()

print("phase plane figures done")
