# meal simulations: normal response, changing c, linear vs nonlinear, check with scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model_and_solver import *

h = 0.05
T = 400.0

# 1) one meal with normal c
t, G, I = meal_sim(c_nom, h, T)
M = []
for tt in t:
    M.append(meal(tt))

kp = np.argmax(G)
km = np.argmin(G)
print("c = 0.04 run")
print("  peak glucose", round(G[kp], 2), "at t =", round(t[kp], 2))
print("  min glucose", round(G[km], 2), "at t =", round(t[km], 2))
print("  peak insulin", round(max(I), 2), "at t =", round(t[np.argmax(I)], 2))
print("  at t = 400: G =", round(G[-1], 4), " I =", round(I[-1], 4))

fig, ax = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
ax[0].plot(t, M, color='orange', linewidth=2)
ax[0].set_ylabel("M(t)  (mg/dL/min)")
ax[0].set_title("(a) Meal impulse: area D = 60 mg/dL delivered over 5 min")
ax[0].grid(alpha=0.3)

ax[1].plot(t, G, 'b', linewidth=2, label="G(t)")
ax[1].axhline(Gb, color='gray', linestyle='--', label="basal $G_b$ = 90")
ax[1].set_ylabel("glucose G (mg/dL)")
ax[1].set_title("(b) Blood glucose")
ax[1].legend()
ax[1].grid(alpha=0.3)

ax[2].plot(t, I, 'r', linewidth=2, label="I(t)")
ax[2].axhline(Ib, color='gray', linestyle='--', label=r"basal $I_b$ = 10")
ax[2].set_ylabel(r"insulin I ($\mu$U/mL)")
ax[2].set_xlabel("time t (min)")
ax[2].set_title("(c) Plasma insulin")
ax[2].legend()
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_meal_response.png", dpi=150)
plt.close()

# 2) different values of c
cvals = [0.004, cs, 0.04, 0.15, 0.4]
labs = ["c = 0.004 (node)", "c = c* = %.4f (critical)" % cs, "c = 0.04 (nominal, spiral)", "c = 0.15 (spiral)", "c = 0.4 (spiral)"]
cols = ['green', 'black', 'blue', 'orange', 'red']

fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for i in range(5):
    t, G, I = meal_sim(cvals[i], h, T)
    ax[0].plot(t, G, color=cols[i], linewidth=1.8, label=labs[i])
    ax[1].plot(t, I, color=cols[i], linewidth=1.8, label=labs[i])
ax[0].axhline(90, color='gray', linestyle='--', linewidth=1)
ax[0].set_ylabel("glucose G (mg/dL)")
ax[0].set_title("(a) Glucose response for different insulin production rates")
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.3)
ax[1].axhline(10, color='gray', linestyle='--', linewidth=1)
ax[1].set_ylabel(r"insulin I ($\mu$U/mL)")
ax[1].set_xlabel("time t (min)")
ax[1].set_title("(b) Insulin response for different insulin production rates")
ax[1].grid(alpha=0.3)
plt.xlim(0, 300)
plt.tight_layout()
plt.savefig("../figures/fig_sensitivity_c.png", dpi=150)
plt.close()

# 3) peak, undershoot, settling time and peak insulin for each c
band = 2.0   # we say glucose is back to normal when |G - Gb| < 2


def settle_time(t, G):
    last = 0.0
    for k in range(len(t)):
        if abs(G[k] - Gb) > band:
            last = t[k]
    return last - t_meal


print("  c, peak rise, undershoot, settling time, peak insulin")
for c in cvals:
    t, G, I = meal_sim(c, h, T)
    u = Gb - min(G)
    if u < 0:
        u = 0
    line = str(round(c, 4)) + ", " + str(round(max(G) - Gb, 2)) + ", " + str(round(u, 2)) + ", " + str(round(settle_time(t, G), 1)) + ", " + str(round(max(I), 2))
    print("  ", line)

# 4) nonlinear vs linearised for small, normal and big meal
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
Dlist = [30.0, 60.0, 150.0]
cols = ['green', 'blue', 'red']
for i in range(3):
    t, G, I = meal_sim(c_nom, h, T, Dm=Dlist[i])
    t, x, y = meal_sim(c_nom, h, T, Dm=Dlist[i], linear=True)
    ax[0].plot(t, G, color=cols[i], linewidth=2, label="nonlinear, D = %d" % Dlist[i])
    ax[0].plot(t, x + Gb, color=cols[i], linewidth=1.5, linestyle='--', label="linearised, D = %d" % Dlist[i])
    ax[1].plot(t, I, color=cols[i], linewidth=2)
    ax[1].plot(t, y + Ib, color=cols[i], linewidth=1.5, linestyle='--')
    print("  D =", Dlist[i], " max |G_nonlinear - G_linear| =", round(max(abs(G - (x + Gb))), 3))
ax[0].set_xlabel("time (min)")
ax[0].set_ylabel("glucose G (mg/dL)")
ax[0].set_title("(a) Glucose: nonlinear (solid) vs linearised (dashed)")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_xlim(0, 250)
ax[1].set_xlabel("time (min)")
ax[1].set_ylabel(r"insulin I ($\mu$U/mL)")
ax[1].set_title("(b) Insulin: nonlinear (solid) vs linearised (dashed)")
ax[1].grid(alpha=0.3)
ax[1].set_xlim(0, 250)
plt.tight_layout()
plt.savefig("../figures/fig_nonlinear_vs_linear.png", dpi=150)
plt.close()

# 5) check our rk4 against scipy


def f_scipy(t, u):
    dG, dI = f(t, u[0], u[1], c_nom, meal(t))
    return [dG, dI]


t, G, I = meal_sim(c_nom, h, T)
sol = solve_ivp(f_scipy, [0, T], [Gb, Ib], t_eval=t, max_step=0.05)
print("  rk4 vs solve_ivp: max diff in G =", max(abs(G - sol.y[0])))

print("simulation figures done")
