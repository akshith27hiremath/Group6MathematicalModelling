# extension: type 1 diabetes
# no insulin from the pancreas so s0 = 0 and c = 0

import numpy as np
import matplotlib.pyplot as plt
from model_and_solver import *

h = 0.05

# new equilibrium when there is no insulin
Gstar = R0/b
Istar = 0.0
J = np.array([[-b - a*Istar, -a*Gstar], [0.0, -d]])
print("Type-1 equilibrium: G* =", Gstar, " I* =", Istar)
print("Jacobian:")
print(J)
print("eigenvalues:", np.linalg.eigvals(J))
print("1/b =", 1/b, "min")

# 1) one meal, healthy vs type 1
fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
t, G, I = meal_sim(c_nom, h, 1200.0)
ax[0].plot(t, G, 'b', linewidth=2, label="healthy (c = 0.04)")
ax[1].plot(t, I, 'b', linewidth=2, label="healthy (c = 0.04)")
print("healthy  G at 1200 =", round(G[-1], 2), " max G =", round(max(G), 2), " I at 1200 =", round(I[-1], 4))

t, G, I = meal_sim(0.0, h, 1200.0, s=0.0)
ax[0].plot(t, G, 'r', linewidth=2, label="Type-1 (s0 = 0, c = 0)")
ax[1].plot(t, I, 'r', linewidth=2, label="Type-1 (s0 = 0, c = 0)")
print("type 1   G at 1200 =", round(G[-1], 2), " max G =", round(max(G), 2), " I at 1200 =", round(I[-1], 4))
for k in range(len(t)):
    if G[k] > 180:
        print("   type 1 glucose goes above 180 at t =", round(t[k], 1))
        break

ax[0].axhline(360, color='r', linestyle=':', label=r"Type-1 equilibrium $G^* = R_0/b$ = 360")
ax[0].axhline(180, color='gray', linestyle='--', label="renal threshold 180 mg/dL")
ax[0].axhline(90, color='b', linestyle=':', label="healthy basal 90 mg/dL")
ax[0].set_ylabel("glucose G (mg/dL)")
ax[0].set_title("(a) Glucose after one meal: healthy vs Type-1 diabetes")
ax[0].legend(loc='center right')
ax[0].grid(alpha=0.3)
ax[1].set_ylabel(r"insulin I ($\mu$U/mL)")
ax[1].set_xlabel("time (min)")
ax[1].set_title("(b) Insulin: no production in Type-1, insulin is cleared to zero")
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_type1_single_meal.png", dpi=150)
plt.close()

# 2) phase plane for type 1
Gax = np.linspace(50, 420, 24)
Iax = np.linspace(0, 30, 20)
U = np.zeros((len(Iax), len(Gax)))
W = np.zeros((len(Iax), len(Gax)))
for i in range(len(Iax)):
    for j in range(len(Gax)):
        dG, dI = f(0, Gax[j], Iax[i], 0.0, 0.0, 0.0)
        dG = dG/370
        dI = dI/30
        L = np.sqrt(dG**2 + dI**2)
        U[i, j] = dG/L
        W[i, j] = dI/L

plt.figure(figsize=(9, 6))
plt.quiver(Gax, Iax, U, W, color='lightgray', scale=35, width=0.003)
starts = [[90, 10], [150, 10], [60, 25], [200, 28], [400, 20], [400, 2], [300, 28], [120, 0.5]]
for p0 in starts:
    t, G, I = rk4(f, p0[0], p0[1], 0.0, 1500.0, 0.1, 0.0, 0.0, 0.0)
    plt.plot(G, I, 'r', linewidth=1.5)
    plt.plot(p0[0], p0[1], 'ro', markersize=5)
plt.plot(Gstar, 0, 'r*', markersize=18, label="Type-1 equilibrium (360, 0)")
plt.plot(90, 10, 'b*', markersize=18, label="healthy equilibrium (90, 10) - lost")
Gl = np.linspace(50, 420, 300)
plt.plot(Gl, (R0 - b*Gl)/(a*Gl), color='orange', linewidth=2, label="G-nullcline")
plt.axhline(0, color='green', linewidth=3, label="I-nullcline (I = 0)")
plt.xlim(50, 420)
plt.ylim(-1, 30)
plt.xlabel("glucose G (mg/dL)")
plt.ylabel(r"insulin I ($\mu$U/mL)")
plt.title("Phase plane for Type-1 diabetes: every trajectory goes to high glucose")
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_type1_phase.png", dpi=150)
plt.close()

print("type 1 figures done")
