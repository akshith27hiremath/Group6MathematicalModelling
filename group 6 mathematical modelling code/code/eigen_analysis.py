# eigenvalue analysis of the linearised system

import numpy as np
import matplotlib.pyplot as plt
from model_and_solver import *

print("alpha =", alpha)
print("beta  =", beta)
print("d     =", d)
print("critical c* =", cs)
print()

# table for the report
clist = [0.004, cs, 0.04, 0.15, 0.4]
for c in clist:
    tr, det, disc, l1, l2 = eig_vals(c)
    J = np.array([[-alpha, -beta], [c, -d]])
    zeta = -tr/(2*np.sqrt(det))

    if disc > 1e-12:
        typ = "node"
        T = 0
    elif disc < -1e-12:
        typ = "spiral"
        w = np.sqrt(-disc)/2
        T = 2*np.pi/w
    else:
        typ = "critical (repeated root)"
        T = 0

    print("c =", round(c, 5))
    print("   trace =", round(tr, 5), " det =", round(det, 6), " disc =", round(disc, 6))
    print("   eigenvalues from formula:", l1, l2)
    print("   eigenvalues from numpy:  ", np.linalg.eigvals(J))
    print("  ", typ, " zeta =", round(zeta, 4), " period =", round(T, 1))

# eigenvalue locus, c from 0 to 0.5
cc = np.linspace(0, 0.5, 20001)
re1 = []
im1 = []
re2 = []
im2 = []
for c in cc:
    tr, det, disc, l1, l2 = eig_vals(c)
    re1.append(np.real(l1))
    im1.append(np.imag(l1))
    re2.append(np.real(l2))
    im2.append(np.imag(l2))

plt.figure(figsize=(7, 5.5))
plt.plot(re1, im1, 'b', linewidth=2, label=r"$\lambda_1$")
plt.plot(re2, im2, 'r', linewidth=2, label=r"$\lambda_2$")
plt.plot([-alpha, -d], [0, 0], 'ko', label="c = 0")
plt.plot(-(alpha + d)/2, 0, 's', color='green', label=r"$c = c^* = %.4f$ (repeated root)" % cs)
tr, det, disc, l1, l2 = eig_vals(0.04)
plt.plot(np.real(l1), np.imag(l1), '^', color='purple', label="c = 0.04 (nominal)")
plt.plot(np.real(l2), np.imag(l2), '^', color='purple')
plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.xlabel(r"Re($\lambda$)  (1/min)")
plt.ylabel(r"Im($\lambda$)  (1/min)")
plt.title("Eigenvalue locus as the insulin production rate c increases")
plt.xlim(-0.11, 0.01)
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_eigen_locus.png", dpi=150)
plt.close()

# real part, frequency and damping ratio vs c
cc = np.linspace(0, 0.5, 2001)
slow = []
fast = []
freq = []
z = []
for c in cc:
    tr, det, disc, l1, l2 = eig_vals(c)
    slow.append(max(np.real(l1), np.real(l2)))
    fast.append(min(np.real(l1), np.real(l2)))
    freq.append(abs(np.imag(l1)))
    z.append(-tr/(2*np.sqrt(det)))

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].plot(cc, slow, 'b', linewidth=2, label=r"max Re($\lambda$)")
ax[0].plot(cc, fast, 'r--', linewidth=2, label=r"min Re($\lambda$)")
ax[0].axvline(cs, color='green', linestyle=':', linewidth=2, label=r"$c^*$")
ax[0].set_xlabel("c")
ax[0].set_ylabel(r"Re($\lambda$) (1/min)")
ax[0].set_title("(a) Real parts (decay rates)")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(cc, freq, color='purple', linewidth=2)
ax[1].axvline(cs, color='green', linestyle=':', linewidth=2)
ax[1].set_xlabel("c")
ax[1].set_ylabel(r"$\omega$ = |Im($\lambda$)| (rad/min)")
ax[1].set_title("(b) Oscillation frequency")
ax[1].grid(alpha=0.3)

ax[2].plot(cc, z, 'k', linewidth=2)
ax[2].axhline(1.0, color='gray', linestyle='--')
ax[2].axvline(cs, color='green', linestyle=':', linewidth=2)
ax[2].set_xlabel("c")
ax[2].set_ylabel(r"damping ratio $\zeta$")
ax[2].set_title(r"(c) Damping ratio $\zeta$")
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/fig_eigen_vs_c.png", dpi=150)
plt.close()

print("eigen figures done")
