"""
symulacja mikromagnetyczna: dynamika pojedynczego makrospinu
opisana rownaniem Landaua-Lifshitza-Gilberta (LLG).

"""

import numpy as np
import matplotlib.pyplot as plt

#Parametry fizyczne (jednostki znormalizowane) 
alpha = 0.1                          # tlumienie Gilberta (bezwymiarowe)
gamma = 1.0                          # wspolczynnik zyromagnetyczny (znorm.)
H_eff = np.array([0.0, 0.0, 1.0])    # pole efektywne wzdluz osi z

#Warunek poczatkowy: namagnesowanie odchylone od pola 
theta0 = np.deg2rad(170)             # kat poczatkowy wzgledem osi z
m = np.array([np.sin(theta0), 0.0, np.cos(theta0)])
m = m / np.linalg.norm(m)            # |m| = 1 (wektor jednostkowy)


def llg(m, H):
    r"""Prawa strona rownania LLG w jawnej postaci Landaua-Lifshitza.

    """
    precesja = np.cross(m, H)
    tlumienie = alpha * np.cross(m, np.cross(m, H))
    return -gamma / (1 + alpha**2) * (precesja + tlumienie)


#Calkowanie w czasie metoda Rungego-Kutty 4. rzedu 
dt = 0.01
n_steps = 4000
t = np.linspace(0, n_steps * dt, n_steps)
M = np.zeros((n_steps, 3))           # historia wektora namagnesowania

for i in range(n_steps):
    M[i] = m
    k1 = llg(m, H_eff)
    k2 = llg(m + 0.5 * dt * k1, H_eff)
    k3 = llg(m + 0.5 * dt * k2, H_eff)
    k4 = llg(m + dt * k3, H_eff)
    m = m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    m = m / np.linalg.norm(m)        # numeryczna korekta: |m| = 1

#Wizualizacja: dynamika namagnesowania w czasie
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t, M[:, 0], label=r'$m_x$', lw=1.5)
ax.plot(t, M[:, 1], label=r'$m_y$', lw=1.5)
ax.plot(t, M[:, 2], label=r'$m_z$', lw=1.5)
ax.set_xlabel('czas (jedn. znormalizowane)')
ax.set_ylabel('namagnesowanie')
ax.set_title('Dynamika namagnesowania (rownanie LLG)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('llg_simulation.png', dpi=130)
print('Zapisano wykres: llg_simulation.png')
print(f'Stan koncowy m = {M[-1].round(3)}  (powinien dazyc do [0, 0, 1])')