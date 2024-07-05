import numpy as np
import matplotlib.pyplot as plt

from NeNi_Gummel import NeNi_Gummel
from fuBetta import fuBetta
from fuBettaP import fuBettaP
from kex1 import kex1
from ki1 import ki1
from ksi1 import ksi1
from urpuas1 import urpuas1

# Загрузка данных
NiP = np.loadtxt('NiP.txt')
E = np.loadtxt('E.txt')
TaP = np.loadtxt('TaP.txt')
TeP = np.loadtxt('TeP.txt')
NeP = np.loadtxt('NeP.txt')
NmP = np.loadtxt('NmP.txt')
nx = int(np.loadtxt('nx.txt'))
nt = int(np.loadtxt('nt.txt'))
Va = float(np.loadtxt('Va.txt'))
D = float(np.loadtxt('D.txt'))

# Глобальные переменные
kB = 1.381e-23
e = 1.6e-19
gamma = 0.01
e00 = 8.85e-12
R21 = 6.2e-16

# Физические параметры
P = 13.3
Fg = 13.56e6
Ta0 = 300

Max_Per = 1000
K_Per_Safe = 100
nt_sloy = 10000
step_sloy_print = 10

Ni0 = 9e15
Nm0 = 1e15
Te0 = 34800

me = 9.1e-31
mAr = 1.66e-27 * 40

# Расчетные константы
omega = 2 * np.pi * Fg
Num = 5.3e9 * (P / 133) * 273 / 300

Mue = e / Num / me
Mui = e / Num / (mAr / 2)
T_per = 1 / Fg

De_coef = kB * Mue / e
Di_coef = kB * Mui / e

# Сетка по пространственной переменной
xk = D
x0 = 0

xh = (xk - x0) / nx
x = np.linspace(x0, xk, nx + 1)
x1 = np.linspace(x0 + xh / 2, xk - xh / 2, nx)

# Сетка по временной переменной
t0 = 0
tk = 2 * np.pi / omega
th = (tk - t0) / nt

# Начальные приближения
NeP0 = NeP.copy()
Gamma_e = np.zeros(nx)
Gamma_i = np.zeros(nx)

# Е в половинных целых
E0 = E.copy()

# Усредненные параметры
dg1 = np.zeros(nx)
sE1 = np.zeros(nx)
sNe1 = np.zeros(nx)
sTe = np.zeros(nx)
sTa = np.zeros(nx)

# Температура постоянная
R31 = kex1(TeP)
R11 = ksi1(TeP)

Te1 = TeP[0]
Ta1 = TaP[0]

for i in range(nx):
    Ta2 = TaP[i + 1]
    Te2 = TeP[i + 1]

    sTe[i] = (Te1 + Te2) / 2
    sTa[i] = (Ta1 + Ta2) / 2

    Te1 = Te2
    Ta1 = Ta2

# Расчет Na на этом временном слое
Na = np.zeros(nx + 1)
for i in range(nx + 1):
    T1 = TaP[i] * kB
    Na[i] = (P - kB * TeP[i] * NeP[i] - T1 * NiP[i] - T1 * NmP[i]) / T1

NaP = Na.copy()

# Коэффициенты диффузии в средних узлах
De = De_coef * sTe
Di = Di_coef * sTa

# Инициализация цикла
w = 1
k = 1
k1 = 0
sloy_print = 0
t_cur = 0
i_per_safe = 0

# Цикл по периодам
for period in range(1, Max_Per + 1):
    t_per = 0
    i_per_safe += 1

    while t_per <= T_per:
        t_cur += th
        t_per += th
        segm = t_per / T_per

        sloy_print += 1

        Nui = ki1(TeP)
        Betta = fuBetta(TeP, TaP, NeP, nx)
        BettaP = fuBettaP(TeP, TaP, NeP, nx)

        fm = Nui * NeP - Betta * NiP * NeP + R21 * NmP**2 + R11 * NmP * NeP

        Ne, Ni, Gamma_e, Gamma_i = NeNi_Gummel(NeP, NiP, De, Di, Mue, Mui, E, gamma, fm, xh, th)

        s = (e / e00) * (Ni - Ne)
        fik = Va * np.cos(omega * t_cur)
        fi = urpuas1(xh, s, 0, fik)

        Ep = E.copy()
        for i in range(nx):
            E[i] = -(fi[i + 1] - fi[i]) / xh

        J_sm = e00 * (E - E0) / th
        E0 = E.copy()

        J_el = -e * Gamma_e
        J_i = e * Gamma_i
        J_poln = J_sm + J_el + J_i
        delta_J = np.max(J_poln) - np.min(J_poln)

        if sloy_print == step_sloy_print:
            sloy_print = 0

            plt.figure(1)

            plt.subplot(3, 2, 1)
            plt.semilogy(x, Ne, x, Ni)
            plt.xlabel('x, m')
            plt.ylabel('N_e, N_i, m^-3')
            plt.title(f'Period={period}, Segment={segm:.2f}')

            plt.subplot(3, 2, 2)
            plt.plot(x, fi)
            plt.xlabel('x, m')
            plt.ylabel('phi, V')

            plt.subplot(3, 2, 3)
            plt.plot(x1, E)
            plt.xlabel('x, m')
            plt.ylabel('E, V/m')

            plt.subplot(3, 2, 4)
            plt.plot(x1, J_el, label='j_e')
            plt.plot(x1, J_i, label='j_i')
            plt.plot(x1, J_sm, label='j_b')
            plt.xlabel('x, m')
            plt.ylabel('j, A/m^2')
            plt.legend()

            plt.subplot(3, 2, 5)
            plt.plot(x1, J_poln)
            plt.xlabel('x, m')
            plt.ylabel('j_{tot}, A/m^2')

            plt.show()

        for i in range(nx - 1):
            E2 = E[i + 1]
            dg1[i] = (dg1[i] * k1 + E[i] * Mue * Ne[i]) / k
            E1 = E2
            sE1[i] = (sE1[i] * k1 + E[i]) / k
            sNe1[i] = (sNe1[i] * k1 + Ne[i]) / k

        sNe1[nx - 1] = (sNe1[nx - 1] * k1 + Ne[nx - 1]) / k
        k1 = k

        NeP = Ne.copy()
        NiP = Ni.copy()
        t_cur += th

    if i_per_safe == K_Per_Safe:
        i_per_safe = 0
        np.savetxt('NiP.txt', NiP)
        np.savetxt('E.txt', E)
        np.savetxt('TaP.txt', TaP)
        np.savetxt('TeP.txt', TeP)
        np.savetxt('NeP.txt', NeP)
        np.savetxt('NmP.txt', NmP)
        np.savetxt('nx.txt', nx)
        np.savetxt('nt.txt', nt)
        np.savetxt('Va.txt', Va)
        np.savetxt('D.txt', D)
        np.savetxt('P.txt', P)
        np.savetxt('Fg.txt', Fg)
        np.savetxt('NaP.txt', NaP)

NaP = Na.copy()
dg = dg1.copy()
sE = sE1.copy()
sNe = sNe1.copy()
sNe[nx - 1] = sNe1[nx - 1]

sgi = np.zeros(nx)
sTe = TeP.copy()
