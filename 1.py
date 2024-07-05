# %%
import numpy as np
import matplotlib.pyplot as plt

from NeNi_Gummel import Concentration_Meta, NeNi_Gummel
from fuBetta import fuBetta
from fuBettaP import fuBettaP
from kex1 import kex1
from ki1 import ki1
from ksi1 import ksi1
from urpuas1 import urpuas1

# Входные параметры
D = 0.025  # межэлектродное расстояние, м
Va = 25  # амплитуда напряжения
P = 133  # давление
Fg = 13.56e6  # частота генератора, Гц
Ta0 = 300  # температура атомов на границе, К
scet = 0
# параметры дискретизации
nx = 400  # число отрезков разбиения сетки по оси x
nt = 2000  # число отрезков разбиения сетки по оси t в одном периоде
nt_sloy = 10000  # количество слоев для расчета

# параметры процесса по-слоям
Max_Per = 10  # максимальное число периодов
step_sloy_print = 100  # задает слои, на которых рисуются графики
K_Per_Safe = 100  # число периодов для записи результатов

Ni0 = 9e15  # начальное значение для Ni, Ne
Nm0 = 1e15  # начальное приближение для Nm
Te0 = 34800  # начальное приближение для Te

# физические константы
e = 1.6e-19  # заряд электрона, Кл
e00 = 8.85e-12  # электрическая постоянная
kB = 1.381e-23  # постоянная Больцмана
me = 9.1e-31  # масса электрона, кг
mAr = 1.66e-27 * 40  # масса атома/иона Ar, кг
gamma = 0.01  # коэффициент вторичной электронной эмиссии

R21 = 6.2e-16
R4 = 3e-21
R5 = 1.1e-43

# рассчитываемые константы
omega = 2 * np.pi * Fg  # круговая частота электромагнитного поля
Num = 5.3e9 * (P / 133) * 273 / 300  # эффективная частота столкновений
Mue = e / Num / me  # подвижность электронов
Mui = e / Num / (mAr / 2)  # подвижность ионов
T_per = 1 / Fg

De_coef = kB * Mue / e
Di_coef = kB * Mui / e

# сетка по пространственной переменной
xk = D
x0 = 0

xh = (xk - x0) / nx  # шаг по x
x = np.linspace(x0, xk, nx + 1)  # массив точек по пространственной переменной
x1 = np.linspace(x0 + xh / 2, xk - xh / 2, nx)  # массив половинных точек по оси x

# сетка по временной переменной
t0 = 0  # начальный узел сетки по оси t
tk = 2 * np.pi / omega  # период по оси t
th = (tk - t0) / nt  # шаг по оси времени

# начальные приближения
NiP = 1e9 + Ni0 * (x / xk) * (1 - x / xk)
NeP = NiP
NeP0 = NeP
TeP = 1e3 * np.ones(nx + 1)
NmP = 1e5 * np.ones(nx + 1)
TaP = Ta0 * np.ones(nx + 1)
Gamma_e = np.zeros(nx)
Gamma_i = np.zeros(nx)

# E в половинных целых
E = (Va / D) * (x1 / xk)
E0 = E  # запоминаем E для вычисления тока смещения

# начальное приближение для слоя t=t0 "1" периода
t = t0

# усредненные параметры sgi, sTe, sNe, dg, sE
dg1 = np.zeros(nx)
sE1 = np.zeros(nx)
sNe1 = np.zeros(nx)
sTe = np.zeros(nx)
sTa = np.zeros(nx)

# температура постоянная
R31 = kex1(TeP)  # скорость процесса возбуждения по статье экономью
R11 = ksi1(TeP)  # скорость процесса ступенчатой ионизации по статье экономью

# температуры в средних узлах
Te1 = TeP[0]
Ta1 = TaP[0]

for i in range(nx):
    Ta2 = TaP[i + 1]
    Te2 = TeP[i + 1]
    
    sTe[i] = (Te1 + Te2) / 2
    sTa[i] = (Ta1 + Ta2) / 2
    
    Te1 = Te2
    Ta1 = Ta2

min_J = 0
max_J = 0

# расчет Na на этом временном слое
Na = np.zeros(nx + 1)
for i in range(nx + 1):
    T1 = TaP[i] * kB
    Na[i] = (P - kB * TeP[i] * NeP[i] - T1 * NiP[i] - T1 * NmP[i]) / T1
    
NaP = Na

# коэффициенты диффузии в средних узлах
De = De_coef * sTe
Di = Di_coef * sTa

# инициализация цикла
w = 1  # индекс периода
k = 1  # номер слоя в периоде
k1 = 0  # вспомогательная переменная
sloy_print = 0  # счетчик для вывода графиков
t_cur = 0  # текущее время
i_per_safe = 0  # счетчик периодов для сохранения файлов
sloy = 0

# начальное значение магнитного поля
B = np.zeros(nx + 1)

# цикл по периодам
for period in range(Max_Per):
    t_per = 0  # время в периоде
    i_per_safe += 1
    
    while t_per <= T_per:
        sloy += 1
        
        t_cur += th  # текущее время
        t_per += th  # время в периоде
        segm = t_per / T_per  # доля внутри периода
        
        sloy_print += 1  # счетчик слоев для вывода графика

        Nui = ki1(TeP)  # частота ионизации по статье экономью
        Betta = fuBetta(TeP, TaP, NeP, nx)  # коэффициент рекомбинации
        BettaP = fuBettaP(TeP, TaP, NeP, nx)  # производная от коэффициента рекомбинации по Ne

        # правая часть уравнений
        fm = Nui * NeP - Betta * NiP * NeP

        Ne, Ni, Gamma_e, Gamma_i = NeNi_Gummel(NeP, NiP, De, Di, Mue, Mui, E, gamma, fm, xh, th)
        Nm, gamma_m = Concentration_Meta(NeP, De, Mue, E, fm, xh, th, gamma)
        # уравнение Пуассона
        s = (e / e00) * (Ni - Ne)
        fik = Va * np.cos(omega * t)
        fi = urpuas1(xh, s, 0, fik)

        # расчет значения поля
        Ep = E

        for i in range(nx):
            E[i] = -(fi[i + 1] - fi[i]) / xh

        E1 = E[0]
        # вычисляем ток смещения
        J_sm = e00 * (E - E0) / th
        # запоминаем значение E
        E0 = E
        # вычисляем электронный и ионный токи
        J_el = -e * Gamma_e
        J_i = e * Gamma_i

        # полный ток
        J_poln = J_sm + J_el + J_i

        # расчет напряженности магнитного поля B без явного градиента
        for i in range(1, nx):
            B[i] = B[i] - th * (E[i] - E[i-1]) / xh

        # проверяем условие постоянства тока
        delta_J = np.max(J_poln) - np.min(J_poln)

        # печать графиков
        if sloy_print == step_sloy_print:
            sloy_print = 0  # обнуляем счетчик
            print (period, t_per, T_per)
            plt.semilogy(x, Ne, label='Ne')
            plt.semilogy(x, Ni, label='Ni')
            plt.xlabel('x, m')
            plt.ylabel('Ne, Ni, m^-3')
            plt.title(f'Период={period}, Сегмент={segm:.2f}')
            plt.legend(loc='lower center')
            plt.grid()
            plt.savefig(f'NiNe{scet}')
            plt.show()

            plt.plot(x, fi)
            plt.xlabel('x, m')
            plt.ylabel('phi, V')
            plt.ylim(-26, 26)
            plt.title(f'Период={period}, Сегмент={segm:.2f}')
            plt.grid()
            plt.savefig(f'phi{scet}')
            plt.show()
            
            plt.figure(figsize=(7.2,4.8))
            plt.plot(x1, E)
            plt.ylim(-10500, 10500)
            plt.title(f'Период={period}, Сегмент={segm:.2f}')
            plt.grid()
            plt.xlabel('x, m')
            plt.ylabel('E, V/m')
            plt.savefig(f'E{scet}')
            plt.show()

            plt.plot(x, Nm)
            plt.xlabel('x, m')
            plt.ylabel('Nm, m^-3')
            plt.title(f'Период={period}, Сегмент={segm:.2f}')
            plt.ylim()
            plt.grid()
            plt.savefig(f'Nm{scet}')
            plt.show()

            # График напряженности магнитного поля
            plt.figure(figsize=(7.2,4.8))
            plt.plot(x, B)
            plt.ylim(np.min(B), np.max(B))
            plt.title(f'Магнитное поле, Период={period}, Сегмент={segm:.2f}')
            plt.grid()
            plt.xlabel('x, m')
            plt.ylabel('B, T')
            plt.savefig(f'B{scet}')
            plt.show()

            scet += 1

        # перерасчет усредненных параметров
        for i in range(nx - 1):
            E2 = E[i + 1]
            dg1[i] = (dg1[i] * k1 + E1 * Mue * Ne[i]) / k
            E1 = E2
            sE1[i] = (sE1[i] * k1 + E[i]) / k
            sNe1[i] = (sNe1[i] * k1 + Ne[i]) / k

        sNe1[nx - 1] = (sNe1[nx - 1] * k1 + Ne[nx - 1]) / k
        k1 = k

        # переход на следующий временной слой
        NeP = Ne
        NiP = Ni
        t += th

    if i_per_safe == K_Per_Safe:
        i_per_safe = 0
        # запись
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
        #np.savetxt('sE.txt', sE)
        #np.savetxt('sNe.txt', sNe)
        #np.savetxt('sgi.txt', sgi)
        np.savetxt('sTe.txt', sTe)
        np.savetxt('w.txt', w)
        np.savetxt('Fg.txt', Fg)
        np.savetxt('NaP.txt', NaP)
B = B - th * np.gradient(E, x[:-1])
plt.figure(figsize=(7.2,4.8))
plt.plot(x[:-1], B)
plt.ylim(3.3, 0.2)
plt.title(f'Магнитное поле, Период=9, Сегмент=1')
plt.grid()
plt.xlabel('x, m')
plt.ylabel('B, T')
plt.savefig('B200.svg')
plt.show()
NaP = Na  # начальное условие для нестационарного уравнения для Na
# переход для усредненных параметров на следующий период
dg = dg1
sE = sE1
sNe = sNe1

sgi = np.zeros(nx)
sTe = TeP  # усредненная электронная температура



