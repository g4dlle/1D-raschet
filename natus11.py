import numpy as np
import matplotlib.pyplot as plt

from NeNi_Gummel_v import NeNi_Gummel_v
from fuBetta import fuBetta
from fuBettaP import fuBettaP
from kex1 import kex1
from ki1 import ki1
from ksi1 import ksi1
from urpuas1 import urpuas1

# Входные параметры
D = 0.002  # межэлектродное расстояние, м
Va = 100  # амплитуда напряжения
P = 1e3  # давление
Fg = 13.56e6  # частота генератора, Гц
Ta0 = 3000  # температура атомов на границе, К
StIon = 1e-5  # степень ионизации

# Управляющие параметры
Max_Per = 1  # максимальное число периодов
nx = 200  # число отрезков разбиения сетки по оси x
nt = 10000  # число отрезков разбиения сетки по оси t в одном периоде
nt_sloy = 1  # количество слоев для расчета
step_sloy_print = 1  # задает слои, на которых рисуются графики

# Физические константы
e = 1.6e-19  # заряд электрона, Кл
e00 = 8.85e-12  # электрическая постоянная
kB = 1.381e-23  # постоянная Больцмана

me = 9.1e-31  # масса электрона, кг
mAr = 1.66e-27 * 40  # масса атома/иона Ar, кг
gamma = 0.01  # коэффициент вторичной электронной эмиссии

R21 = 6.2e-16
R4 = 3e-21
R5 = 1.1e-43

# Рассчитываемые константы
omega = 2 * np.pi * Fg  # круговая частота электромагнитного поля
T_per = 1 / Fg  # длина периода
Num = 5.3e9 * (P / 133) * 273 / 300  # эффективная частота столкновений

Mue = e / Num / me  # подвижность электронов
Mui = e / Num / (mAr / 2)  # подвижность ионов

De_coef = kB * Mue / e
Di_coef = kB * Mui / e

Na0 = P / (kB * Ta0)  # плотность нейтральных атомов
Ni0 = StIon * Na0  # начальное значение для Ni, Ne
Nm0 = 1e13  # начальное приближение для Nm

# Сетка по пространственной переменной
xk = D
x0 = 0

xh = (xk - x0) / nx  # шаг по x
x = np.linspace(x0, xk, nx + 1)  # массив точек по пространственной переменной
x1 = np.linspace(x0 + xh / 2, xk - xh / 2, nx)  # массив половинных точек по оси x

lpn = nx + 1  # last point number
mpn = round((nx + 2) / 2)  # middle point number

# Сетка по временной переменной
t0 = 0  # начальный узел сетки по оси t
tk = 2 * np.pi / omega  # период по оси t
th = (tk - t0) / nt  # шаг по оси времени

# Начальные приближения
x0 = D / 2  # середина промежутка

NiP = Ni0 * np.ones(nx + 1)
NeP = NiP.copy()
NeP0 = NeP.copy()
TeP = 1e3 * np.ones(nx + 1)
NmP = 1e5 * np.ones(nx + 1)
TaP = Ta0 * np.ones(nx + 1)
Gamma_e = np.zeros(nx)
Gamma_i = np.zeros(nx)

# Е в половинных целых
E = (Va / D) * (x1 / D)

# начальное приближение для слоя t=t0 "1" периода
t = t0

# Усредненные параметры sgi, sTe, sNe, dg, sE
dg1 = np.zeros(nx)
sE1 = np.zeros(nx)
sNe1 = np.zeros(nx)
sTe = np.zeros(nx)
sTa = np.zeros(nx)

# Температура постоянная
R31 = kex1(TeP)  # скорость процесса возбуждения по статье экономью
R11 = ksi1(TeP)  # скорость процесса ступенчатой ионизации по статье экономью

# Температуры в средних узлах
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

NaP = Na

# Коэффициенты диффузии в средних узлах
De = De_coef * sTe
Di = Di_coef * sTa

# Инициализация цикла
w = 1  # индекс периода
k = 1  # номер слоя в периоде
k1 = 0  # вспомогательная переменная
sloy_print = 0  # счетчик для вывода графиков
t_cur = 0  # текущее время

# Цикл по периодам
for period in range(Max_Per):
    t_per = 0  # время в периоде
    k_sloy = 0
    
    while t_per < T_per:
        k_sloy += 1
        
        if k_sloy > nt_sloy:
            break
        
        V_el = -Mue * E
        V_i = Mui * E
        
        t_cur += th  # текущее время
        t_per += th  # время в периоде
        segm = t_per / T_per  # доля внутри периода
        
        # Вычисляем правую часть уравнений
        Nui = ki1(TeP)  # частота ионизации по статье экономью
        Betta = fuBetta(TeP, TaP, NeP, nx)  # коэффициент рекомбинации
        BettaP = fuBettaP(TeP, TaP, NeP, nx)  # производная от коэффициента рекомбинации по Ne

        fm = Nui * NeP * Na  # правая часть уравнений
        
        Ne, Ni, Gamma_e, Gamma_i = NeNi_Gummel_v(NeP, NiP, De, Di, V_el, V_i, E, gamma, fm, xh, th)
        
        # Уравнение Пуассона
        s = (Ni - Ne)
        s = (e / e00) * s
        fik = Va * np.cos(omega * t)
        fi = urpuas1(xh, s, 0, fik)
        
        # Конец расчета напряжения
        # Расчет значения поля
        Ep = E.copy()

        for i in range(nx):
            E[i] = -(fi[i + 1] - fi[i]) / xh

        E1 = E[0]

        # Печать графиков
        sloy_print += 1
        
        if sloy_print == step_sloy_print:
            sloy_print = 0  # обнуляем счетчик
            # Рисуем графики
            plt.figure(1)
            
            plt.subplot(4, 2, 1)
            plt.plot(x, Ne)
            plt.xlabel('x, m')
            plt.ylabel('Ne, m^-3')
            plt.title(f'Period={period}, Segment={segm:.2f}')
            plt.grid()

            plt.subplot(4, 2, 2)
            plt.plot(x, Ni)
            plt.xlabel('x, m')
            plt.ylabel('Ni, m^-3')
            plt.grid()

            plt.subplot(4, 2, 3)
            plt.plot(x, fi)
            plt.xlabel('x, m')
            plt.ylabel('phi, V')
            plt.grid()

            plt.subplot(4, 2, 4)
            plt.plot(x1, E)
            plt.xlabel('x, m')
            plt.ylabel('E, V/m')
            plt.grid()

            plt.subplot(4, 2, 5)
            plt.plot(x1, Gamma_e)
            plt.xlabel('x, m')
            plt.ylabel('Gamma_e, 1/m^2')
            plt.grid()

            plt.subplot(4, 2, 6)
            plt.plot(x, fm)
            plt.xlabel('x, m')
            plt.ylabel('Right Hand, 1/(s*m^3)')
            plt.grid()

            plt.show()

        # Перерасчет усредненных параметров
        for i in range(nx - 1):
            E2 = E[i + 1]
            dg1[i] = (dg1[i] * k1 + E1 * Mue * Ne[i]) / k

            E1 = E2
            sE1[i] = (sE1[i] * k1 + E[i]) / k
            sNe1[i] = (sNe1[i] * k1 + Ne[i]) / k

        sNe1[nx - 1] = (sNe1[nx - 1] * k1 + Ne[nx - 1]) / k
        k1 = k

        # Переход на следующий временной слой
        NeP = Ne.copy()
        NiP = Ni.copy()
        t += th

NaP = Na  # начальное условие для нестационарного уравнения для Na
# Переход для усредненных параметров на следующий период
dg = dg1.copy()
sE = sE1.copy()
sNe = sNe1.copy()

sgi = np.zeros(nx)
sTe = TeP.copy()  # усредненная электронная температура

# Запись данных
np.save('NiP.npy', NiP)
np.save('E.npy', E)
np.save('TaP.npy', TaP)
np.save('TeP.npy', TeP)
np.save('NeP.npy', NeP)
np.save('NmP.npy', NmP)
np.save('nx.npy', nx)
np.save('nt.npy', nt)
np.save('Va.npy', Va)
np.save('D.npy', D)
np.save('dg.npy', dg)
np.save('P.npy', P)
np.save('sE.npy', sE)
np.save('sNe.npy', sNe)
np.save('sgi.npy', sgi)
np.save('sTe.npy', sTe)
np.save('w.npy', w)
np.save('Fg.npy', Fg)
np.save('NaP.npy', NaP)
