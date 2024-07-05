import numpy as np
import matplotlib.pyplot as plt

from fuBetta import fuBetta
from fuBettaP import fuBettaP
from kex1 import kex1
from ki1 import ki1
from ksi1 import ksi1
from urpuas1 import urpuas1

# Глобальные переменные
global kB, e, gamma, Mue, Mui, e00, R11, R21, R31, R4, R5, DeltaM, Ei, Ei2, Num, Ci, P, me, Ei1, mAr, omega

# Константы
kB = 1.381e-23  # постоянная Больцмана
e = 1.6e-19  # заряд электрона, Кл
me = 9.1e-31  # масса электрона, кг
mAr = 1.66e-27 * 40  # масса атома/иона Ar, кг
mO = 5.44e-26  # масса атома/иона кислорода, кг
Ei = 2.4e-18  # потенциал ионизации, Дж
Ei2 = 11.56 * 1.60219e-19  # 1эВ = 1.60219e-19 Дж
Ei1 = 4.35 * 1.60219e-19
e00 = 8.85e-12  # электрическая постоянная
gamma = 0.01  # коэффициент вторичной электронной эмиссии
Ci = 2e-21  # константа, M^2/Эв
R21 = 6.2e-16
R4 = 3e-21
R5 = 1.1e-43
Cv = 1e-6

# Загрузка данных
D = np.load('D.npy')  # межэлектродное расстояние, м
Va = np.load('Va.npy')  # амплитуда напряжения
P = np.load('P.npy')  # давление газа, Па
nx = np.load('nx.npy')  # число отрезков разбиения сетки по оси х
NeP = np.load('NeP.npy')  # начальное приближение для Ne
E = np.load('E.npy')  # поле
NiP = np.load('NiP.npy')  # начальное приближение для Ni
TaP = np.load('TaP.npy')  # начальное приближение для Ta
TeP = np.load('TeP.npy')  # начальное приближение для Te
NmP = np.load('NmP.npy')  # начальное приближение для Nm
dg = np.load('dg.npy')  # усредненный за период джоулев нагрев
NaP = np.load('NaP.npy')  # начальное приближение для Na
sNe = np.load('sNe.npy')  # усредненная за период концентрация электронов
sTe = np.load('sTe.npy')  # усредненная за период электронная температура
sgi = np.load('sgi.npy')  # усредненный за период ионный ток
w = np.load('w.npy')  # количество уже прощитанных периодов
Fg = np.load('Fg.npy')  # частота генератора, Гц
nt = np.load('nt.npy')  # число отрезков разбиения сетки по оси t в одном периоде
sE = np.load('sE.npy')  # усредненное за период значение поля

# РАСЧИТЫВАЕМЫЕ КОНСТАНТЫ
omega = 2 * np.pi * Fg  # круговая частота электромагнитного поля 
DeltaM = me / 2 / mAr
Pode = 5.852e3 / P  # подвижность электронов [Па^-1]*[м]^2*[В]^-1*[с]^-1
Podi = 13.3 / P  # подвижность ионов

x0 = 0  # начальный узел сетки по оси х 
xk = D  # последний узел сетки  по оси х
xh = (xk - x0) / nx  # шаг сетки по оси х

t0 = 0  # начальный узел сетки по оси t 
tk = 2 * np.pi / omega  # период по оси t 
th = (tk - t0) / nt  # шаг по оси времени

Num = 5.3e9 * (P / 133) * 273 / 300  # Эффиктивная частота столкновений
Mue = e / Num / me  # подвижность электронов [Па^-1]*[м]^2*[В]^-1*[с]^-1
Mui = e / Num / (mAr / 2)  # подвижность ионов

# ОСНОВНОЙ БЛОК
x = np.linspace(x0, xk, nx + 1)  # массив целых точек по оси х
x1 = np.linspace(x0 + xh / 2, xk - xh / 2, nx)  # массив половинных точек по оси х 

# начальное приближение для слоя t=t0
t = t0  # время равное начальному слою

t = t + th  # последний слой предыдущего - первый этого, поэтому считаем со второго 
w1 = 1  # вспомогательная переменная отсчета партии периодов
Vp = Va
VpP = Va
w11 = 0

while w11 < 1:
    w11 += 1
    w1 = 1

    while w1 <= 2:
        # вспомогательные переменные для расчета усредненных параметров по периметру
        dg1 = np.zeros(nx - 1)  # джоулев нагрев

        sgi1 = np.zeros(nx)  # поток ионов
        sge = np.zeros(nx)  # поток электронов
        sE1 = np.zeros(nx)  # значение поля
        sTe1 = np.zeros(nx)  # электронная температура
        sNe1 = np.zeros(nx)  # концентрация электронов

        sNe1 = np.zeros(nx + 1)
        sTe1 = np.zeros(nx + 1)

        w += 1
        w1 += 1

        k1 = 0
        k = 1  # переменная отсчета временных слоев

        while k < nt:
            # вычисление частот и коэффициентов
            Nui = ki1(TeP)  # частота ионизации по статье экономью
            Betta = fuBetta(TeP, TaP, NeP, nx)  # коэффициент рекомбинации
            BettaP = fuBettaP(TeP, TaP, NeP, nx)  # производная от коэффициента рекомбинации по Ne
            R31 = kex1(TeP)  # скорость процесса возбуждения по статье экономью
            R11 = ksi1(TeP)  # скорость процесса ступенчатой ионизации по статье экономью

            # итерации на одном временном слое для 4-х уравнений к значением коэффициентов и поля
            for gg in range(1):
                Na = uravNa1(th, xh, NmP, NeP, NaP, NeP, R31, R11, TaP, Betta, Nui, NiP)  # нестационарное уравнение для Na
                Ne = uravNe1(th, xh, Nui, Betta, TeP, TaP, E, NiP, NeP, BettaP, Na, NmP, R11)  # нестационарное уравнение для Ne
                Ni = uravNi1(th, xh, Nui, Betta, TeP, TaP, E, NiP, Ne, Na, NmP, R11)  # нестационарное уравнение для Ni
                Nm = uravNm1(th, xh, NmP, Ne, Na, Ne, R31, R11)  # нестационарное уравнение для Nm

            # уравнение Пуассона
            s = e / e00 * (Ni - Ne)

            #fik = Va * np.sin(omega * t)
            fik = Vp

            fi = urpuas1(xh, s, 0, fik)

            # расчет новых значения поля
            Ep = E[0]  # вспомогательный массив для хранения значений поля в первой точке
            Ep1 = E.copy()

            for i in range(nx):
                E[i] = -(fi[i + 1] - fi[i]) / (x[i + 1] - x[i])

            # расчет потока электронов
            ge = np.zeros(nx)
            ge[0] = -kB * Mue * (TeP[0] + TeP[1]) / 2 * (Ne[1] - Ne[0]) / xh / e - Mue * (Ne[0] + Ne[1]) / 2 * Ep

            for i in range(1, nx):
                NmPi = NmP[i]
                NmP2 = NmPi ** 2
                ge[i] = -(Ne[i] - NeP[i]) / th * xh + Nui[i] * Ne[i] * xh - Betta[i] * Ne[i] * Ni[i] * xh + ge[i - 1] + NmPi * Ne[i] * R11[i] * xh + R21 * NmP2 * xh

            # усредненная часть джоулева нагрева
            for i in range(nx - 1):
                E1 = E[i]
                E1q = E1 ** 2
                dg1[i] = (ge[i] + dg1[i] * (k - 1)) / k

            # итерации для нелинейности в коэффициенте теплопроводности
            TeN = TeP.copy()

            for i in range(1):
                th1 = th
                Nui = ki1(TeN)
                Betta = fuBetta(TeN, TaP, Ne, nx)
                R31 = kex1(TeN)
                R11 = ksi1(TeN)
                TeN = TeG22(th, xh, Nui, Betta, TeP, TaP, E, Ne, TeN, NmP, dg, ge, NeP, R11, R31, Na, Ni, TeN, sTe, sNe, sE)

            TeP = TeN.copy()

            # расчет ионного тока
            gi = np.zeros(nx)
            gi[0] = -kB * Mui * (TaP[0] + TaP[1]) / 2 * (Ni[1] - Ni[0]) / xh / e + Mui * (Ni[0] + Ni[1]) / 2 * Ep

            for i in range(1, nx):
                NmPi = NmP[i]
                NmP2 = NmPi ** 2
                gi[i] = -(Ni[i] - NiP[i]) / th * xh + Nui[i] * Ne[i] * xh - Betta[i] * Ne[i] * Ni[i] * xh + gi[i - 1] + R11[i] * NmPi * Ne[i] * xh + R21 * NmP2 * xh

            jt = (gi[0] - ge[0]) / 2

            for i in range(1, nx - 1):
                jt += (gi[i] - ge[i])

            jt = (jt + (gi[nx - 1] - ge[nx - 1]) / 2) * xh
            Vp = (Cv * omega * D * (-Va) * np.cos(omega * t) - e * jt) * th / (e00 + Cv * D) + VpP
            VpP = Vp

            TaP = TiG22(th, xh, Nui, Betta, TeP, TaP, E, Ne, TeP, NmP, dg, sgi, NeP, R11, R31, Na, Ni, TeN, sTe, sNe, NiP, gi, sE)  # нестационарное уравнение для Та

            # усредненные параметры
            for i in range(nx):
                sgi1[i] = (sgi1[i] * k1 + gi[i]) / k
                sE1[i] = (sE1[i] * k1 + E[i]) / k

            for i in range(nx + 1):
                sNe1[i] = (sNe1[i] * k1 + Ne[i]) / k
                sTe1[i] = (sTe1[i] * k1 + TeP[i]) / k
                PPP[i] = NaP[i] * kB * TaP[i] + kB * TeP[i] * NeP[i] + TaP[i] * NiP[i] * kB + TaP[i] * NmP[i] * kB

            # переход на следующий временной слой
            k1 = k

            if k == 1:
                Ne1 = Ne.copy()
                Ev1 = E.copy()
                Tv1 = TeP.copy()

                N1m = np.zeros(20)
                T12m = np.zeros(20)
                for i in range(20):
                    N1m[i] = Ne[1 + 10 * (i - 1)]
                    T12m[i] = TeP[1 + 10 * (i - 1)]

            if k == 50:
                Tv1 = TeP.copy()

            if k == 100:
                Ne2 = Ne.copy()
                Ev2 = E.copy()
                Tv2 = TeP.copy()

                N2m = np.zeros(20)
                Tv2m = np.zeros(20)
                for i in range(20):
                    N2m[i] = Ne[1 + 10 * (i - 1)]
                    Tv2m[i] = TeP[1 + 10 * (i - 1)]

            if k == 200:
                Ne3 = Ne.copy()
                Nic = Ni.copy()

                Nicm = np.zeros(20)
                Nem = np.zeros(20)
                xm = np.zeros(20)
                Tv3m = np.zeros(20)
                for i in range(20):
                    Nicm[i] = Ni[1 + 10 * (i - 1)]
                    Nem[i] = Ne[1 + 10 * (i - 1)]
                    xm[i] = x[1 + 10 * (i - 1)]
                    Tv3m[i] = TeP[1 + 10 * (i - 1)]

                Ev3 = E.copy()
                Tv3 = TeP.copy()
                Tac = TaP.copy()

                np.save('Ne3.npy', Ne3)
                np.save('Nic.npy', Nic)

            if k == 300:
                Ne4 = Ne.copy()
                Ev4 = E.copy()
                Tv4 = TeP.copy()

                N4m = np.zeros(20)
                Tv4m = np.zeros(20)
                for i in range(20):
                    N4m[i] = Ne[1 + 10 * (i - 1)]
                    Tv4m[i] = TeP[1 + 10 * (i - 1)]

            if k == 399:
                Ne5 = Ne.copy()
                Ev5 = E.copy()
                Tv5 = TeP.copy()

                N5m = np.zeros(20)
                Tv5m = np.zeros(20)
                for i in range(20):
                    N5m[i] = Ne[1 + 10 * (i - 1)]
                    Tv5m[i] = TeP[1 + 10 * (i - 1)]

            k += 1
            t += th

            NeP0 = NeP.copy()
            NiP = Ni.copy()
            NeP = Ne.copy()
            NmP = Nm.copy()

            # прорисовка
            plt.figure(1)
            plt.subplot(4, 2, 1)
            plt.plot(x1, E)
            plt.xlim([0, 0.026])

            plt.subplot(4, 2, 2)
            plt.plot(x, TaP)
            plt.xlim([0, 0.026])

            plt.subplot(4, 1, 2)
            plt.plot(x, TeP)
            plt.xlim([0, 0.026])

            plt.subplot(4, 1, 3)
            plt.plot(x, Ne, x, Ni)
            plt.xlim([0, 0.026])

            plt.subplot(4, 1, 4)
            plt.plot(x, NmP)
            plt.xlim([0, 0.026])

            for i in range(nx):
                jp = e * (gi - ge) + e00 * (E - Ep1) / th

            plt.figure(2)
            plt.plot(x1, jp)

            plt.show()

            jpv = np.zeros(nt - 1)
            jtv = np.zeros(nt - 1)
            vav = np.zeros(nt - 1)

            jpv[k - 1] = jp[0]
            jtv[k - 1] = t - th
            vav[k - 1] = fik

            # переход на следующий период по усредненным параметрам
            for i in range(nx - 1):
                dg[i] = dg1[i]

            for i in range(nx):
                sgi[i] = sgi1[i]
                sE[i] = sE1[i]

            for i in range(nx + 1):
                sNe[i] = sNe1[i]
                sTe[i] = sTe1[i]

        # визуализация
        t_min = min(jtv)
        t_max = max(jtv)

        plt.figure(3)
        plt.plot(jtv, jpv, 'k')
        plt.xlim([t_min, t_max])
        plt.xlabel('t, s')
        plt.ylabel('J, A/m^2')
        plt.show()

        plt.figure(4)
        plt.plot(jtv, vav, 'k')
        plt.xlim([t_min, t_max])
        plt.xlabel('t, s')
        plt.ylabel('φ, V')
        plt.show()

        plt.figure(5)
        plt.plot(x, Ne3, '-k', x, Nic, '--k')
        plt.xlim([0.0, 0.026])
        plt.xlabel('d, m')
        plt.ylabel('N_e, N_i, 1/m^3')
        plt.legend(['1', '2'])
        plt.show()

        plt.figure(6)
        plt.semilogy(x, Ne3, '-k', x, Nic, '--k')
        plt.xlim([0.0, 0.026])
        plt.xlabel('d, m')
        plt.ylabel('N_e, N_i, 1/m^3')
        plt.legend(['1', '2'])
        plt.show()

        plt.figure(7)
        plt.plot(x, Ne1, '-k', x, Ne2, '--k', x, Ne4, '-.k', x, Ne5, ':k')
        plt.xlim([0.003, 0.01])
        plt.xlabel('d, m')
        plt.ylabel('N_e, 1/m^3')
        plt.legend(['1', '2', '3', '4'])
        plt.show()

        plt.figure(8)
        plt.semilogy(x, Ne1, '-k', x, Ne2, '--k', x, Ne4, '-.k', x, Ne5, ':k')
        plt.xlim([0.003, 0.01])
        plt.xlabel('d, m')
        plt.ylabel('N_e, 1/m^3')
        plt.legend(['1', '2', '3', '4'])
        plt.show()

        plt.figure(9)
        plt.plot(x, Tv3 / 11605, '-k', x, Tv4 / 11605, '--k', x, Tv5 / 11605, '-.k')
        plt.xlim([0, 0.026])
        plt.xlabel('d, m')
        plt.ylabel('T_e, eV')
        plt.legend(['1', '2', '3'])
        plt.show()

        plt.figure(10)
        plt.plot(x, sTe / 11605, '-k')
        plt.xlim([0, 0.026])
        plt.xlabel('d, m')
        plt.ylabel('T_{e, av}, eV')
        plt.show()

        plt.figure(11)
        plt.plot(x, Tac - 300, 'k')
        plt.xlim([0, 0.026])
        plt.xlabel('d, m')
        plt.ylabel('(T_a - 300), K')
        plt.show()
