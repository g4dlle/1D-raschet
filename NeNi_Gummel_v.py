import numpy as np

from Potok import Potok

def NeNi_Gummel_v(Ne, Ni, De, Di, V_e, V_i, E, gamma, F, delta_x, delta_t):
    """
    Программа численно решает систему уравнений диффузии
    dN_k/dt + dГ_k/dx = F (k = e, i)
    Г_е = -(De dNe/dx + mu_k*E*Ne)
    Г_i = -(Di dNi/dx - mu_k*E*Ni)
    со смешанными граничными условиями
    Программа делает один шаг по времени по методу Гуммеля.
    """

    n = len(Ne)
    nm1 = n - 1

    # Массивы для вычисления потока 
    Gamma_e = np.zeros(nm1)
    Gamma_i = np.zeros(nm1)

    # Запоминаем предыдущее приближение и выделяем память
    NeP = Ne.copy()
    NiP = Ni.copy()

    # знаки в слагаемых Vdr соответствуют знакам в уравнении
    Vdr_e = -V_e
    Vdr_i = -V_i

    # Вычисление потоков
    Gamma_e = Potok(delta_x, Ne, De, Vdr_e)
    Gamma_i = Potok(delta_x, Ni, Di, Vdr_i)

    # Вычисление концентраций в целых узлах
    Gamma_e1 = Gamma_e[0]
    Gamma_i1 = Gamma_i[0]

    for i in range(nm1 - 1):
        Gamma_e2 = Gamma_e[i + 1]
        Gamma_i2 = Gamma_i[i + 1]

        # Решение на новом слое
        delta_Ne = delta_t * (F[i + 1] - (Gamma_e2 - Gamma_e1) / delta_x)
        delta_Ni = delta_t * (F[i + 1] - (Gamma_i2 - Gamma_i1) / delta_x)

        NeP[i + 1] = Ne[i + 1] + delta_Ne
        NiP[i + 1] = Ni[i + 1] + delta_Ni

        Gamma_e1 = Gamma_e2
        Gamma_i1 = Gamma_i2

    # Уточняем потоки для использования в граничных условиях и впоследствии для расчета полного тока
    Gamma_e = Potok(delta_x, Ne, De, Vdr_e)
    Gamma_i = Potok(delta_x, Ni, Di, Vdr_i)

    # Учитываем граничные условия при x=0
    E2 = (E[0] + E[1]) / 2  # интерполяция в точке x=h
    E1 = (3 * E[0] - E[1]) / 2  # экстраполяция в точке x=0

    if E1 < 0:  # если E<0
        NiP[0] = E2 * NiP[1] / E1  # dГi/dx = 0
        NeP[0] = gamma * Gamma_i[0] / Vdr_e[0]  # Ге = - gamma*Гi
    elif E1 > 0:  # если E>0
        alpha = delta_x * Vdr_i[0] / Di[0]
        NiP[0] = np.exp(alpha) * NiP[1]  # Гi = 0
        NeP[0] = E2 * NeP[1] / E1  # dГe/dx = 0
    else:  # если E = 0, дрейфа нет, только диффузия
        NiP[0] = NiP[1]
        NeP[0] = NeP[1] - gamma * Gamma_i[0] * delta_x / De[0]  # Ге = - gamma*Гi

    Enm1 = (E[nm1 - 1] + E[nm1 - 2]) / 2  # интерполяция в точке x=x(n-1)
    En = (3 * E[nm1 - 1] - E[nm1 - 2]) / 2  # экстраполяция в точке x=x(n)

    # Учитываем граничные условия при x=L
    if En > 0:  # если E>0
        NeP[n - 1] = gamma * Gamma_i[nm1 - 1] / Vdr_e[nm1 - 1]  # Ге = - gamma*Гi
        NiP[n - 1] = Enm1 / En * NiP[n - 2]  # dГi/dx = 0
    elif En <= 0:  # если E<0
        NeP[n - 1] = Enm1 * NeP[n - 2] / En  # dГе/dx = 0
        alpha = delta_x * Vdr_i[nm1 - 1] / Di[nm1 - 1]
        NiP[n - 1] = np.exp(-alpha) * NiP[nm1 - 1]
    else:  # если En = 0, дрейфа нет, только диффузия
        NeP[n - 1] = NeP[n - 2]  # Гi = 0
        NeP[n - 1] = NeP[n - 2] + gamma * Gamma_i[n - 2] * delta_x / De[n - 2]  # Ге = - gamma*Гi

    return NeP, NiP, Gamma_e, Gamma_i