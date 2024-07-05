import numpy as np

def Potok(delta_x, N, D, Vdr):
    """
    Программа вычисляет поток на слое в алгоритме Shhaefter-Gummel 
    N - значение концентрации, заданная в целых узлах 
    D - коэффициент диффузии заданный в полуцелых узлах
    Vdr - дрейфовая скорость, задана в полуцелых узлах 
    delta_x - шаг пространственной сетки
    """
    n = len(N)
    Gamma = np.zeros(n-1)
    for i in range(n-1):
        N1 = N[i]
        N2 = N[i + 1]
        if Vdr[i] != 0:
            alpha = delta_x * Vdr[i] / D[i]
            exp_alpha = np.exp(alpha)
            hI0 = (exp_alpha - 1) / alpha * delta_x
            Gamma[i] = D[i] * (N1 - exp_alpha * N2) / hI0
        else:
            Gamma[i] = D[i] * (N1 - N2) / delta_x
    return Gamma

def NeNi_Gummel(Ne, Ni, De, Di, mu_e, mu_i, E, gamma, F, delta_x, delta_t):
    n = len(Ne)
    nm1 = n - 1

    # Массивы для вычисления потока 
    Gamma_e = np.zeros(nm1)
    Gamma_i = np.zeros(nm1)

    # запоминаем предыдущее приближение и выделяем память
    NeP = np.copy(Ne)
    NiP = np.copy(Ni)

    # знаки в слагаемых Vdr соотвествуют знакам в уравнении. 
    Vdr_e = mu_e * E     # знак + потому что знак учтен в уравнении (2), перед этим слагаемым минус.
    Vdr_i = -mu_i * E    # знак - потому что знак учтен в уравнении (3) перед этим слагаемым минус

    Gamma_e = Potok(delta_x, Ne, De, Vdr_e)
    Gamma_i = Potok(delta_x, Ni, Di, Vdr_i)

    # вычисление концентраций в целых узлах 
    Gamma_e1 = Gamma_e[0]
    Gamma_i1 = Gamma_i[0]

    for i in range(nm1 - 1):
        Gamma_e2 = Gamma_e[i + 1]
        Gamma_i2 = Gamma_i[i + 1]

        # решение на новом слое
        delta_Ne = delta_t * (F[i + 1] - (Gamma_e2 - Gamma_e1) / delta_x)
        delta_Ni = delta_t * (F[i + 1] - (Gamma_i2 - Gamma_i1) / delta_x)

        NeP[i + 1] = Ne[i + 1] + delta_Ne
        NiP[i + 1] = Ni[i + 1] + delta_Ni

        Gamma_e1 = Gamma_e2
        Gamma_i1 = Gamma_i2

    # Уточняем потоки для использования в граничных условиях и впоследствии для расчета полного тока
    Gamma_e = Potok(delta_x, Ne, De, Vdr_e)
    Gamma_i = Potok(delta_x, Ni, Di, Vdr_i)

    # учитываем граничные условия при x=0:
    # Вычисляем поле в целых узлах
    E2 = (E[0] + E[1]) / 2  # интерполяция в точке х=h
    E1 = (3 * E[0] - E[1]) / 2  # экстраполяция в точке x=0

    if E1 < 0:  # если E<0
        NiP[0] = E2 * NiP[1] / E1  # dГi/dx = 0,
        NeP[0] = gamma * Gamma_i[0] / Vdr_e[0]  # Ге = - gamma*Гi,
    elif E1 > 0:  # если E>0
        alpha = delta_x * Vdr_i[0] / Di[0]
        NiP[0] = np.exp(alpha) * NiP[1]  # Гi = 0,
        NeP[0] = E2 * NeP[1] / E1  # dГe/dx = 0,
    else:  # если E =0, дрейфа нет, только диффузия
        NiP[0] = NiP[1]
        NeP[0] = NeP[1] - gamma * Gamma_i[0] * delta_x / De[0]  # Ге = - gamma*Гi

    Enm1 = (E[-2] + E[-3]) / 2  # интерполяция в точке х=x(n-1)
    En = (3 * E[-2] - E[-3]) / 2  # экстраполяция в точке х=x(n)

    # учитываем граничные условия при x=L:
    if En > 0:  # если E>0
        NeP[-1] = gamma * Gamma_i[nm1 - 1] / Vdr_e[nm1 - 1]  # Ге = - gamma*Гi
        NiP[-1] = Enm1 / En * NiP[-2]  # dГi/dx = 0,
    elif En <= 0:  # если E<0,
        NeP[-1] = Enm1 * NeP[-2] / En  # dГе/dx = 0,
        alpha = delta_x * Vdr_i[nm1 - 1] / Di[nm1 - 1]
        NiP[-1] = np.exp(-alpha) * NiP[nm1 - 1]
    else:  # если En = 0, дрейфа нет, только диффузия
        NeP[-1] = NeP[-2]  # Гi = 0,
        NeP[-1] = NeP[-2] + gamma * Gamma_i[-2] * delta_x / De[-2]  # Ге = - gamma*Гi

    return NeP, NiP, Gamma_e, Gamma_i

def Concentration_Meta(Ne, De, mu_e, E, F, delta_x, delta_t, gamma):
    n = len(Ne)
    nm1 = n - 1

    # Массивы для вычисления потока 
    Gamma_e = np.zeros(nm1)

    # Запоминаем предыдущее приближение и выделяем память
    NeP = np.copy(Ne)

    # Знак в слагаемом Vdr соотвествует знаку в уравнении. 
    Vdr_e = mu_e * E  # знак + потому что знак учтен в уравнении

    Gamma_e = Potok(delta_x, Ne, De, Vdr_e)

    # Вычисление концентраций в целых узлах 
    Gamma_e1 = Gamma_e[0]

    for i in range(nm1 - 1):
        Gamma_e2 = Gamma_e[i + 1]

        # Решение на новом слое
        delta_Ne = delta_t * (F[i + 1] - (Gamma_e2 - Gamma_e1) / delta_x)
        NeP[i + 1] = Ne[i + 1] + delta_Ne

        Gamma_e1 = Gamma_e2

    # Уточняем потоки для использования в граничных условиях и впоследствии для расчета полного тока
    Gamma_e = Potok(delta_x, Ne, De, Vdr_e)

    # Учитываем граничные условия при x=0:
    E2 = (E[0] + E[1]) / 2  # интерполяция в точке х=h
    E1 = (3 * E[0] - E[1]) / 2  # экстраполяция в точке x=0

    if E1 < 0:  # если E<0
        NeP[0] = gamma * Gamma_e[0] / Vdr_e[0]  # Ге = - gamma*Гi,
    elif E1 > 0:  # если E>0
        NeP[0] = E2 * NeP[1] / E1  # dГe/dx = 0,
    else:  # если E =0, дрейфа нет, только диффузия
        NeP[0] = NeP[1] - gamma * Gamma_e[0] * delta_x / De[0]  # Ге = - gamma*Гi

    Enm1 = (E[-2] + E[-3]) / 2  # интерполяция в точке х=x(n-1)
    En = (3 * E[-2] - E[-3]) / 2  # экстраполяция в точке х=x(n)

    # Учитываем граничные условия при x=L:
    if En > 0:  # если E>0
        NeP[-1] = gamma * Gamma_e[nm1 - 1] / Vdr_e[nm1 - 1]  # Ге = - gamma*Гi
    elif En <= 0:  # если E<0,
        NeP[-1] = Enm1 * NeP[-2] / En  # dГе/dx = 0,
    else:  # если En = 0, дрейфа нет, только диффузия
        NeP[-1] = NeP[-2] + gamma * Gamma_e[-2] * delta_x / De[-2]  # Ге = - gamma*Гi

    return NeP, Gamma_e
