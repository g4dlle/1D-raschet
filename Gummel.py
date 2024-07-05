import numpy as np

def gummel_10(N, D, Vdr, F, delta_x, delta_t, K_step):
    """
    Программа решает уравнение диффузии
    dN/dt - d/dx(D dN/dx + Vdr*N) = F
    с однородными граничными условиями первого рода
    N(x[0]) = N(x[-1]) = 0
    
    Параметры:
    N - начальное приближение к искомой функции, заданной на сетке setka_x
    D - коэффициент дифузии - сеточная функция, заданная на сетке setka_x1
    Vdr - дрефовая скорость, задана в полуцелых узлах на сетке setka_x1
    F - правая часть уравнения - сеточная функция, заданная на сетке setka_x
    delta_x - шаг по пространственной переменной
    delta_t - шаг по времени
    K_step - количество шагов по времени
    """

    n = len(N)
    n1 = len(Vdr)

    if n1 != n-1 or n != len(F) or n1 != len(D):
        raise ValueError('Ошибка – размерности массивов не совпадают')

    # Массив для вычисления потока
    Gamma = np.zeros(n1)

    # Цикл по слоям
    for K in range(K_step):
        # вычисление потоков в полуцелых узлах
        N1 = N[0]
        
        for i in range(n1):
            N2 = N[i + 1]
            
            alpha = delta_x * Vdr[i] / D[i]
            exp_alpha = np.exp(alpha)
            
            hI0 = (exp_alpha - 1) / exp_alpha / delta_x
            Gamma[i] = D[i] * (N1 - exp_alpha * N2) / hI0
            
            N1 = N2
        
        # вычисление концентраций в целых узлах
        Gamma1 = Gamma[0]

        for i in range(1, n - 1):
            Gamma2 = Gamma[i]
            # решение на новом слое
            delta_N = delta_t * ((Gamma2 - Gamma1) / delta_x + F[i])
            N[i] = N[i] - delta_N
            Gamma1 = Gamma2
    
    return N
