import numpy as np

def fuBetta(TeP, TaP, NeP, nx):
    """
    Рассчитывает коэффициенты рекомбинации Betта для каждой точки сетки.

    Parameters:
    TeP : array
        Электронная температура в каждой точке сетки.
    TaP : array
        Температура атомов в каждой точке сетки.
    NeP : array
        Концентрация электронов в каждой точке сетки.
    nx : int
        Количество отрезков разбиения сетки.

    Returns:
    Betta : array
        Коэффициенты рекомбинации для каждой точки сетки.
    """
    # Глобальные переменные (необходимо определить в вашей рабочей среде)
    global kB, e, gamma, Mue, Mui, e00, R11, R21, R31, R4, R5, DeltaM, Ei, Ei2, Num, Ci, P, me

    # Коэффициенты
    BettaF_coef = 2.7e-19 / (11605) ** (-0.75)
    BettaTR_coef = 8.75e-39 / (11605) ** (-4.5)

    # Инициализация массива Betta
    Betta = np.zeros(nx + 1)

    # Первая точка
    Ta1 = (TaP[0] + TaP[1]) / 2
    Te1 = (TeP[0] + TeP[1]) / 2
    Ne1 = (NeP[0] + NeP[1]) / 2
    BettaF = BettaF_coef * (Te1) ** (-0.75)
    BettaTR = BettaTR_coef * Ne1 * (Te1) ** (-4.5)
    Betta[0] = BettaF + BettaTR

    # Средние точки
    for i in range(1, nx):
        Te1 = TeP[i]
        Ta1 = TaP[i]
        Ne1 = NeP[i]
        BettaF = BettaF_coef * (Te1) ** (-0.75)
        BettaTR = BettaTR_coef * Ne1 * (Te1) ** (-4.5)
        Betta[i] = BettaF + BettaTR

    # Последняя точка
    Ta1 = (TaP[nx - 1] + TaP[nx]) / 2
    Te1 = (TeP[nx - 1] + TeP[nx]) / 2
    Ne1 = (NeP[nx - 1] + NeP[nx]) / 2
    BettaF = BettaF_coef * (Te1) ** (-0.75)
    BettaTR = BettaTR_coef * Ne1 * (Te1) ** (-4.5)
    Betta[nx] = BettaF + BettaTR

    return Betta