import numpy as np

def fuBettaP(TeP, TaP, NeP, nx):
    # Константы
    BettaF_coef = 2.7e-19 / (11605) ** -0.75
    BettaTR_coef = 8.75e-39 / (11605) ** -4.5
    
    # Инициализация массивов
    BettaP = np.zeros(nx + 1)
    
    # Первая точка
    Ta1 = (TaP[0] + TaP[1]) / 2
    Te1 = (TeP[0] + TeP[1]) / 2
    Ne1 = (NeP[0] + NeP[1]) / 2
    BettaF = BettaF_coef * (Te1) ** -0.75
    BettaTR = BettaTR_coef * Ne1 * (Te1) ** -4.5

    BettaP[0] = 2 * BettaTR_coef * NeP[0] * (Te1) ** -4.5 + BettaF_coef * (Te1) ** -0.75

    # Расчет в точках 2..nx
    for i in range(1, nx):
        Te1 = TeP[i]
        Ta1 = TaP[i]
        Ne1 = NeP[i]
        BettaF = BettaF_coef * (Te1) ** -0.75  # фото рекомбинация
        BettaTR = BettaTR_coef * Ne1 * (Te1) ** -4.5  # троичная рекомбинация
        BettaP[i] = 2 * BettaTR_coef * NeP[i] * (Te1) ** -4.5 + BettaF_coef * (Te1) ** -0.75

    # Последняя точка
    Ta1 = (TaP[nx - 1] + TaP[nx]) / 2
    Te1 = (TeP[nx] + TeP[nx - 1]) / 2
    Ne1 = (NeP[nx] + NeP[nx - 1]) / 2
    BettaF = BettaF_coef * (Te1) ** -0.75
    BettaTR = BettaTR_coef * Ne1 * (Te1) ** -4.5

    BettaP[nx] = 2 * BettaTR_coef * NeP[nx] * (Te1) ** -4.5 + BettaF_coef * (Te1) ** -0.75
    
    return BettaP