import numpy as np

def fuNui(TeP, TaP, nx):
    Na_coef = P / kB * Ci
    Nui_coef1 = -2.4e-18 / kB
    Nui_coef2 = kB * 2 * 6.2e18
    dd = P / kB / np.sqrt(2 * 2.4e-18 / me) / Ci / 15.76
    Nuiv0_coef = Num / dd
    Nuiv1_coef = 0.89 * ((3 / 2)**(1 / 3)) * DeltaM**(2 / 3)
    Nuiv2_coef = 2.4e-18 / kB

    Nui = np.zeros(nx + 1)

    # Первая точка
    Ta1 = (TaP[0] + TaP[1]) / 2
    Te1 = (TeP[0] + TeP[1]) / 2
    v_tepl = 6.21e3 * np.sqrt(Te1)
    Nuim = Na_coef / Ta1 * v_tepl * (15.76 + Te1 * Nui_coef2) * np.exp(Nui_coef1 / Te1)
    Nui[0] = Nuiv1_coef * ((Nuiv2_coef / Te1)**(4 / 3)) * ((Nuiv0_coef * Ta1)**(2 / 3)) * Nuim

    for i in range(1, nx):
        Te1 = TeP[i]
        Ta1 = TaP[i]
        v_tepl = 6.21e3 * np.sqrt(Te1)
        Nuim = Na_coef / Ta1 * v_tepl * (15.76 + Nui_coef2 * Te1) * np.exp(Nui_coef1 / Te1)
        Nui[i] = Nuiv1_coef * ((Nuiv2_coef / Te1)**(4 / 3)) * ((Nuiv0_coef * Ta1)**(2 / 3)) * Nuim

    # Последняя точка
    Ta1 = (TaP[nx] + TaP[nx + 1]) / 2
    Te1 = (TeP[nx + 1] + TeP[nx]) / 2
    v_tepl = 6.21e3 * np.sqrt(Te1)
    Nuim = Na_coef / Ta1 * v_tepl * (15.76 + Te1 * Nui_coef2) * np.exp(Nui_coef1 / Te1)
    Nui[nx] = Nuiv1_coef * ((Nuiv2_coef / Te1)**(4 / 3)) * ((Nuiv0_coef * Ta1)**(2 / 3)) * Nuim

    return Nui
