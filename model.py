import numpy as np

def hill_function(x, threshold, n_coeff):
    """
    Реализация функции Хилла (сигмоида).
    x: текущее значение (A или M)
    threshold: порог полунасыщения (theta)
    n_coeff: коэффициент степени (крутизна)
    """
    # Защита от деления на ноль и отрицательных значений
    if x <= 0: return 0.0
    val = (x**n_coeff) / (x**n_coeff + threshold**n_coeff)
    return val

def get_stimulus_value(t, schedule):
    """
    Вычисляет значение входного сигнала в момент времени t.
    schedule: список кортежей [(t_start, t_duration, amplitude), ...]
    """
    value = 0.0
    for (start, duration, amp) in schedule:
        if start <= t <= start + duration:
            value += amp
    return value

def cognitive_system(t, y, params, scenario):
    """
    Система дифференциальных уравнений.
    t: текущее время
    y: вектор состояния [M, A]
    params: словарь параметров из config.py
    scenario: словарь с расписанием стимулов
    """
    M, A = y  # Распаковка вектора состояния

    # Ограничение переменных для физической корректности (клиппинг)
    # Хотя уравнения имеют защиту, при численном решении возможны микро-выходы за 0/1
    M = np.clip(M, 0, 1)
    A = np.clip(A, 0, 1)

    # 1. Расчет значений входных функций на текущий момент
    I_M = get_stimulus_value(t, scenario['I_M_schedule']) # Инфо-стимул
    I_A = get_stimulus_value(t, scenario['I_A_schedule']) # Стимул внимания
    D_t = get_stimulus_value(t, scenario['D_schedule'])   # Дистракция (помехи)

    # 2. Расчет функций обратной связи (Хилла)
    f_A = hill_function(A, params['THETA_M'], params['N_HILL']) # Влияние A на M
    g_M = hill_function(M, params['THETA_A'], params['M_HILL']) # Влияние M на A

    # 3. Уравнение для Памяти (dM/dt)
    # dM = Кодирование - Забывание + Поддержка Вниманием (с насыщением)
    dM_dt = (params['ALPHA'] * I_M * (1 - M)
             - params['BETA'] * M
             + params['GAMMA'] * f_A * (1 - M))

    # 4. Уравнение для Внимания (dA/dt)
    # dA = Стимуляция - Утечка - Дистракция + Мобилизация Памятью (с насыщением)
    dA_dt = (params['DELTA'] * I_A * (1 - A)
             - params['EPSILON'] * A
             - params['ZETA'] * D_t * A
             + params['ETA'] * g_M * (1 - A))

    return [dM_dt, dA_dt]