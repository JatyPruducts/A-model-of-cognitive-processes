import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import config as cfg
import model
import scenarios


def run_simulation():
    # Собираем параметры в словарь для передачи в функцию
    params = {k: v for k, v in vars(cfg).items() if not k.startswith('__')}

    for sc in scenarios.SCENARIOS:
        print(f"Запуск симуляции: {sc['name']}...")

        # Начальные условия (M=0, A=0 - состояние покоя)
        y0 = [0.01, 0.01]

        # Временной интервал
        t_span = (cfg.T_START, cfg.T_END)
        # Точки, где нам нужно решение (для плавных графиков)
        t_eval = np.linspace(cfg.T_START, cfg.T_END, 1000)

        # === РЕШЕНИЕ ОДУ ===
        # Используем метод RK45 (Рунге-Кутта 4-5 порядка)
        sol = solve_ivp(
            fun=model.cognitive_system,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            args=(params, sc),  # Передаем параметры и текущий сценарий
            max_step=cfg.DT_MAX
        )

        # === ВИЗУАЛИЗАЦИЯ ===
        plot_results(sol, sc, params)


def plot_results(solution, scenario, params):
    """Строит графики временных рядов и фазовый портрет"""
    t = solution.t
    M = solution.y[0]
    A = solution.y[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(scenario['name'], fontsize=16)

    # --- График 1: Временные ряды M(t) и A(t) ---
    ax1.plot(t, M, label='Память M(t)', color='blue', linewidth=2)
    ax1.plot(t, A, label='Внимание A(t)', color='red', linewidth=2, linestyle='--')

    # Визуализация помехи (серым фоном)
    for (start, dur, amp) in scenario['D_schedule']:
        ax1.axvspan(start, start + dur, color='gray', alpha=0.3, label='Помеха D(t)')

    # Визуализация стимулов памяти (снизу штрихами)
    for (start, dur, amp) in scenario['I_M_schedule']:
        ax1.hlines(y=-0.05, xmin=start, xmax=start + dur, color='blue', linewidth=4, alpha=0.6)

    ax1.set_xlabel('Время (сек)')
    ax1.set_ylabel('Уровень активности [0..1]')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title('Динамика процессов во времени')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Удаляем дубликаты в легенде (из-за цикла помех)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')

    # --- График 2: Фазовый портрет ---
    # Строим векторное поле для "спокойного" состояния (без внешних стимулов),
    # чтобы видеть, куда система стремится (аттракторы)
    m_range = np.linspace(0, 1, 20)
    a_range = np.linspace(0, 1, 20)
    MM, AA = np.meshgrid(m_range, a_range)

    # Считаем вектора (dM, dA) в каждой точке сетки при I=0, D=0
    empty_scenario = {'I_M_schedule': [], 'I_A_schedule': [], 'D_schedule': []}
    U, V = np.zeros(MM.shape), np.zeros(MM.shape)

    for i in range(MM.shape[0]):
        for j in range(MM.shape[1]):
            # Вызываем функцию модели для каждой точки сетки
            derivs = model.cognitive_system(0, [MM[i, j], AA[i, j]], params, empty_scenario)
            U[i, j] = derivs[0]
            V[i, j] = derivs[1]

    # Рисуем потоки (streamplot)
    ax2.streamplot(MM, AA, U, V, color='gray', linewidth=0.5, density=1.5)

    # Рисуем реальную траекторию симуляции поверх поля
    ax2.plot(M, A, color='purple', linewidth=2.5, label='Траектория системы')
    # Отмечаем начало и конец
    ax2.scatter(M[0], A[0], color='green', s=100, label='Старт')
    ax2.scatter(M[-1], A[-1], color='red', s=100, label='Финиш')

    ax2.set_xlabel('Память M')
    ax2.set_ylabel('Внимание A')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Фазовый портрет (Траектория в пространстве состояний)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()