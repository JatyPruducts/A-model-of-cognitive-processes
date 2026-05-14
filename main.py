# main.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import config as cfg
import model
import scenarios


def run_simulation():
    params = {k: v for k, v in vars(cfg).items() if not k.startswith('__')}

    for sc in scenarios.SCENARIOS:
        print(f"Запуск симуляции: {sc['name']}...")
        y0 = [0.01, 0.01]
        t_span = (cfg.T_START, cfg.T_END)

        # ПРАВКА 1: Увеличение частоты дискретизации до 5000 точек
        t_eval = np.linspace(cfg.T_START, cfg.T_END, 5000)

        sol = solve_ivp(
            fun=model.cognitive_system,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            args=(params, sc),
            max_step=cfg.DT_MAX
        )

        plot_results(sol, sc, params)


def plot_results(solution, scenario, params):
    t = solution.t
    M = solution.y[0]
    A = solution.y[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plt.suptitle(scenario['name'], fontsize=16)

    # --- График 1: Временные ряды ---
    ax1.plot(t, M, label='Память M(t)', color='blue', linewidth=2, zorder=4)
    ax1.plot(t, A, label='Внимание A(t)', color='red', linewidth=2, linestyle='--', zorder=4)

    # Визуализация дистракции (серые зоны)
    for i, (start, dur, amp) in enumerate(scenario['D_schedule']):
        ax1.axvspan(start, start + dur, color='gray', alpha=0.2,
                    label='Помеха D(t)' if i == 0 else "")

    # ПРАВКА 3: Использование fill_between вместо hlines для стимулов памяти
    for i, (start, dur, amp) in enumerate(scenario['I_M_schedule']):
        ax1.fill_between([start, start + dur], -0.05, 0.05, color='blue', alpha=0.2,
                         label='Стимул памяти (Im)' if i == 0 else "")

    ax1.set_xlabel('Время (сек)')
    ax1.set_ylabel('Уровень активности [0..1]')
    ax1.set_ylim(-0.08, 1.05)
    ax1.set_title('Динамика процессов во времени')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize='small')

    # --- График 2: Фазовый портрет ---
    m_range = np.linspace(-0.05, 1.05, 25)
    a_range = np.linspace(-0.05, 1.05, 25)
    MM, AA = np.meshgrid(m_range, a_range)

    empty_scenario = {'I_M_schedule': [], 'I_A_schedule': [], 'D_schedule': []}
    U, V = np.zeros(MM.shape), np.zeros(MM.shape)

    for i in range(MM.shape[0]):
        for j in range(MM.shape[1]):
            derivs = model.cognitive_system(0, [MM[i, j], AA[i, j]], params, empty_scenario)
            U[i, j] = derivs[0]
            V[i, j] = derivs[1]

    ax2.streamplot(MM, AA, U, V, color=(0.5, 0.5, 0.5, 0.4), linewidth=0.8, density=1.2)

    # ПРАВКА 2: Уменьшение толщины линии траектории до 1.5
    ax2.plot(M, A, color='purple', linewidth=1.5, label='Траектория процесса', zorder=5)

    ax2.scatter(M[0], A[0], color='green', s=100, label='Старт', zorder=6)
    ax2.scatter(M[-1], A[-1], color='red', s=150, marker='*', label='Финиш', zorder=6)

    ax2.set_xlabel('Память M')
    ax2.set_ylabel('Внимание A')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title('Фазовое пространство')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()