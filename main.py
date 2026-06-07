import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import config as cfg
import model
import scenarios
from mpl_toolkits.mplot3d import Axes3D


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
    """Смешанная визуализация: 2D временные ряды и 3D фазовая траектория"""
    t = solution.t
    M = solution.y[0]
    A = solution.y[1]

    # Создаем фигуру с двумя подграфиками
    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(scenario['name'], fontsize=16)

    # --- ЛЕВЫЙ ГРАФИК: 2D Временная динамика ---
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Отрисовка переменных
    ax1.plot(t, M, label='Память M(t)', color='blue', linewidth=2, zorder=4)
    ax1.plot(t, A, label='Внимание A(t)', color='red', linewidth=2, linestyle='--', zorder=4)
    
    # Возвращаем визуализацию помехи (серые зоны)
    for i, (start, dur, amp) in enumerate(scenario['D_schedule']):
        ax1.axvspan(start, start+dur, color='gray', alpha=0.2, 
                    label='Помеха D(t)' if i == 0 else "")
    
    # Возвращаем fill_between для стимулов памяти
    for i, (start, dur, amp) in enumerate(scenario['I_M_schedule']):
        ax1.fill_between([start, start + dur], -0.05, 0.05, color='blue', alpha=0.3, 
                         label='Стимул памяти (Im)' if i == 0 else "")

    # Настройка сетки и осей для левого графика
    ax1.set_xlabel('Время (сек)')
    ax1.set_ylabel('Уровень активности [0..1]')
    ax1.set_ylim(-0.08, 1.05)
    ax1.set_title('Динамика процессов во времени')
    ax1.grid(True, which='both', linestyle=':', alpha=0.6) # Явно включаем сетку
    ax1.legend(loc='upper right', fontsize='small')

    # --- ПРАВЫЙ ГРАФИК: 3D Траектория в пространстве (M, A, t) ---
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Рисуем траекторию: по осям X и Y - когнитивные уровни, по Z - время
    ax2.plot(M, A, t, color='purple', linewidth=2, label='Траектория (M, A, t)', zorder=5)
    
    # Рисуем "проекцию" на плоскость (дно графика) для наглядности (t=0)
    ax2.plot(M, A, zs=0, zdir='z', color='black', alpha=0.2, label='Проекция на плоскость MA')

    # Ключевые точки в 3D
    ax2.scatter(M[0], A[0], t[0], color='green', s=100, label='Старт (t=0)')
    ax2.scatter(M[-1], A[-1], t[-1], color='red', s=150, marker='*', label='Финиш (t=end)')

    # Настройка осей и сетки для 3D
    ax2.set_xlabel('Память M')
    ax2.set_ylabel('Внимание A')
    ax2.set_zlabel('Время t (сек)')
    ax2.set_title('Диаграмма, представляющая поведение неавтономной системы')
    ax2.grid(True) # Включаем сетку в 3D
    
    # Начальный угол обзора (можно крутить мышкой)
    ax2.view_init(elev=25, azim=45)
    ax2.legend(loc='lower left', fontsize='small')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()