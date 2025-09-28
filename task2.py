from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import scipy.integrate as spi
from numpy.typing import NDArray


def f(x: float) -> float:
    """Інтегрована функція: f(x) = x^2 (можна замінити на свою)."""
    return x * x


def monte_carlo_integration(func: Callable[[float], float],
                            a: float, b: float, n_points: int,
                            *, seed: int | None = 123) -> float:
    """
    Обчислення інтеграла методом Монте-Карло (hit-or-miss).
    y_max оцінюємо програмно на щільній сітці.
    """

    x_grid: NDArray[np.float64] = np.linspace(a, b, 2000)
    y_grid: NDArray[np.float64] = np.array([func(float(x)) for x in x_grid], dtype=np.float64)
    y_min: float = float(np.min(y_grid))
    y_max: float = float(np.max(y_grid))

    rng = np.random.default_rng(seed)
    area_total = 0.0

    # верхня (позитивна) частина
    if y_max > 0.0:
        xs = rng.uniform(a, b, size=n_points)
        ys = rng.uniform(0.0, y_max, size=n_points)
        under = np.sum(ys <= np.array([func(float(x)) for x in xs], dtype=float))
        p = under / n_points
        area_total += (b - a) * y_max * p

    # нижня (негативна) частина
    if y_min < 0.0:
        xs = rng.uniform(a, b, size=n_points)
        ys = rng.uniform(y_min, 0.0, size=n_points)
        over = np.sum(ys >= np.array([func(float(x)) for x in xs], dtype=float))
        p = over / n_points
        area_total -= (b - a) * (-y_min) * p
    return float(area_total)


def monte_carlo_multiple_experiments(
    func: Callable[[float], float],
    a: float, b: float,
    n_points: int, n_experiments: int,
    *, seed: int | None = 123
) -> Tuple[float, list[float]]:
    """Багаторазові експерименти hit-or-miss. Повертає (середній_результат, список_усіх_результатів)."""
    rng = np.random.default_rng(seed)
    results: list[float] = []
    for _ in range(n_experiments):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        est = monte_carlo_integration(func, a, b, n_points, seed=sub_seed)
        results.append(float(est))

    arr = np.asarray(results, dtype=float)
    avg = float(np.mean(arr))
    return avg, results


def analytical_solution(a: float, b: float) -> float:
    """Аналітичне значення ∫ x^2 dx = x^3/3 | a..b."""
    return (b**3 - a**3) / 3.0


def compare_with_scipy(func: Callable[[float], float], a: float, b: float) -> Tuple[float, float]:
    """Порівняння з scipy.integrate.quad. Повертає (значення, оцінка_помилки)."""
    val, err = spi.quad(lambda x: func(float(x)), a, b)
    return float(val), float(err)


if __name__ == "__main__":
    a: float = 0.0
    b: float = 2.0

    print("=" * 64)
    print("Завдання 2: Інтегрування f(x) = x^2 методом Монте-Карло на [0, 2]")
    print("=" * 64)

    analytical_result: float = analytical_solution(a, b)
    print(f"1) Аналітичний результат:  {analytical_result:.12f}")

    scipy_result, scipy_error = compare_with_scipy(f, a, b)
    print(f"2) SciPy quad:             {scipy_result:.12f}  (reported abs err ~ {scipy_error:.2e})")

    print("\n3) Метод Монте-Карло з різною кількістю точок (hit-or-miss):")
    point_counts: list[int] = [1_000, 10_000, 100_000, 500_000]
    for n_points in point_counts:
        mc_result: float = monte_carlo_integration(f, a, b, n_points, seed=123)
        abs_err: float = abs(mc_result - analytical_result)
        rel_err_pct: float = abs_err / analytical_result * 100.0
        print(f"   N = {n_points:>7,}: MC = {mc_result:10.6f}   "
              f"|MC−exact| = {abs_err:8.6f}  ({rel_err_pct:5.2f}%)")

    print("\n4) 50 експериментів по 10 000 точок (hit-or-miss):")
    avg_result, all_results = monte_carlo_multiple_experiments(f, a, b, 10_000, 50, seed=777)
    abs_err_avg = abs(avg_result - analytical_result)
    rel_err_avg = abs_err_avg / analytical_result * 100.0
    std_dev = float(np.std(all_results, ddof=1))
    print(f"   Середній результат:    {avg_result:.6f}")
    print(f"   Стандартне відхилення: {std_dev:.6f}")
    print(f"   |Avg−exact|:           {abs_err_avg:.6f}  ({rel_err_avg:.3f}%)")

    print("\n5) Залежність точності від кількості експериментів (по 5 000 точок кожен):")
    experiment_counts: list[int] = [1, 10, 50, 100]
    for n_exp in experiment_counts:
        avg_res, _ = monte_carlo_multiple_experiments(f, a, b, 5_000, n_exp, seed=2025)
        err = abs(avg_res - analytical_result)
        rel = err / analytical_result * 100.0
        print(f"   {n_exp:3d} експеримент(ів):  Avg = {avg_res:10.6f}   |Avg−exact| = {err:8.6f}  ({rel:5.2f}%)")
