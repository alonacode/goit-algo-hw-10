import timeit

COINS = [50, 25, 10, 5, 2, 1]


def find_coins_greedy(amount: int, coins=COINS) -> dict[int, int]:
    """Жадібний алгоритм видачі решти для канонічних номіналів.
    Повертає {номінал: кількість}.
    Часова складність: O(k), пам’ять: O(k)."""
    result: dict[int, int] = {}
    for c in sorted(coins, reverse=True):
        if amount <= 0:
            break
        take = amount // c
        if take:
            result[c] = take
            amount -= take * c
    if amount != 0:
        raise ValueError("Суму неможливо видати цими монетами")
    return result


def find_min_coins(amount: int, coins=COINS) -> dict[int, int]:
    """Алгоритм динамічного програмування для мінімізації кількості монет.
    Часова складність: O(amount * k), пам’ять: O(amount)."""
    INF = 10**9
    dp = [(0, 0)] + [(INF, 0) for _ in range(amount)]
    for x in range(1, amount + 1):
        best = (INF, 0)
        for c in coins:
            if c <= x:
                cand = (dp[x - c][0] + 1, c)
                if cand[0] < best[0]:
                    best = cand
        dp[x] = best
    if dp[amount][0] >= INF:
        raise ValueError("Суму неможливо видати цими монетами")
    counts: dict[int, int] = {}
    x = amount
    while x > 0:
        c = dp[x][1]
        counts[c] = counts.get(c, 0) + 1
        x -= c
    return {d: counts[d] for d in sorted(counts)}


if __name__ == "__main__":
    amount = int(input("Сума для видачі решти: "))

    greedy_timer = timeit.Timer(stmt=lambda: find_coins_greedy(amount))
    dp_timer = timeit.Timer(stmt=lambda: find_min_coins(amount))

    REPEAT = 5
    NUMBER = 200

    greedy_times = greedy_timer.repeat(repeat=REPEAT, number=NUMBER)
    dp_times = dp_timer.repeat(repeat=REPEAT, number=NUMBER)

    g_best_ms = (min(greedy_times) / NUMBER) * 1000
    d_best_ms = (min(dp_times) / NUMBER) * 1000
    g_avg_ms = (sum(greedy_times) / (REPEAT * NUMBER)) * 1000
    d_avg_ms = (sum(dp_times) / (REPEAT * NUMBER)) * 1000

    print(f"Жадібний алгоритм : найкраще {g_best_ms:.4f} мс/виклик, середнє {g_avg_ms:.4f} мс/виклик")
    print(f"ДП (мін. монет)  : найкраще {d_best_ms:.4f} мс/виклик, середнє {d_avg_ms:.4f} мс/виклик")

    print("Результат (Жадібний) :", find_coins_greedy(amount))
    print("Результат (ДП)       :", find_min_coins(amount))
