import math
import random
import time
import matplotlib.pyplot as plt


def math_expect(s):
    """Calculate math expectation"""
    return sum(s) / len(s)


def dispersion(s, M=None):
    """Calculate dispersion.
    Use M - as math expectation if it's known,
    else calculate it.
    """
    if M is None:
        M = math_expect(s)
    n = len(s)
    D = 0
    for i in range(n):
        D += (s[i] - M) ** 2
    return D / n


def white_noise():
    s = 0
    for i in range(12):
        s += random.random()
    return s - 6


def ud_4_lognormal_dist(M, D):
    """Calculate coefficients for lognormal distribution.
    M - math exception
    D - dispersion
    """
    d = math.log(D / math.exp(2 * math.log(M)) + 1)
    u = math.log(M) - d / 2
    return u, d ** 0.5


def normal_dist_val(u, d):
    return white_noise() * d + u


def lognormal_dist_val(u, d):
    return math.exp(normal_dist_val(u, d))


def count_unten(n: float, min_count=5) -> int:
    """Return the number of tens not in number
    0.1234 -> 4; 0.11 -> 2; 0.1 -> 1
    """
    i = 0
    if n == 0:
        return i
    flag = False
    while True:
        n *= 10
        mod = int(n) % 10
        if not flag and mod != 0:
            flag = True
        if flag and mod == 0:
            break
        i += 1
        if i >= min_count:
            break
    return i


def identify_random_process(M1: float, M2: float, data: list):
    """M1 - math exception 1
    M2 - math exception 2
    return math exception for data and determine to whom to relate
    ans == 0 - M is M1
    ans == 1 - M is M2
    ans == 2 - M is undefined"""
    sub = abs(M2 - M1)
    e = 0.01
    if sub < 1:
        e = 0.1 / 10 ** count_unten(sub)
    flag1 = flag2 = False
    vals = []
    data = data.copy()
    while not flag1 and not flag2 and len(data) > 0:
        vals.append(data.pop())
        M = math_expect(vals)
        flag1 = abs(M - M1) <= e
        flag2 = abs(M - M2) <= e
    if flag1:
        ans = 0
    elif flag2:
        ans = 1
    else:
        ans = 3
    return ans, M


def input_interval(**kwargs):
    a = kwargs.get("a")
    b = kwargs.get("b")
    text1 = kwargs.get("text1")
    text2 = kwargs.get("text2")
    sign = kwargs.get("sign")
    if a is None:
        a = 0
    if b is None:
        b = 0
    if text1 is None:
        text1 = ""
    if text2 is None:
        text2 = ""
    if sign is None:
        sign = lambda a, b: b > a
    while not sign(a, b):
        a = float(input(text1))
        b = float(input(text2))
    return a, b


def main():
    M1, M2 = input_interval(text1="M1 = ", text2="M2 = ")
    D1, D2 = input_interval(text1="D1 = ", text2="D2 = ")
    M = M1 if random.random() >= 0.5 else M2
    list_T = []
    list_D = []
    D = D1
    while D <= D2:
        list_D += [D]
        u, d = ud_4_lognormal_dist(M, D)
        data = [lognormal_dist_val(u, d) for i in range(1000)]
        t0 = time.time()
        ans, M = identify_random_process(M1, M2, data)
        list_T += [time.time() - t0]
        D += 1

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.plot(list_D, list_T, 'o', list_D, list_T, '-', color='purple')
    axs.set_title('График зависимости времени идентификации от дисперсии')
    axs.set_xlabel('Дисперсия, D')
    axs.set_ylabel('Время, t (s)')
    axs.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
