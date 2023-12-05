import math
import random


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


def ud_4_normal_dist(M, D):
    """Calculate coefficients for normal distribution.
    M - math exception
    D - dispersion
    """
    return M, D ** 0.5


def normal_dist_val(u, d):
    return white_noise() * d + u


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


def main():
    M1 = M2 = 0
    while M2 <= M1:
        M1 = float(input("M1 = "))
        M2 = float(input("M2 = "))
    D = float(input("D = "))
    sub = abs(M2 - M1)
    e = 0.01
    if sub < 1:
        e = 0.1 / 10 ** count_unten(sub)
    M = M1 if random.random() >= 0.5 else M2
    vals = []
    u, d = ud_4_normal_dist(M, D)
    flag1 = flag2 = False
    while not (flag1 or flag2) and len(vals) < 1000000:
        vals += [normal_dist_val(u, d)]
        M = math_expect(vals)
        flag1 = abs(M - M1) <= e
        flag2 = abs(M - M2) <= e

    print("Математическое ожидание: ", end="")
    if flag1:
        print("M1", end="")
    elif flag2:
        print("M2", end="")
    else:
        print("undefined", end="")
    print(f" = {M:f}")


if __name__ == "__main__":
    main()
