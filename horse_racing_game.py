# Horse racing game:
#   At each turn, roll dice, advance horse i one step with probability p[i].
#   First horse to advance n[i] steps wins.

import numpy as np
from numpy.polynomial import polynomial as P
from functools import reduce
from fractions import Fraction
import sympy
import itertools
from multiprocessing import Pool

def p_win1(p, n):
    """Probability of first horse winning."""
    g1 = P.polypow([Fraction(0), p[0]], n[0] - 1) / np.math.factorial(n[0] - 1)
    g2 = reduce(P.polymul,
                ([pi ** k / np.math.factorial(k) for k in range(ni)]
                 for pi, ni in zip(p[1:], n[1:])))
    g = reduce(P.polymul, [[p[0]], g1, g2])
    return sum(c * np.math.factorial(m) for m, c in enumerate(g))

def p_win(p, n):
    """Probability distribution of each horse winning."""
    return [p_win1(p[j:] + p[:j], n[j:] + n[:j]) for j in range(len(n))]

# Scratch horses prior to race:
#   Roll dice, scratch horse i with probability ps[i].
#   Repeat until s distinct horses have been scratched.
#   Race remaining len(p)-s horses with normalized probabilities pr[i].

def p_scratch(q):
    """Probability of scratching subset with dice probabilities q[:]."""
    a = sympy.eye(2 ** len(q))
    for i in range(2 ** len(q) - 1):
        for h, p_roll in enumerate(q):
            j = i | (2 ** h)
            a[i, j] -= p_roll
    return sympy.linsolve((a, sympy.eye(2 ** len(q))[:, -1])).args[0][0]

def all_races(ps, pr, s):
    """All races with s scratches, scratching with ps, racing with pr."""
    for scratched in itertools.combinations(range(len(ps)), s):
        racing = [horse for horse in range(len(p)) if not horse in scratched]
        p_total = sum(pr[horse] for horse in racing)
        yield (float(p_scratch([ps[horse] for horse in scratched])),
               racing,
               [float(pr[horse] / p_total) for horse in racing])

# Pre-compute all scratches of s=4 horses.
p = [Fraction(n, 36) for n in [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]]
races = list(all_races(p, p, 4))

def eval_board(n):
    """Distribution of each horse winning considering scratched races."""
    p_overall = [0] * len(n)
    for p_race, racing, p_dice in races:
        for horse, q in zip(racing,
                            p_win(p_dice, [n[horse] for horse in racing])):
            p_overall[horse] += p_race * q
    return (n, p_overall)

def all_boards(len_p, n_max):
    """Generate all symmetric/monotonic boards up to n_max steps."""
    m = (len_p + 1) // 2
    for board in itertools.combinations(range(n_max + m - 1), m):
        n = np.cumsum(np.diff(
            [-1] + list(board) + [n_max + m - 1])[0:-1] - 1) + 1
        yield list(n) + list(reversed(n[:-1]))

if __name__ == '__main__':
    with Pool(6) as pool:
        for n, p_overall in pool.imap(eval_board, all_boards(len(p), 17)):
            print([n, p_overall])
