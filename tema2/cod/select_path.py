import sys
import numpy as np
import pdb


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_greedy_path(E):
    line = 0
    col = np.argmin(E[0])
    path = [(line, col)]
    for i in range(1, E.shape[0]):
        line = i
        col_prev = path[-1][1]
        if col_prev == 0:
            opt = min(E[i, col_prev], E[i, col_prev + 1])
            col = col_prev if opt == E[i, col_prev] else col_prev + 1
        elif col_prev == len(E[0]) - 1:
            opt = min(E[i, col_prev], E[i, col_prev - 1])
            col = col_prev if opt == E[i, col_prev] else col_prev - 1
        else:
            opt = min(E[i, col_prev + 1], E[i, col_prev], E[i, col_prev - 1])
            if opt == E[i, col_prev]:
                col = col_prev
            elif opt == E[i, col_prev - 1]:
                col = col_prev - 1
            else:
                col = col_prev + 1

        path.append((line, col))

    '''
    inf_col = np.ones((E.shape[0], 1)) * np.inf
    greedy_E = np.concatenate((inf_col, E), axis=1)
    greedy_E = np.concatenate((greedy_E, inf_col), axis=1)
    line = 0
    col = np.argmin(greedy_E[0])
    path = [(line, col)]
    for i in range(1, greedy_E.shape[0]):
        line = i
        col_prev = path[-1][1]
        opt = min(greedy_E[i, col_prev - 1], greedy_E[i, col_prev], greedy_E[i, col_prev + 1])
        if opt == greedy_E[i, col_prev - 1]:
            path.append((line, col_prev - 1))
        elif opt == greedy_E[i, col_prev]:
            path.append((line, col_prev))
        else:
            path.append((line, col_prev + 1))
    '''
    return path


def select_dynamic_programming_path(E):
    dp = np.zeros(E.shape)
    dp[0] = E[0]

    for i in range(1, E.shape[0]):
        for j in range(E.shape[1]):
            if j == 0:
                dp[i, j] = min(dp[i - 1, j], dp[i - 1, j + 1]) + E[i, j]
            elif j == E.shape[1] - 1:
                dp[i, j] = min(dp[i - 1, j - 1], dp[i - 1, j]) + E[i, j]
            else:
                dp[i, j] = min(dp[i - 1, j - 1], dp[i - 1, j], dp[i - 1, j + 1]) + E[i, j]
    path = []
    row = len(dp) - 1
    col = np.argmin(dp[len(dp) - 1])
    path.append((row, col))
    while row != 0:
        row -= 1
        if col == 0:
            opt = min(dp[row, col], dp[row, col + 1])
            if opt == dp[row, col]:
                path.append((row, col))
            elif opt == dp[row, col + 1]:
                path.append((row, col + 1))
                col = col + 1
        elif col == len(dp[0]) - 1:
            opt = min(dp[row, col - 1], dp[row, col])
            if opt == dp[row, col]:
                path.append((row, col))
            elif opt == dp[row, col - 1]:
                path.append((row, col - 1))
                col = col - 1
        else:
            opt = min(dp[row, col - 1], dp[row, col], dp[row, col + 1])
            if opt == dp[row, col]:
                path.append((row, col))
            elif opt == dp[row, col - 1]:
                path.append((row, col - 1))
                col = col - 1
            else:
                path.append((row, col + 1))
                col = col + 1

    return path[::-1]


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)