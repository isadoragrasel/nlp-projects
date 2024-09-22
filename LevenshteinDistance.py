"""
Problem 2 aims to calculate the minimum edit distance (Levenshtein) between 2 strings. In Levenshtein distance, insertions and
deletions cost 1 and substitutions cost 2.
"""


def problem2(s1, s2):
    m = len(s1) # get length of strings to build matrix
    n = len(s2)
    mtx = [[0] * (n + 1) for _ in range(m + 1)] # initialize matrix with one extra row and column
    # the one extra row and column is used to represent comparisons to an empty string
    for i in range(m + 1):
        mtx[i][0] = i
    for j in range(n + 1):
        mtx[0][j] = j
    # start filling out matrix after extra row and column
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]: # match, so no change
                mtx[i][j] = mtx[i - 1][j -1] + 0
            else: # deletion, insertion, and substitution (costs 2)
                mtx[i][j] = min(mtx[i - 1][j] + 1, mtx[i][j - 1] + 1, mtx[i - 1][j - 1] + 2)
    # print(mtx) to see full matrix and how what the algorithm does is to visit the nearer slots and pick the lower number
    return mtx[m][n]  # min edit distance at the corner of the matrix

