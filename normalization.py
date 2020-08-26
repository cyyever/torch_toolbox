import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def row_norms(mat):
    return np.linalg.norm(mat, 2, axis=1)


def column_norms(mat):
    return np.linalg.norm(mat, 2, axis=0)


def normalize_for_heatmap(mat):
    rn = 1 / (row_norms(mat) ** 0.5)
    cn = 1 / (column_norms(mat) ** 0.5)
    mat = np.matmul(np.diag(rn), mat)
    mat = np.matmul(mat, np.diag(cn))
    return mat
