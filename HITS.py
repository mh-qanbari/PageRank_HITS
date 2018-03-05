import numpy as np


class HITS:
    def __init__(self, max_err):
        self.A = np.ndarray((0, 0), dtype=float)
        self.A_t = np.ndarray((0, 0), dtype=float)
        self.h = np.ndarray(0, dtype=float)
        self.a = np.ndarray(0, dtype=float)
        self.maxError = max_err

    def initMat(self, mat, mat_t):
        del self.A
        self.A = mat.copy()
        del self.A_t
        self.A_t = mat_t.copy()
        __n = mat.shape[0]
        del self.h
        self.h = np.ones(__n, dtype=float)
        del self.a
        self.a = np.ones(__n, dtype=float)

    def normalize(self):
        self.A = self.A / self.A.sum(axis=0)
        self.A[np.isnan(self.A)] = 0

    def run(self):
        __i = 0
        while True:
            __i += 1
            __a = self.A_t.dot(self.h)
            __a = __a / np.max(__a)
            __h = self.A.dot(__a)
            __h = __h / np.max(__h)
            __is_converged = self.__getError(__h, __a) < self.maxError
            del self.h
            self.h = __h.copy()
            del self.a
            self.a = __a.copy()
            if __is_converged:
                break
        # self.h = self.h / max(self.h)
        # self.a = self.a / max(self.a)
        return __i

    def __getError(self, h, a):
        return sum(np.abs(self.h - h)) + sum(np.abs(self.a - a))
