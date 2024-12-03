import numpy as np


def riccati_recursion(P, B, M, N, S):
    return M + P - (S.T + B.T @ P).T @ np.linalg.inv(N + B.T @ P @ B) @ (S.T + B.T @ P) 

class LQR:
    def __init__(self, num_iterations, B, M_final, M, N, S):
        """
        Cost function: J = x'Mx + u'Nu + 2x'Su
        """
        self.num_iterations = num_iterations
        self.M_final = M_final
        self.B = B
        self.M = M
        self.N = N
        self.S = S

        self.riccati_covars = self._make_riccati_covars()

    def _make_riccati_covars(self):
        riccati_covars = [self.M_final]
        for i in range(self.num_iterations-1):
            riccati_covars.append(riccati_recursion(riccati_covars[-1], self.B, self.M, self.N, self.S))
        return riccati_covars[::-1]

    def calc_gain(self, riccati_covar):
        P = riccati_covar
        return np.linalg.inv(self.N + self.B.T @ P @ self.B) @ (self.S.T + self.B.T @ P)
    
    def gain_schedule(self):
        return np.array([self.calc_gain(riccati_covar) for riccati_covar in self.riccati_covars])